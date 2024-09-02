import json
import os
import torch
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from DynamicCode.train import get_dataset, initialize_params, initialize_optimizer, get_batch, \
    report_progress, get_loss, initialize_per_timestep, initialize_post_first_timestep
from tqdm import tqdm
from DynamicCode.external import densify, calc_psnr, calc_ssim
from DynamicCode.helpers import params2cpu, save_params, l1_loss_v1, params2rendervar, setup_camera
from DynamicCode.visualize import init_camera, render, visualize, rgbd2pcd
from PIL import Image
import open3d as o3d
import time
from Models.transModel import DeformationNetworkBilinearCombination, DeformationNetworkCompletelyConnected, \
    DeformationNetworkSeparate
import gc
import matplotlib.pyplot as plt
import wandb
import copy
from random import randint


class trainer:

    def __init__(self, version, epochs=1000, lr=0.001, model=0, pickle_path=os.path.join('saved_model')):
        assert model >= 0 and model < 3 and type(model) == int
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # use basketball self.base
        self.fps = 20
        self.base = 'basketball'
        self.epochs = epochs
        self.lr = lr
        if model == 0:
            self.model_mlp = DeformationNetworkSeparate().cuda()
            self.model_name = 'separate'
        elif model == 1:
            self.model_mlp = DeformationNetworkCompletelyConnected().cuda()
            self.model_name = 'compl_conn'
        else:
            self.model_mlp = DeformationNetworkBilinearCombination().cuda()
            self.model_name = 'bilinear'
        self.pickle_path = pickle_path
        assert type(version) == str
        self.version = version
        self.start_time = 0


    def quaternion_raw_mult(self, a: torch.Tensor, b: torch.Tensor): # from pytorch3d library
        aw, ax, ay, az = torch.unbind(a, -1)
        bw, bx, by, bz = torch.unbind(b, -1)
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack((ow, ox, oy, oz), -1)
    def loss_after_init(self, params, curr_data, variables):  # change of the loss function after using initialization iteration
                                                              # new loss function only calculates l1 loss.
        losses = {}

        rendervar = params2rendervar(params)
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        curr_id = curr_data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        losses['im'] = l1_loss_v1(im, curr_data['im'])
        variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

        loss_weights = {'im': 1.0}
        loss = loss_weights['im'] * losses['im']
        return loss, variables

    def scene_info_from_model(self, render_var, t):  # helper function for the get_image_from_pcd function
                                                     # calcualtes new scene using the model and changing the means and rotation
        with torch.no_grad():
            delta_means, delta_rot = self.model_mlp(render_var['means3D'], render_var['rotations'], t=t)
            rendervar = render_var.copy()
            rendervar['means3D'] = torch.add(render_var['means3D'], delta_means)
            rendervar['rotations'] = self.quaternion_raw_mult(delta_rot, rendervar['rotations'])
        return rendervar

    def get_image_from_pcd(self, scene_info, meta, max_t, test=False):
        # based on the visualization function from the dynamic3DGaussian code
        # we take snapshots for a certain camera angle and save it in the test_pics folder
        # rendering is done by thegaussian renderer referenced in the original dynamic3DGaussian repository
        if test:
            type = 'test'
        else:
            type = 'train'
        w2c = np.copy(np.array(meta['w2c'][0][0]))
        intr_k = np.copy(np.array(meta['k'][0][0]))

        im, depth = render(w2c, intr_k, self.scene_info_from_model(scene_info, 0))

        init_pts, init_cols = rgbd2pcd(im, depth, w2c, intr_k, show_depth=False)

        pcd = o3d.geometry.PointCloud()
        pcd.points = init_pts
        pcd.colors = init_cols

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=meta['w'], height=meta['h'], visible=False)
        vis.add_geometry(pcd)
        view_k = intr_k * 1
        view_k[2, 2] = 1
        view_control = vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        cparams.extrinsic = w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height = meta['h']
        cparams.intrinsic.width = meta['w']
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
        render_options = vis.get_render_option()
        render_options.point_size = 1
        render_options.light_on = True
        t = 0

        if not os.path.exists(f'./{type}_pics/'):
            os.mkdir(f'./{type}_pics/')
        if test:
            if not os.path.exists(f'./{type}_pics/{self.version}_{self.model_name}_frames_{max_t}'):
                os.mkdir(f'./{type}_pics/{self.version}_{self.model_name}_frames_{max_t}')
        else:
            if not os.path.exists(f'./{type}_pics/{self.version}_{self.model_name}_{self.lr}'):
                os.mkdir(f'./{type}_pics/{self.version}_{self.model_name}_{self.lr}')
            if not os.path.exists(f'./{type}_pics/{self.version}_{self.model_name}_{self.lr}/{type}{max_t}'):
                os.mkdir(f'./{type}_pics/{self.version}_{self.model_name}_{self.lr}/{type}{max_t}')


        while t <= max_t:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            intr_k = view_k / 1
            intr_k[2, 2] = 1
            w2c = cam_params.extrinsic

            cam_params = view_control.convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

            im, depth = render(w2c, intr_k, self.scene_info_from_model(scene_info, t))
            pts, cols = rgbd2pcd(im, depth, w2c, intr_k, show_depth=False)

            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)
            if not vis.poll_events():
                break
            vis.update_renderer()

            if test:
                vis.capture_screen_image(
                    f'./{type}_pics/{self.version}_{self.model_name}_frames_{max_t}/{type}ing{t}.png',
                    do_render=False)
            else:
                vis.capture_screen_image(
                    f'./{type}_pics/{self.version}_{self.model_name}_{self.lr}/{type}{max_t}/{type}ing{t}.png',
                    do_render=False)
            t += 1
        vis.destroy_window()
        del view_control
        del vis
        del render_options

        return

    def train_gaussian(self, max_t):
        # training function
        # separates between initial learning and successive learning
        # initial learning is based on the original dynamic3DGaussian method
        # successive learning iteration are done as described in the white paper
        # for sinmplicity, we have tried to use the same dictionary formats for our iterations as well

        # just initializing the first gaussian splats self.based on the dynamic gaussian code
        md_train = json.load(open(f'dynamic_data/{self.base}/train_meta.json', 'r'))
        if max_t > len(md_train['fn']):
            max_t = len(md_train['fn'])
        params_train, variables_train = initialize_params(self.base, md_train)
        optimizer_first = initialize_optimizer(params_train, variables_train)
        output_params_first = []
        # wandb.init(
        #     project='learning without 4d gaussian',
        #     config={
        #         'group': f'trial {max_t} frames. {self.epochs} epochs per timestep, 9 seconds',
        #         'self.lr': self.lr,
        #         'architecture': self.model_name,
        #         'time_max': max_t,
        #         'epochs': self.epochs,
        #         'version': self.version,
        #     }
        # )
        self.start_time = time.time()
        for t in range(max_t):
            gc.collect()
            torch.cuda.empty_cache()

            self.model_mlp.train()
            is_initial_timestep = (t == 0)
            dataset_first = get_dataset(t, md_train, self.base)
            todo_dataset = []
            num_iter_per_timestep = 500 if is_initial_timestep else self.epochs
            if is_initial_timestep:
                progress_bar = tqdm(range(num_iter_per_timestep), desc='gt_gaussian')
            else:
                progress_bar = tqdm(range(num_iter_per_timestep), desc=f'training_mlp {t} out of {max_t}')
            for i in range(num_iter_per_timestep):
                curr_data = get_batch(todo_dataset, dataset_first)
                if is_initial_timestep:
                    loss_first, variables_train = get_loss(params_train, curr_data, variables_train,
                                                           is_initial_timestep)

                    loss_first.backward(retain_graph=False)
                else:
                    delta_mean, delta_quat = self.model_mlp(params_train['means3D'],
                                                            params_train['unnorm_rotations'],
                                                            t=t)
                    old_means = params_train['means3D'].clone()
                    params_train['means3D'] = old_means + delta_mean
                    old_quats = params_train['unnorm_rotations'].clone()
                    params_train['unnorm_rotations'] = self.quaternion_raw_mult(delta_quat, old_quats)


                    loss_first, _ = self.loss_after_init(params_train, curr_data, variables_train)
                    loss_first.backward(retain_graph=False)

                with (torch.no_grad()):

                    if is_initial_timestep:
                        report_progress(params_train, dataset_first[0], i, progress_bar, every_i=10)
                        params_train, variables_train = densify(params_train, variables_train,
                                                                optimizer_first, i)
                        optimizer_first.step()
                        optimizer_first.zero_grad(set_to_none=True)
                        im_log, _, _, = Renderer(raster_settings=dataset_first[0]['cam'])(
                            **params2rendervar(params_train))
                        output_params_first.append(params2cpu(params_train, is_initial_timestep))
                    else:
                        report_progress(params_train, dataset_first[0], i, progress_bar, every_i=10)
                        optimizer_first.step()
                        optimizer_first.zero_grad(set_to_none=True)
                        im_log, _, _, = Renderer(raster_settings=dataset_first[0]['cam'])(
                            **params2rendervar(params_train))
                        output_params_first.append(params2cpu(params_train, is_initial_timestep))

                    im_log = torch.exp(params_train['cam_m'][dataset_first[0]['id']])[:, None, None] * im_log + \
                             params_train['cam_c'][dataset_first[0]['id']][:, None, None]
                    psnr_res = calc_psnr(im_log, dataset_first[0]['im'])

                    # wandb.log({f'epoch{t}': i + 1, f'loss{t}': loss_first.item(),
                    #            'psnr 0': psnr_res[0],
                    #            'psnr 1': psnr_res[1],
                    #            'psnr 2': psnr_res[2],
                    #            'psnr mean': psnr_res.mean(),
                    #            'ssim': calc_ssim(im_log, dataset_first[0]['im'])})

                    if not is_initial_timestep:
                        params_train['means3D'] = old_means.clone()
                        params_train['unnorm_rotations'] = old_quats.clone()
                        del old_quats, old_means
                    del im_log
            progress_bar.close()


            if is_initial_timestep:
                variables_train = initialize_post_first_timestep(params_train, variables_train,
                                                                 optimizer_first)
                params_train, variables_train = initialize_per_timestep(params_train, variables_train,
                                                                        optimizer_first)
                del optimizer_first
                params_train_h = {}
                variables_train_h = {}
                for k, v in params_train.items():
                    v.grad = None
                    params_train_h[k] = v.clone().detach().requires_grad_(False)
                for k, v in variables_train.items():
                    if type(v) == torch.Tensor:
                        v.grad = None
                        variables_train_h[k] = v.clone().detach().requires_grad_(False)
                    else:
                        variables_train_h[k] = v
                del params_train
                del variables_train
                params_train = copy.deepcopy(params_train_h)
                variables_train = copy.deepcopy(variables_train_h)
                del params_train_h
                del variables_train_h

                optimizer_first = torch.optim.Adam(self.model_mlp.parameters(), lr=self.lr, weight_decay=1e-15)
                optimizer_first.zero_grad(set_to_none=True)
            del loss_first
            gc.collect()
            torch.cuda.empty_cache()
            # evaluating
            with torch.no_grad():
                self.get_image_from_pcd(params2rendervar(params_train), md_train, t, test=False)
        # wandb.log({'runtime': time.time() - self.start_time})
        # wandb.finish()
        torch.save(self.model_mlp.state_dict(), f'{self.pickle_path}/{self.version}_{self.model_name}_{self.lr}.pt')
        return output_params_first

        # based on the visualize class of dynamic 3d gaussians

    def get_dataset_test(self, t, md, seq):  # do not need the seg images from the dataset for testing
        dataset = []
        for c in range(len(md['fn'][t])):
            w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
            cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
            fn = md['fn'][t][c]
            im = np.array(copy.deepcopy(Image.open(f"dynamic_data/{seq}/ims/{fn}")))
            im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
            dataset.append({'cam': cam, 'im': im, 'id': c})
        return dataset

    def test_gaussian(self, max_t, model=None): # separate function for testing
                                                # ieratrions are pretty similar to training gaussians
                                                # but there are slight modifications.
                                                # for overview, we have defined another function
        if model is None:
            mlp_mod = self.model_mlp
        else:
            assert type(model) is str
            if 'bilinear' in model:
                mlp_model = DeformationNetworkBilinearCombination()
                mlp_model.load_state_dict(torch.load(f'{self.pickle_path}/{model}'))
            elif 'separate' in model:
                mlp_model = DeformationNetworkSeparate()
                mlp_model.load_state_dict(torch.load(f'{self.pickle_path}/{model}'))
            elif 'compl' in model:
                mlp_model = DeformationNetworkCompletelyConnected()
                mlp_model.load_state_dict(torch.load(f'{self.pickle_path}/{model}'))
            else:
                raise RuntimeError('model name format is not supported')

        md_test = json.load(open(f'dynamic_data/{self.base}/test_meta.json', 'r'))
        md_init = json.load(open(f'dynamic_data/{self.base}/train_meta.json', 'r'))
        if max_t > len(md_test['fn']):
            max_t = len(md_test['fn'])
        params_test, variables_test = initialize_params(self.base, md_test)
        # optimizer only needed for initialization of Gaussians
        optimizer_init = initialize_optimizer(params_test, variables_test)
        output_params = []
        json_result = []

        for t in range(max_t):

            is_initial_timestep = (t == 0)
            if is_initial_timestep:
                dataset = get_dataset(t, md_init, self.base)
            else:
                dataset = self.get_dataset_test(t, md_test, self.base)
            todo_dataset = []

            if is_initial_timestep:
                progress_bar = tqdm(range(1000), desc='gt_gaussian')

                for i in range(1000):
                    curr_data = get_batch(todo_dataset, dataset)
                    loss_first, variables_test = get_loss(params_test, curr_data, variables_test,
                                                          is_initial_timestep)

                    loss_first.backward(retain_graph=False)
                    with (torch.no_grad()):
                        report_progress(params_test, dataset[0], i, progress_bar, every_i=100)

                        params_test, variables_test = densify(params_test, variables_test,
                                                              optimizer_init, i)
                    optimizer_init.step()
                    optimizer_init.zero_grad(set_to_none=True)
                progress_bar.close()
                output_params.append(params2cpu(params_test, True))
                variables_test = initialize_post_first_timestep(params_test, variables_test, optimizer_init)

            else:
                with torch.no_grad():
                    self.model_mlp.eval()
                    # calculate loss for each timestep except 0
                    curr_data = get_batch(todo_dataset, dataset)

                    delta_mean, delta_quat = self.model_mlp(params_test['means3D'],
                                                            params_test['unnorm_rotations'], t=t)

                    old_means = params_test['means3D'].clone()
                    params_test['means3D'] = old_means + delta_mean
                    old_quats = params_test['unnorm_rotations'].clone()
                    params_test['unnorm_rotations'] = self.quaternion_raw_mult(delta_quat, old_quats)

                    loss_first, _ = self.loss_after_init(params_test, curr_data, variables_test)

                    psnr_mean_t = 0
                    ssim_mean_t = 0
                    for i in range(len(dataset)):
                        im_log, _, _, = Renderer(raster_settings=dataset[i]['cam'])(**params2rendervar(params_test))
                        im_log = torch.exp(params_test['cam_m'][dataset[i]['id']])[:, None, None] * im_log + \
                                 params_test['cam_c'][dataset[i]['id']][:, None, None]
                        psnr_res = calc_psnr(im_log, dataset[i]['im'])
                        psnr_mean_t += psnr_res.mean()
                        ssim_mean_t += calc_ssim(im_log, dataset[i]['im'])
                        del im_log
                    output_params.append(params2cpu(params_test, False))
                    json_result.append(
                        {'time': t, 'psnr': psnr_mean_t / len(dataset), 'ssim': ssim_mean_t / len(dataset)})
                    params_test['means3D'] = old_means.clone()
                    params_test['unnorm_rotations'] = old_quats.clone()
                    del delta_mean, delta_quat
        self.get_image_from_pcd(params2rendervar(params_test), md_test, max_t, test=True)
        save_params(output_params, self.base, 'exp')
        return json_result
