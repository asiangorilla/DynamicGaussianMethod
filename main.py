from train_4dGaussians import trainer
import argparse
import os

max_t = 50
pickle_path = os.path.join('saved_model')
output_path_train = os.path.join('train_pics')
output_path_test = os.path.join('test_pics')

parser = argparse.ArgumentParser()

parser.add_argument('-trb', '--train_mlp_bilinear', help='Train mlp with bilinear network. Additionally, you can '
                                                         'add a perfered learning rate', nargs=1, default=None,
                    type=float)

parser.add_argument('-trc', '--train_mlp_connected', help='Train mlp with connected network. Additionally, you can '
                                                          'add a perfered learning rate', nargs=1, default=None,
                    type=float)

parser.add_argument('-trs', '--train_mlp_separate', help='Train mlp with separate network. Additionally, you can '
                                                         'add a perfered learning rate', nargs=1, default=None,
                    type=float)
parser.add_argument('-trall', '--train_mlp_all', help='Train mlp with all 3 networks. Additionally, you can '
                                                    'add a perfered learning rate', nargs=1, default=None, type=float)

parser.add_argument('-test', '--test_mlp',
                    help='render images for the evaluation camera angle. Specify the model name from the saved_model folder',
                    nargs=1)

args = parser.parse_args()

if not os.path.exists(output_path_train):
    os.mkdir(output_path_train)

if not os.path.exists(output_path_test):
    os.mkdir(output_path_test)

if args.train_mlp_bilinear is not None:
    assert type(args.train_mlp_bilinear[0]) is float, 'learning rate must be a float'
    trainer = trainer(version='training_bi', epochs=200, model=2, lr=args.train_mlp_bilinear[0])
    trainer.train_gaussian(max_t=max_t)
elif args.train_mlp_separate is not None:
    assert type(args.train_mlp_separate[0]) is float, 'learning rate must be a float'
    trainer = trainer(version='training_separate', epochs=200, model=0, lr=args.train_mlp_separate[0])
    trainer.train_gaussian(max_t=max_t)
elif args.train_mlp_connected is not None:
    assert type(args.train_mlp_separate[0]) is float, 'learning rate must be a float'
    trainer = trainer(version='training_connected', epochs=200, model=1, lr=args.train_mlp_connected[0])
    trainer.train_gaussian(max_t=max_t)
elif args.train_mlp_all is not None:
    for mod in range(3):
        trainer = trainer(version='training_all', epochs=200, model=mod, lr=args.test_mlp[0])
        trainer.train_gaussian(max_t=max_t)
elif args.test_mlp is not None:
    model_name = args.test_mlp[0]
    assert type(model_name) == str, 'input arg must be a string'
    assert os.path.exists(f'{pickle_path}/{model_name}'), 'model must be in the saved_model folder'
    tester = trainer(version='testing', epochs=10, model=0, lr=0.01)
    tester.test_gaussian(max_t, model=model_name)
else:
    print('no option specified. Specify an option as referred to in the README.')