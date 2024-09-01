<h1>Dynamic Gaussian method of moving Gaussian.
</h1>

Execute over commandline with 'python main.py [OPTION]'

Options include <br />
&emsp; train mlp with bilinear architecture: '-trb [learning rate], --train_mlp_bilinear [learning rate]' <br />
       &emsp; &emsp;  this option  is to train the MLP with a specified learning rate as described in the paper with the bilinear architecture.<br />
    &emsp; train mlp with separate architecture: '-trs [learning rate], --train_mlp_separate'<br />
        &emsp; &emsp; this option is to train the MLP with a specified learning rate as described in the paper with the separate network architecture.<br />
    &emsp; render images from a trained model: '-test [model_name], --test_mlp [model_name]'<br />
        &emsp; &emsp; Rander images from the test angle from the basketball dataset. the state dict of the model should be in the saved_model folder. Additionally the name of the file containing the state dict should include separate, compl, or bilinear to indicate which architecture is used<br />
    &emsp; train mlp with separate architecture: '-trall [learning rate], --train_mlp_all [learning rate]'<br />
        &emsp; &emsp; this option is to train all 3 architectures described in the paper with the specified learning rate.<br />
<br />
For each training option, the hyperparameters are already preset to 200 epochs. Additionally, for the train and test options, the total amount is set to 50. Both of these settings can be changed in the main file.<br />
For the training options, for each time step iteration, as described in the paper, the images are rendered and saved in the train_pics folder.<br />
Similarly for the testing option, the rendered images are saved in the test_pics folder.<br />
<br />
<h1>Prerequisites</h1>
This code and project has as its foundation the Dynamic 3D Gaussian method introduced by Luiten etal. Their code can be seen here:<a href="https://github.com/JonathonLuiten/Dynamic3DGaussians"> Dynamic 3D Gaussians</a> <br />

We have not uploaded the raw data for the Panoptics basketball dataset. The basketball dataset can be downloaded from the Dynamic 3D Gaussians GitHub. Add the dynamic_data Folder. <br />
Our dependencies can be seen in the environment.yml file. This file can be used to set up an anaconda environment. For the rendering, we used the diff-gauss-rasterization-w-depth library. This setup of this library is also described in the Dynamic 3D Gaussians repository.
    

