# RetinaChlng

Retina segmentation project. Database is available at: https://drive.grand-challenge.org/

Project progress and journal: ProjectProgress.pptx

Use main.py to train and evaluate NNs. The jupyter notebook files were only used for drafts and are not updated. 

Custom binary loss functions  (written as compatible pytorch classes)         : m_loss_functionals.py <br>
Custom NN architectures (UNet, ResNet-UNet, HighResUNet,....)                 : m_network_architectures.py <br>
Custom image transformation(written as compatible torchvision transformations): m_transforms.py <br>
Auxiliary dataset handling functions and visualization functions              : m_network_architectures.py <br>
