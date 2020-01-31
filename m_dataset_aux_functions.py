from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torchvision


def get_mask_path(x_image_path, y_train_dir):
    """Get the path of the segmented training file for the path of an original image. \n

    Keyword arguments:
    path -- a pathlib Path to the original data image.

    Output
    SegmentedMaskPath -- a pathlib Path to the matching segmeneted training image.
    """
    x = x_image_path
    y_ShortFilename = x.stem.replace('training', 'manual1') + '.gif'
    y_path = y_train_dir / y_ShortFilename
    return y_path


def get_screen_path(x_image_path, screen_dir):
    """Get the path of the segmented training file for the path of an original image. \n

    Keyword arguments:
    path -- a pathlib Path to the original data image.

    Output
    SegmentedMaskPath -- a pathlib Path to the matching segmeneted training image.
    """
    x = x_image_path
    screen_ShortFilename = x.stem.replace('training', 'training_mask') + '.gif'
    screen_path = screen_dir / screen_ShortFilename
    return screen_path


class Dataset(BaseDataset):
    CLASSES_pool = ['no', 'yes']

    def __init__(
            self,
            image_paths,
            screen_paths,
            mask_paths=None,
            classes=['yes'],
            augmentation=None,
            preprocessing=None,
            size=None,
            interpolation=2,  # 0 means no interpolation. Mask will remain binary.
            ImageNetNorm=True
    ):

        if not mask_paths:
            self.istest = True
        else:
            self.istest = False

        self.ids = [int(file_path.stem[:2]) for file_path in
                    image_paths]  # the id numbers provided for each sample (image, mask, screen)
        self.images = [str(image_path.absolute()) for image_path in image_paths]  # a list of paths to images.
        self.screens = [str(image_path.absolute()) for image_path in screen_paths]  # a list of paths to screens.

        if mask_paths:
            self.masks = [str(image_path.absolute()) for image_path in mask_paths]  # a list of paths to masks.
        self.class_values = [self.CLASSES_pool.index(cls.lower()) for cls in classes]
        self.size = size
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.interpolation = interpolation
        self.ImageNetNorm = ImageNetNorm
    def __getitem__(self, i):
        # load image, screen and mask (target). The tiff images open only with Pillow version 5.2.0
        image = Image.open(self.images[i])
        screen = Image.open(self.screens[i])
        if not self.istest:
            mask = Image.open(self.masks[i])
        # Basic preprocessing:
        Flip = transforms.RandomHorizontalFlip(p=0.5)
        CenterCrop = transforms.CenterCrop(self.size)

        ToTensor = transforms.ToTensor()  # this normalizies to [0,1] range, so must appear before the ImageNet normalization
        BasicTransforms = list([
            #                                 Resize,
            #                                 CenterCrop,
            ToTensor,
        ])
        ImageTransforms = list([
            #                                 Resize,
            #                                 CenterCrop,
            ToTensor,
        ])
        if isinstance(self.size, int) and not isinstance(self.size, bool):
            Resize = transforms.Resize((self.size, self.size),
                                       self.interpolation)  # the zero is extremely important, or it will change the values
            BasicTransforms.insert(0, Resize)
            ImageTransforms.insert(0, Resize)
        if self.ImageNetNorm:
            Normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ImageTransforms.append(Normalize)
        ImageTransformer = transforms.Compose(ImageTransforms)
        MaskTransformer = transforms.Compose(BasicTransforms)

        image = ImageTransformer(image)
        screen = MaskTransformer(screen)

        if not self.istest:
            mask = MaskTransformer(mask)
        # apply augmentations
        if self.augmentation:
            1;

        # apply preprocessing
        if self.preprocessing:
            1;

        if self.istest:
            return image, screen
        return image, mask, screen

    def __len__(self):
        return len(self.ids)


def MakeDatasets(images_dir, screen_dir, target_dir, MaxTrainingSetSize=4,
                 ValidationFraction=0.2, size=None, interpolation=2, ImageNetNorm=True):
    """Take image directory and produce training and validation Dataset (2 Dataset using my custom class. ). . \n
        Uses functions "get_screen_path" and "get_mask_path" to include the correct mask (y) and screen to the sets. \n

        Keyword arguments:
        images_dir, screen_dir, target_dir: all Path types.
        size : either int or None. If int, datasets will resize the images to a square with side size. \n
                The interpolation is done according to the definition in nn.transforms.resize() .
                If None, there will be no resizing and no interpolation. Note: non-square size would not work with the old UNet I wrote.
        interpolation: int. If size equals None, this will be ignored.
        MaxTrainingSetSize : int.
        ValidationFraction: float. fraction of samples to be reserved for validation set. Fraction taken from training set size, not total number of samples.

        Output
        two Datasets (custom class): training and validation.
        """
    x_images_paths = [image_path for image_path in images_dir.glob('*.tif')]

    TrainingSetSize = int(min(MaxTrainingSetSize, np.ceil(len(x_images_paths) * (1 - ValidationFraction))))
    ValidationSetSize = int(min(len(x_images_paths) - TrainingSetSize, np.ceil(TrainingSetSize * ValidationFraction)))
    x_images_paths = [x_images_paths[i] for i in np.random.permutation(len(x_images_paths))]

    x_images_paths_train = x_images_paths[:TrainingSetSize]
    x_images_paths_vldt = x_images_paths[TrainingSetSize:TrainingSetSize + ValidationSetSize]

    screen_paths_train = [get_screen_path(Path(image_path), screen_dir) for
                          image_path in x_images_paths_train]
    screen_paths_vldt = [get_screen_path(Path(image_path), screen_dir) for
                         image_path in x_images_paths_vldt]

    target_paths_train = [get_mask_path(Path(image_path), target_dir) for
                          image_path in x_images_paths_train]
    target_paths_vldt = [get_mask_path(Path(image_path), target_dir) for
                         image_path in x_images_paths_vldt]
    trainDataset = Dataset(x_images_paths_train, screen_paths_train, target_paths_train,
                           size=size, interpolation=interpolation, ImageNetNorm=ImageNetNorm)

    vldtnDataset = Dataset(x_images_paths_vldt, screen_paths_vldt, target_paths_vldt,
                           size=size, interpolation=interpolation, ImageNetNorm=ImageNetNorm)
    return (trainDataset, vldtnDataset)


def visualizeDataset(dataset, UnNormalize=True):
    """PLot images in one row. Only works for one class"""
    tmp_dataset = dataset
    fig, ax, = plt.subplots(1, 2, figsize=(16, 7))
    ind = np.random.randint(0, len(tmp_dataset.ids))

    tmp_image_tensor = (tmp_dataset[ind][0]).permute(1, 2, 0)
    tmp_mask_tensor = np.squeeze((tmp_dataset[ind][1]).permute(1, 2, 0))
    if UnNormalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        ax[0].imshow((tmp_image_tensor * torch.tensor(std)) + torch.tensor(mean))
    else:
        ax[0].imshow(tmp_image_tensor)
    ax[0].set(xticks=(), yticks=(), title='Image number ' + str(tmp_dataset.ids[ind]));
    ax[1].imshow(tmp_mask_tensor, cmap='gray')
    ax[1].set(xticks=(), yticks=());

def eval_epoch_vldtn_loss(model, data_loader, loss_criterion, metric=None):
    with torch.no_grad():
        model.eval()
        val_Epoch_losses = list()
        for ii, (data, target, screen) in enumerate(data_loader):
            data, target, screen = data.cuda(), target.cuda(), screen.cuda()
            output = model(data)
            #             val_loss = diceLoss(output, target)
            val_loss = loss_criterion(output, target, screen)
            #             val_loss = criterion(output, target, screen, alpha = torch.tensor(1.), gamma = 2.)
            val_Epoch_losses.append(val_loss.item())
            del val_loss
    return np.mean(val_Epoch_losses)
