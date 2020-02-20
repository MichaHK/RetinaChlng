from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from PIL import ImageOps
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torchvision
import m_transforms
import m_loss_functionals
import random

from datetime import datetime




def getLoaclTrainDataPaths():
    BaseFolder = Path.cwd()
    DataFolder = BaseFolder / 'Data'

    x_train_dir = DataFolder / 'training' / 'images'
    y_train_dir = DataFolder / 'training' / '1st_manual';
    screen_train_dir = DataFolder / 'training' / 'mask'
    return x_train_dir, y_train_dir, screen_train_dir

def displayImageAndMaskFromFolder(images_dir, y_train_dir, screen_train_dir):
    x_images_paths = [str(image_path.absolute()) for image_path in images_dir.glob('*.tif')]
    y_masks_paths = [str(get_mask_path(image_path, y_train_dir).absolute()) for
                     image_path in images_dir.glob('*.tif')]
    screen_paths = [str(get_screen_path(image_path, screen_train_dir).absolute()) for
                     image_path in images_dir.glob('*.tif')]
    im = Image.open(x_images_paths[0])
    im_mask = Image.open(y_masks_paths[0])
    im_screen = Image.open(screen_paths[0])
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(im)
    ax[1].imshow(im_mask, cmap='gray')
    ax[2].imshow(im_screen, cmap='gray')
    fig.show()
    # fig.savefig('ImageAndMask_.jpg', dpi = 500)

def VisulaizePrediction(model, dataloader, ind_in_batch=0, threshold=0.5, plotAll = False):
    ind = ind_in_batch
    mn = threshold
    with torch.no_grad():
        model.eval()

        for ii, (data, target, screen) in enumerate(dataloader):
            data, target, screen = data.cuda(), target.cuda(), screen.cuda()
            output = model(data)

            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            predicted = torch.sigmoid(output[ind, 0, :, :]).clone().cpu().detach().numpy()
            screen_numpy = screen[ind, 0, :, :].clone().cpu().detach().numpy()

            predicted[screen_numpy < 0.8] = 0
            predicted[predicted > mn] = 1
            predicted[predicted < mn] = 0
            #         predicted[predicted > mn] = 1
            t_array = target[ind, 0, :, :].clone().cpu().detach().numpy()
            ax[0].imshow(predicted, cmap='gray')
            ax[1].imshow(t_array, cmap='gray')
            fig.show()
            if not plotAll:
                break

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
            preprocessing_trnsfrms=list(),
            mask_trnsfrms=list(),
            UseGreenOnly=False,
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
        self.preprocessing_trnsfrms = preprocessing_trnsfrms
        self.mask_trnsfrms = mask_trnsfrms
        self.UseGreenOnly = UseGreenOnly

    def __getitem__(self, i):
        # load image, screen and mask (target). The tiff images open only with Pillow version 5.2.0
        image = Image.open(self.images[i])
        screen = Image.open(self.screens[i])

        if not self.istest:
            mask = Image.open(self.masks[i])

        if self.UseGreenOnly:
            _, image, _ = image.split()

        # Flip = transforms.RandomHorizontalFlip(p=0.5)
        # CenterCrop = transforms.CenterCrop(self.size)


        ImageTransforms = self.preprocessing_trnsfrms.copy()
        MaskTransforms = self.mask_trnsfrms.copy()

        # This makes sure that both image and target are random cropped the same way.
        # I should replace this to either use same random seed (import random), or use a mutable list to include the random x,y,w,h as a class variable.
        if isinstance(ImageTransforms[0], torchvision.transforms.transforms.RandomCrop):
            CropSize = ImageTransforms[0].size
            if isinstance(CropSize, int):
                CropSize = (CropSize, CropSize)
            del ImageTransforms[0]
            del MaskTransforms[0]
            y = np.random.randint(0, image.size[0] - CropSize[0])
            x = np.random.randint(0, image.size[1] - CropSize[1])
            image = image.crop((y, x, y + CropSize[0], x + CropSize[1]))  # left, upper, right, and lower
            screen = screen.crop((y, x, y + CropSize[0], x + CropSize[1]))  # left, upper, right, and lower
            if not self.istest:
                mask = mask.crop((y, x, y + CropSize[0], x + CropSize[1]))  # left, upper, right, and lower

        ImageTransformer = transforms.Compose(ImageTransforms)

        MaskTransformer = transforms.Compose(MaskTransforms)

        image = ImageTransformer(image)
        screen = MaskTransformer(screen)

        if not self.istest:
            mask = MaskTransformer(mask)
        if self.istest:
            return image, screen
        return image, mask, screen



    def __len__(self):
        return len(self.ids)


def MakeDatasets(images_dir, screen_dir, target_dir, MaxTrainingSetSize=4,
                 ValidationFraction=0.2, mask_trnsfrms=list(), UseGreenOnly=False, preprocessing_trnsfrms=list()):
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

    trainDataset = Dataset(x_images_paths_train, screen_paths_train, mask_paths=target_paths_train,
                           UseGreenOnly=UseGreenOnly, preprocessing_trnsfrms=preprocessing_trnsfrms,
                           mask_trnsfrms = mask_trnsfrms)

    vldtnDataset = Dataset(x_images_paths_vldt, screen_paths_vldt, mask_paths=target_paths_vldt,
                           UseGreenOnly=UseGreenOnly, preprocessing_trnsfrms=preprocessing_trnsfrms,
                           mask_trnsfrms=mask_trnsfrms)
    return (trainDataset, vldtnDataset)


def visualizeDataset(dataset, mean = None, std = None):
    """PLot images in one row. Only works for one class"""
    tmp_dataset = dataset
    fig, ax, = plt.subplots(1, 2, figsize=(16, 7))
    ind = np.random.randint(0, len(tmp_dataset.ids))
    image, mask, screen = tmp_dataset[ind]
    def PermuteDimsForPlot(tensor):
        if tensor.shape[0] == 1:
            return tensor.squeeze()
        elif tensor.shape[0] == 3:
            return tensor.permute(1, 2, 0)

    UnNormalize = isinstance(mean, np.ndarray)

    if UnNormalize:
        for i in range(image.shape[0]): # color index starts since this is the tensor, not PIL
            image[i, :, :] = image[i, :, :]* std[i] + mean[i]   # color index starts since this is the tensor, not PIL
    else:
        image = image

    image, mask = PermuteDimsForPlot(image), PermuteDimsForPlot(mask)
    # ToPIL = torchvision.transforms.ToPILImage()

    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(mask, cmap='gray')
    ax[0].set(xticks=(), yticks=(), title='Image number ' + str(tmp_dataset.ids[ind]))
    ax[1].set(xticks=(), yticks=())
    fig.show()

def eval_epoch_vldtn_loss(model, data_loader, loss_criterion, metric=None, UseOneOnly = True):
    with torch.no_grad():
        model.eval()
        val_Epoch_losses = list()
        metric_epoch_vals = list()

        for ii, (data, target, screen) in enumerate(data_loader):
            data, target, screen = data.cuda(), target.cuda(), screen.cuda()
            output = model(data)
            val_loss = loss_criterion(output, target, screen)
            metric_epoch_val = metric(output, target, screen)
            val_Epoch_losses.append(val_loss.item())
            metric_epoch_vals.append(metric_epoch_val.item())
            del val_loss
            if UseOneOnly:
                break
    return np.mean(val_Epoch_losses), np.mean(metric_epoch_vals)

def calculate_ROC(model, data_loader):
    """
    Warning: Does not work with interpolated images yet. Need to resize and run the resized back to original size here.

    :param model:
    :param data_loader:
    :return:
    """
    thresholds = np.linspace(0,1, 3000)
    with torch.no_grad():
        model.eval()
        TPR_func = m_loss_functionals.Sensitivity()
        FPR_func = lambda output, target, screen=None: 1 - m_loss_functionals.specificity()(seg_output, target, screen)
        TPR_array, FPR_array  = np.zeros((len(data_loader), len(thresholds))), np.zeros((len(data_loader), len(thresholds)))
        for ind_image, (data, target, screen) in enumerate(data_loader):
            data, target, screen = data.cuda(), target.cuda(), screen.cuda()
            output = model(data)
            for ind_thrshld, threshold in enumerate(thresholds):
                seg_output = output.clone().detach()
                seg_output[torch.sigmoid(seg_output) < threshold] = -1e10
                seg_output[torch.sigmoid(seg_output) > threshold] = +1e10
                TPR_array[ind_image, ind_thrshld] = TPR_func(seg_output, target, screen).item()
                FPR_array[ind_image, ind_thrshld] = FPR_func(seg_output, target, screen).item()
    FPR_list = np.mean(FPR_array, axis = 0)
    TPR_list = np.mean(TPR_array, axis=0)
    fig, ax = plt.subplots(1)
    ax.plot(FPR_list, TPR_list)
    AUC = np.trapz(TPR_list, FPR_list)
    ax.set(xlabel = 'FPR (1-Specificity)', ylabel = 'TPR (Sensitivity', title = 'ROC. AUC = {:.2f}'.format(AUC))
    fig.show()
    # print('dsfdsfs')
    return None

def eval_final(model, data_loader, list_metrices = list(), threshold = None):
    with torch.no_grad():
        model.eval()
        columns = [str(metric()) for metric in list_metrices]
        vldtn_df = pd.DataFrame(columns=columns)
        for ii, (data, target, screen) in enumerate(data_loader):
            data, target, screen = data.cuda(), target.cuda(), screen.cuda()
            output = model(data)
            if threshold:
                output[torch.sigmoid(output) < threshold] = 0
                output[torch.sigmoid(output) > threshold] = 1
            loss_items = [metric()(output, target, screen).item() for metric in list_metrices]
            tmp_df = pd.DataFrame(data=[loss_items], columns=columns)
            vldtn_df = vldtn_df.append(tmp_df, ignore_index=True)
    return vldtn_df

def visualizeTransforms(x_train_dir, screen_train_dir, y_train_dir, preprocessing_trnsfrms=list(), imageInd = 0,
                        UseGreenOnly = False, mean = None, std = None):
    def PermuteDimsForPlot(tensor):
        if tensor.shape[0] == 1:
            return tensor.squeeze()
        elif tensor.shape[0] == 3:
            return tensor.permute(1, 2, 0)

    SkipTransformationAfterToTensor = True
    UnNormalize = isinstance(mean, np.ndarray)
    assert len(preprocessing_trnsfrms) > 0

    if SkipTransformationAfterToTensor:
        indicator_of_ToTensor = [i for i, transform in enumerate(preprocessing_trnsfrms) if
                                 isinstance(transform, torchvision.transforms.ToTensor)][0]
        preprocessing_trnsfrms = preprocessing_trnsfrms[:indicator_of_ToTensor]
        UnNormalize = False
    NumOfTransforms = len(preprocessing_trnsfrms)

    if UseGreenOnly:
        cmap = 'gray'
    else:
        cmap= None

    transform_names = list()

    fig, ax, = plt.subplots(1, NumOfTransforms + 1, figsize=(16, 7))
    # im = Image.open(x_images_paths[0])
    # ax[0].imshow(im)

    seed = np.random.randint(0,42)

    for ind, trnsfrm in enumerate(preprocessing_trnsfrms):
        transform_names.append(str(trnsfrm))
        if ind == 0:
            np.random.seed(seed)
            _, CompleteDataset = MakeDatasets(x_train_dir, screen_train_dir, y_train_dir, MaxTrainingSetSize=20,
                                              ValidationFraction=0.5, mask_trnsfrms=list(), UseGreenOnly=UseGreenOnly,
                                              preprocessing_trnsfrms=list([transforms.CenterCrop(564),
                                                                           torchvision.transforms.ToTensor()]))
            image, _, _ = CompleteDataset[imageInd] # 1
            image = PermuteDimsForPlot(image)
            min_value = image.min()
            if min_value < 0:
                image_tmp = (image - min_value) / (image.max() - min_value)
            else:
                image_tmp = image.detach().cpu()
            ax[ind].imshow(image_tmp, cmap = cmap)
            ax[ind].set(title='Original')
        np.random.seed(seed)
        _, CompleteDataset = MakeDatasets(x_train_dir, screen_train_dir, y_train_dir, MaxTrainingSetSize=20,
                     ValidationFraction=0.5, mask_trnsfrms=list(), UseGreenOnly=UseGreenOnly, preprocessing_trnsfrms=preprocessing_trnsfrms[:ind+1])
        image, _, _ = CompleteDataset[imageInd] # 1
        if not isinstance(image, torch.Tensor):
            ToTensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image = ToTensor(image)

        isFinalImage = ind == NumOfTransforms-1
        if (UnNormalize and isFinalImage):
            for i in range(image.shape[0]):  # color index starts since this is the tensor, not PIL
                image[i, :, :] = image[i, :, :] * std[i] / 255 + mean[i] / 255   # color index starts since this is the tensor, not PIL
        else:
            image = image

        image = PermuteDimsForPlot(image)
        ax[ind+1].imshow(image, cmap = cmap)
        ax[ind+1].set(title = str(trnsfrm)[:])

    now = datetime.now()
    filename = now.strftime("%d-%m-%Y-%H-%M")

    # fig.savefig('TransformationData_'+filename+'.jpg', dpi =600)
    fig.show()
    # savefig(fname, dpi=None, facecolor='w', edgecolor='w',
    #         orientation='portrait', papertype=None, format=None,
    #         transparent=False, bbox_inches=None, pad_inches=0.1,
    #         frameon=None, metadata=None)


def FindAvgSTD_for_images(images_dir, screen_train_dir):
    '''
    :param images_dir:
    :param screen_train_dir:
    :return:
    '''
    x_images_paths = [str(image_path.absolute()) for image_path in images_dir.glob('*.tif')]
    screen_paths = [str(get_screen_path(image_path, screen_train_dir).absolute()) for
                     image_path in images_dir.glob('*.tif')]
    image_sum = np.array([0,0,0])
    image_std = np.array([0,0,0])
    ind = 0
    for image_path, screen_path in zip(x_images_paths, screen_paths):
        image = Image.open(image_path)
        screen = Image.open(screen_path)
        for i in range(3):
            image_sum[i] += np.mean(np.array(image, dtype='uint')[:,:,i][np.array(screen) > 0.9])
            image_std[i] += np.std(np.array(image, dtype='uint')[:,:,i][np.array(screen) > 0.9])
        ind += 1
    return image_sum/ind, image_std/ind
