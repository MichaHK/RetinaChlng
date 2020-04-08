# ======================================================================
# pytorch transformations compatible with albumenations package
# ======================================================================
#
# make sure you always work with cv2 conventions:
# Dimensions: w x h x Channels
# Scale and type:   [0,255] image in uint 8
#                   [-1,1] or [0,1] image in float32
#
# ======================================================================




from skimage.util import random_noise
import random
import cv2
import numpy as np
from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform, NoOp, to_tuple
from albumentations.core.utils import format_args
import matplotlib.pyplot as plt


def RescaleAndType(image, output_type):
    """
    Function rescales GRAYSCALE images to desired range and format: uint8, scaled to [0,256]; or: float scaled to [0,1]
    For a batch, rescales every image in batch separately.

    Args:
        image: in BS x w x h x 1, or w x h x 1.
        output_type: 'float32' or 'uint8'

    Targets:
        image, same shape. uint8, scaled to [0,256]; or: float scaled to [0,1]

    """
    assert image.ndim == 4 or image.ndim == 3, 'Dimensions are {}'.format(image.shape)
    assert output_type in ['float32', 'uint8']
    assert image.shape[-1] == 1, 'only grayscale'

    if image.ndim == 4:
        min_of_each_sample_in_bs = np.min(image, axis=(1, 2, 3))
        max_of_each_sample_in_bs = np.max(image, axis=(1, 2, 3))
        image = np.transpose(
            (np.transpose(image) - (min_of_each_sample_in_bs)) / (max_of_each_sample_in_bs - min_of_each_sample_in_bs))
    if image.ndim == 3:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

    if output_type == 'uint8':
        image = (image * 255).astype('uint8')
    elif output_type == 'float32':
        image = image.astype('float32')

    return image


class RescaleForAug(ImageOnlyTransform):
    """
    Rescale to [0,255] and uint8 to be compatible with the albu / torchvision transformations
    For a batch, rescales every image in batch separately.

    Args:
        image in BS x w x h x 1, or w x h x 1 if Ch(color) = 1.
        output_type = 'uint8'

    Targets:
        image, same shape. uint8, scaled to [0,256]

    """

    def __init__(self, always_apply=True, p=1):
        super(RescaleForAug, self).__init__(always_apply, p)

    def apply(self, img, **params):
        img = RescaleAndType(img, 'uint8')
        return img

    def __repr__(self):
        return self.__class__.__name__+'()'

    def get_transform_init_args_names(self):
        return ()

################# all the old ones used before albumentations ##############

class invert(object):
    """color invert image
    """

    # def __init__(self):
    def __call__(self, PIL_img):
        """
        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image: Cropped image.
        """

        return ImageOps.invert(PIL_img)

    def __repr__(self):
        return self.__class__.__name__


class threshold(object):
    """color invert image
    """

    def __init__(self, TopThresholdValue, replaceWith=0):
        self.TopThresholdValue, self.replaceWith = TopThresholdValue, replaceWith

    def __call__(self, PIL_img):
        """
        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image: Cropped image.
        """
        self.TopThresholdValue = 240
        PIL_img = PIL_img.point(lambda x: x if x < self.TopThresholdValue else self.replaceWith)  # threshold sides
        return PIL_img

    def __repr__(self):
        return self.__class__.__name__ + '(Val={0}'.format(self.TopThresholdValue)


class ImageNetNorm(object):
    """color invert image
    """

    # def __init__(self):
    def __call__(self, PIL_img):
        """
        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image: Cropped image.
        """
        Normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return Normalize(PIL_img)

    def __repr__(self):
        return self.__class__.__name__


class Clahe_trnsfrm(object):
    """color invert image
    """

    def __init__(self, clipLimit=2.0, tileGridSize=(20, 20)):
        self.clipLimit, self.tileGridSize = clipLimit, tileGridSize

    def __call__(self, PIL_img):
        """
        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image: Cropped image.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_array = np.array(PIL_img, dtype=np.uint8)
        hasSingleChannel = len(img_array.shape) < 3
        if hasSingleChannel:
            img_array = np.expand_dims(img_array, axis =0)

        for i in range(img_array.shape[2]):
            img_array[:, :, i] = clahe.apply(img_array[:, :, i])

        if hasSingleChannel:
            img_array = img_array[0,:,:]

        return Image.fromarray(img_array)

    def __repr__(self):
        return self.__class__.__name__ + '(clip={0}, tile={1})'.format(self.clipLimit, self.tileGridSize)

class HistEqualize(object):
    """color invert image
    """

    # def __init__(self):
    def __call__(self, PIL_img):
        """
        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image: Cropped image.
        """
        return ImageOps.equalize(PIL_img, mask=None)

    def __repr__(self):
        return self.__class__.__name__


class Gamma_cor(object):
    """color invert image
    """

    def __init__(self, gamma=0.5):
        self.gamma = gamma

    def __call__(self, PIL_img):
        """
        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image: Cropped image.
        """
        img_array = np.array(PIL_img, dtype=np.uint8)
        img_array = 255.0 * (img_array / 255.0)**(1 / self.gamma)
        return Image.fromarray(np.uint8(img_array))

    def __repr__(self):
        return self.__class__.__name__ + '(g={0})'.format(self.gamma)


class DatasetNormalizationToTensor(object):
    """Dataset normalization using z-score. Works poorly since the z-score can get negative values.
    there are three common techniques for value normalization:
    (x - x.min()) / (x.max() - x.min()) # values from 0 to 1
    2*(x - x.min()) / (x.max() - x.min()) - 1 # values from -1 to 1
    (x - x.mean()) / x.std() # values from ? to ?, but mean at 0
    """

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, PIL_img):
        """
        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image: Cropped image.
        """
        img_array = np.array(PIL_img, dtype=np.int16)
        hasSingleChannel = len(img_array.shape) < 3
        if hasSingleChannel:
            img_array = np.expand_dims(img_array, axis =0)
        thresholds = [40, 40, 40]
        tmp_screen = (img_array[:, :, 0] > thresholds[0]) & (img_array[:, :, 1] > thresholds[1]) & (img_array[:, :, 2] > thresholds[2])
        for i in range(img_array.shape[2]):
            img_array[:, :, i][tmp_screen] = (img_array[:, :, i][tmp_screen] - self.mean[i]) / self.std[i]
        if hasSingleChannel:
            img_array = img_array[0,:,:]

        # normalize to -1, 1

        # return Image.fromarray(img_array)

    def __repr__(self):
        return self.__class__.__name__
