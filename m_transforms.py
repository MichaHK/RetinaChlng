import cv2
from PIL import ImageOps, Image
from torchvision import transforms
import numpy as np

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
    def __init__(self, TopThresholdValue, replaceWith = 0):
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
        return self.__class__.__name__

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

    def __init__(self, clipLimit=2.0, tileGridSize=(20,20)):
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
            img_array = np.expand_dims(img_array)

        for i in range(img_array.shape[2]):
            img_array[:, :, i] = clahe.apply(img_array[:,:,i])

        return Image.fromarray(img_array)

    def __repr__(self):
        return self.__class__.__name__


