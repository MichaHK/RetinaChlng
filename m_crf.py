import os
from torch.utils.data import DataLoader
import albumentations
import pydensecrf.densecrf as dcrf
import cv2
from pathlib import Path
import re
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax, unary_from_labels
from m_dataset_aux_functions import getLoaclTrainDataPaths, get_mask_path, get_screen_path, MakeDatasets, \
    VisulaizePrediction
from albumentations import CLAHE, Compose
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


from m_network_architectures import UNet_V2, UNet_V4

serial_model = 6
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# paths and load image.
x_train_dir, y_train_dir, screen_train_dir = getLoaclTrainDataPaths()
x_images_paths = [str(image_path.absolute()) for image_path in x_train_dir.glob('*.tif')]
y_masks_paths = [str(get_mask_path(image_path, y_train_dir).absolute()) for
                 image_path in x_train_dir.glob('*.tif')]
screen_paths = [str(get_screen_path(image_path, screen_train_dir).absolute()) for
                 image_path in x_train_dir.glob('*.tif')]
ind = 1
image_path = x_images_paths[ind]
target_path = y_masks_paths[ind]

image = cv2.imread(image_path)
target = np.array(Image.open(target_path))

# load model
curdir = Path()
model_list = list(curdir.glob('*.pth'))
for filepath in curdir.glob('*.pth'):
    m = re.match(r'(?P<name>\w+)_S(?P<serial>\d+).pth', filepath.name)
    if m:
        if int(m['serial']) == serial_model:
            model_filepath = filepath
            model_type = m['name']
            print(filepath, model_type)
if model_type == 'UNet_V2':
    model = UNet_V2(n_class=1).cuda()
elif (model_type == 'UNetV4' or model_type == 'UNet_V4'):
    model = UNet_V4(n_class=1, bn=True, SingleChannel=False).cuda()
model.load_state_dict(torch.load(model_filepath))
model.eval()
# load image and augment it to match preprocessing of model
preprocessing_trnsfrms = [CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1, always_apply=True)]
print('Check metadata to see if preprocessing is needed.')
preprocessing_trnsfrms.append(albumentations.pytorch.ToTensorV2()) # I don't think this works.
preprocessing_trnsfrms = Compose(preprocessing_trnsfrms)
Image_trsnfrsms_F = Compose(preprocessing_trnsfrms)

im_aug = Image_trsnfrsms_F(image=image)['image']/255.0
im_aug = im_aug[None,...]
im_aug = im_aug.cuda()
with torch.no_grad():
    output = model(im_aug)
# output prediction and save to folder.
# the plan is to save all predictions to npy files in a folder for each model. Then do crf for each. Then see the corrected. if looks good, do the dice evaluation on those too.
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/
# https://github.com/lucasb-eyer/pydensecrf/blob/master/examples/inference.py

softmax = torch.sigmoid(output[0, :, :]).clone().cpu().detach().numpy()
softmax_bin = np.concatenate([1-softmax, softmax])

unary_from_sm = unary_from_softmax(softmax_bin, scale=None, clip=1e-5)  # sm.shape[0] == n_classes
unary_from_sm2 = -np.log(softmax+ 1E-7)
unary_from_sm2 = np.ascontiguousarray(unary_from_sm2.reshape((1, unary_from_sm2.shape[2]*unary_from_sm2.shape[1]))).astype(np.float32)
# U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
# The inputs should be C-continious -- we are using Cython wrapper
n_labels = 2
d = dcrf.DenseCRF(image.shape[0] * image.shape[1], n_labels)

unary = unary_from_sm
d.setUnaryEnergy(unary)
title = ''
# This potential penalizes small pieces of segmentation that are
# spatially isolated -- enforces more spatially consistent segmentations
sdims = (5, 5)
compat = 1
title = title + ('seg dim={}, cmpt={}'.format(sdims, compat))
feats = create_pairwise_gaussian(sdims=(5, 5), shape=image.shape[:2])

d.addPairwiseEnergy(feats, compat=compat,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

# This creates the color-dependent features --
# because the segmentation that we get from CNN are too coarse
# and we can use local color features to refine them
sdims = (5, 5)
compat = 15
title = title + ('. clr dim={}, cmpt={}'.format(sdims, compat))
feats = create_pairwise_bilateral(sdims=sdims, schan=(70, 1, 1),
                                   img=image, chdim=2)
#
d.addPairwiseEnergy(feats, compat=compat,
                     kernel=dcrf.DIAG_KERNEL,
                     normalization=dcrf.NORMALIZE_SYMMETRIC)

####################################
### Do inference and compute MAP ###
####################################

# Run five inference steps.
Q = d.inference(5)

# Find out the most probable class for each pixel.
res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

# Convert the MAP (labels) back to the corresponding colors and save the image.
# Note that there is no "unknown" here anymore, no matter what we had at first.
# cmap = plt.get_cmap('bwr')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18,7))
# ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
ax1.imshow(res, cmap='Greys')
ax1.set_title('Post CRF, ' + title)
probability_graph = ax2.imshow(target, cmap='Greys')
ax2.set_title('Ground-Truth Annotation')
ax3.imshow(softmax[0,...], cmap='Greys')
ax3.set_title('Pre CRF probs')
f.show()

print(res.min(), res.max())
print(softmax.min(), softmax.max())
#
# MaxTrainingSetSize = 20
# UseGreenOnly = False
# batch_size = 1
# num_workers = 4
#
# (trainDataset, vldtnDataset), (paths_train, paths_vldt) = MakeDatasets(x_train_dir, screen_train_dir, y_train_dir,
#                                                 MaxTrainingSetSize=MaxTrainingSetSize,
#                                                 UseGreenOnly=UseGreenOnly,
#                                                 preprocessing_trnsfrms=preprocessing_trnsfrms,
#                                                 )
# print('Training set size: {}, Validation set size: {}'.format(len(trainDataset), len(vldtnDataset)))
#
# train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#
# VisulaizePrediction(model, train_loader, ind_in_batch=0, threshold=0.5)

#
# MaxTrainingSetSize = 20
# UseGreenOnly = False
# batch_size = 1
# num_workers = 4
#
# (trainDataset, vldtnDataset), (paths_train, paths_vldt) = MakeDatasets(x_train_dir, screen_train_dir, y_train_dir,
#                                                 MaxTrainingSetSize=MaxTrainingSetSize,
#                                                 UseGreenOnly=UseGreenOnly,
#                                                 preprocessing_trnsfrms=preprocessing_trnsfrms,
#                                                 )
# print('Training set size: {}, Validation set size: {}'.format(len(trainDataset), len(vldtnDataset)))
# # from m_dataset_aux_functions import *
# import torch
#
# train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
# vldtn_loader = DataLoader(dataset=vldtnDataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
#
# for i in train_loader:
#     break
