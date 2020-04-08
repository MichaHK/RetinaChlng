import platform
import os
import time
from m_dataset_aux_functions import *
from m_network_architectures import *
# from torch.utils.data import DataLoader # mistake
import m_transforms
import m_loss_functionals
from Utils import *
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import matplotlib

from albumentations.pytorch.transforms import ToTensorV2
from albumentations.pytorch import ToTensor

from albumentations import (HorizontalFlip,
                            VerticalFlip,
                            Compose,
                            RandomCrop,
                            RandomSizedCrop,
                            OneOf,
                            RandomRotate90,
                            CLAHE,
                            RandomBrightnessContrast,
                            RandomGamma,
                            Normalize
                            )

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# after crf works, consider how to work with flips and others (the crf GT and image also needs to be flipped).
# TODO: the evaluation at the end is bad, because it uses the train loader, and caannot be used to produce confusion matrices. \
#  Needs to be changed for confusion matrices.
# TODO: eval_final should use the filenames and evalutae, not the data loaders. Then, produce DF with the target filenames, image filenames, metrices, and predicted filenames.


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

### Loading images from directory
if True:
    batch_size, num_workers = SetWorkersBatchSize()
    x_train_dir, y_train_dir, screen_train_dir = getLoaclTrainDataPaths()
    im, im_mask, im_screen = displayImageAndMaskFromFolder(x_train_dir, y_train_dir, screen_train_dir, display = False)

mean, std = FindAvgSTD_for_images(x_train_dir, screen_train_dir)
### setting data preprocessing and transformation
# Threshold = 240
# ImageSize = (300, 300)
# interpolation = 2
UseGreenOnly = False
UseSingleChannel = False
if UseGreenOnly:
    mean, std = [mean[1]], [std[1]]
    print('WARNING: using cv2, so loaded ans BGR not RGB. All green only settings must be changed. ')

serial_string = getSerialStringForTraining()


preprocessing_trnsfrms = [
    # RandomCrop(ImageSize[0], ImageSize[1], always_apply=False, p=1.0),
    CLAHE(clip_limit=2.0, tile_grid_size=(25, 25), p=1, always_apply=True),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    RandomBrightnessContrast(p=0.8),
    RandomGamma(p=0.8),
    ]
Image_trsnfrsms_F = Compose(preprocessing_trnsfrms, additional_targets={
    'mask': 'mask',
    'screen': 'mask',
    })


augmented = Image_trsnfrsms_F(image=im, mask=im_mask, screen=im_screen)
im_prcsd = augmented['image']
im_prcsd_mask = augmented['mask']
im_prcsd_screen = augmented['screen']


visualize_numpy(im_prcsd, im_prcsd_mask, original_image=im, original_mask=im_mask)
visualize_numpy(im_prcsd, im_prcsd_screen, original_image=im, original_mask=im_screen)
# im_screen

### Setting training and validation datasets
MaxTrainingSetSize = 20

ImageNetNorm = False
if True:
    (trainDataset, vldtnDataset), (paths_train, paths_vldt) = MakeDatasets(x_train_dir, screen_train_dir, y_train_dir,
                                                MaxTrainingSetSize=MaxTrainingSetSize,
                                                UseGreenOnly=UseGreenOnly,
                                                preprocessing_trnsfrms=preprocessing_trnsfrms,
                                                )
    print('Training set size: {}, Validation set size: {}'.format(len(trainDataset), len(vldtnDataset)))
    train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    vldtn_loader = DataLoader(dataset=vldtnDataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # visualizeDataset(trainDataset)

# visualizeTransforms(x_train_dir, screen_train_dir, y_train_dir, preprocessing_trnsfrms=preprocessing_trnsfrms,
#                     imageInd=0, UseGreenOnly=UseGreenOnly, mean=mean, std=std)
### Load model and loss function
torch.cuda.empty_cache()
bn = True
model = UNet_V4(n_class=1, bn=bn, SingleChannel=UseSingleChannel).cuda()
# model = UNet_V2(n_class=1).cuda()

# PATH = r'16Run_fullsize_Gamma_Clahe_normDataset_BN_more.pth'
# model.load_state_dict(torch.load(PATH))

metric = m_loss_functionals.DiceLoss()
criterion = m_loss_functionals.WCE(weight=torch.tensor(0.8))
# criterion = m_loss_functionals.FocalLoss(gamma=5, alpha=0.25)


###
initial_lr = 0.01 #  1e-2
num_iterations = 400
epochNumFor_lr_Decrease = 50
# optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr = initial_lr, momentum=0.9) # works well
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=initial_lr)
if True:
    losses = list()
    vldtn_losses = list()
    metric_vals = list()
    model.train()
    lr = initial_lr
    for epoch in range(num_iterations):
        # in case I want to break the training, but still save the weights, this try has a keyboard exception
        try:
            t0 = time.time()
            Epoch_losses = list()
            Epoch_metric = list()
            model.train()
            for ii, (data, target, screen) in enumerate(train_loader):
                data, target, screen = data.cuda(), target.cuda(), screen.cuda()  # please do so before constructing optimizers for it
                optimizer.zero_grad()
                output = model(data)
                target, screen = target[:,None,:,:], screen[:,None,:,:]
                loss = criterion(output, target, screen)
                with torch.no_grad():
                    metric_val = metric(output, target, screen)
                loss.backward()
                optimizer.step()
                Epoch_losses.append(loss.item())
                Epoch_metric.append(metric_val.item())
                del loss
                del metric_val

            losses.append(np.mean(Epoch_losses))
            metric_vals.append(np.mean(Epoch_metric))
            # run validation
            val_Epoch_mean, metric_epoch_mean = eval_epoch_vldtn_loss(model, vldtn_loader, criterion, metric=metric)
            vldtn_losses.append(val_Epoch_mean)

            adjust_learning_rate(lr, optimizer, epoch, ratio=0.5, epochNumForDecrease=epochNumFor_lr_Decrease)
            print('Epoch: {} - Loss: {:.4f} - Metric: {:.3f} , Validation: {:.4f}, Runtime: {:.2f} [s]'.format(epoch + 1,
                                                                                                               np.mean(
                                                                                                                   Epoch_losses),
                                                                                                               np.mean(
                                                                                                                   Epoch_metric),
                                                                                                               val_Epoch_mean,
                                                                                                               time.time() - t0))
        except KeyboardInterrupt:
            print('\n\nLoop interrupted by keyboard, model and metadata will be saved. \n\n')
            break

    # after running, save meta and model
    print('saving model...')
    Base_filename = model._get_name() + '_S' + serial_string
    torch.save(model.state_dict(), Base_filename+ '.pth')
    print('Model saved, ' + Base_filename+ '.pth')

    print('saving metadata...')
    meta_filename = 'Metadata_S' + serial_string + '.txt'
    with open(meta_filename, 'w') as filehandle:
        filehandle.write('preprocessing_trnsfrms:\n')
        for listitem in preprocessing_trnsfrms:
            filehandle.write('%s\n' % listitem)
    with open(meta_filename, 'a') as filehandle:
        filehandle.write('\noptimizer:\n')
        filehandle.write('%s\n' % optimizer)
        filehandle.write('\nLoss function:\n')
        filehandle.write('%s\n' % criterion)
        filehandle.write('\nModel:\n')
        filehandle.write('%s\n' % model)
    print('Metadata Saved')

    # Saving the validation and training set names

    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

else:
    ## for loading files.
    PATH = r'UNet_V2_S0001.pth'
    model = UNet_V2(n_class=1, SingleChannel=UseSingleChannel).cuda()
    model.load_state_dict(torch.load(PATH))
    model.eval()


VisulaizePrediction(model, train_loader, ind_in_batch=0, threshold=0.5)


print('Running evaluation of entire set...')
list_metrices = list([m_loss_functionals.specificity, m_loss_functionals.Sensitivity,
                      m_loss_functionals.Accuracy, m_loss_functionals.DiceLoss])
final_vldn_df = eval_final(model, vldtn_loader, list_metrices=list_metrices)
# ROC for validation set.
thresholds, FPR_list, TPR_list = calculate_ROC(model, vldtn_loader, serial_string)  # very slow

ROC_df = pd.DataFrame(data=list(zip(thresholds, FPR_list, TPR_list)),
             columns= ['threshold', 'FPR', 'TPR'])
ROC_curve_filename = 'ROC_curve_S' + serial_string + '.csv'
ROC_df.to_csv(ROC_curve_filename, index=False)

pseudo_train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
final_train_df = eval_final(model, pseudo_train_loader, list_metrices=list_metrices)
final_train_df.to_csv('Train_metrics_final_' + serial_string + '.csv')
final_vldn_df.to_csv('Vldn_metrics_final_' + serial_string + '.csv')
print(final_train_df)
print(final_vldn_df)
print('Evaluation done.')

# import pdb; pdb.set_trace()

# save to file the losses during training progress.
print('Saving training log')
loss_DF = pd.DataFrame(data=list(zip(losses, vldtn_losses, metric_vals)),
             columns= [str(criterion) + '_trn', str(criterion) + '_vldtn', str(metric) + '_vldtn'])
losses_filename = 'losses_during_training_S' + serial_string + '.csv'
loss_DF.to_csv(losses_filename, index_label='Epoch')

fig, ax = plt.subplots(1, figsize=(7,7))
start_ind = 3
ax.plot(list(range(len(losses)))[start_ind:], losses[start_ind:], label='Training losses')
ax.plot(list(range(len(losses)))[start_ind:], vldtn_losses[start_ind:], label='Validation losses')
ax.set_ylabel('WCE Loss')
ax.set_xlabel('Epoch')
# ax.set_ylim(15000,70000)
ax2 = ax.twinx()
color = 'r'
ax2.set_ylabel('Dice', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.plot(list(range(len(losses)))[start_ind:], metric_vals[start_ind:], label='Validation Dice', color=color)
ax2.set_ylim(0.27,0.75)
fig.legend()
ax.set_title('Serial S{}'.format(serial_string))
fig.show()



print('\n\nLosses in training saved, ' + losses_filename)

print('All done')
# import os
#
# os.system("sudo poweroff")
