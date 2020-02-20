import platform
import os
import time
from m_dataset_aux_functions import *
from m_network_architectures import *
import m_transforms
import m_loss_functionals
from torchsummary import summary

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('GTXAgg')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def SetWorkersBatchSize():
    machine_OS = platform.system()
    if machine_OS == 'Windows':
        batch_size = 1
        num_workers = 0
    elif machine_OS == 'Linux':
        batch_size = 1
        num_workers = 16
    print(machine_OS, 'OS. Batchsize:', batch_size, ', Num of workers:', num_workers)
    return batch_size, num_workers


### Loading images from directory
if True:
    batch_size, num_workers = SetWorkersBatchSize()
    x_train_dir, y_train_dir, screen_train_dir = getLoaclTrainDataPaths()
    displayImageAndMaskFromFolder(x_train_dir, y_train_dir, screen_train_dir)

mean, std = FindAvgSTD_for_images(x_train_dir, screen_train_dir)
mean, std = mean/255, std/255
### setting data preprocessing and transformation
Threshold = 240
ImageSize = (300, 300)
interpolation = 2
UseGreenOnly = False
UseSingleChannel = False
if UseGreenOnly:
    mean, std = [mean[1]], [std[1]]
# transforms.Resize(ImageSize, interpolation), m_transforms.invert(),
# m_transforms.threshold(240), m_transforms.Clahe_trnsfrm()
# torchvision.transforms.RandomCrop(ImageSize), m_transforms.HistEqualize()
# preprocessing_trnsfrms =list([torchvision.transforms.RandomCrop(ImageSize),
#                               m_transforms.invert(),
#                               m_transforms.threshold(Threshold),
#                               m_transforms.Clahe_trnsfrm()
#                              ])
# mask_trnsfrms = list([torchvision.transforms.RandomCrop(ImageSize)])0
preprocessing_trnsfrms = list([
    # transforms.Resize(ImageSize, interpolation),
    # m_transforms.invert(),
    # m_transforms.threshold(Threshold),
    m_transforms.Gamma_cor(gamma=0.6),
    m_transforms.Clahe_trnsfrm(clipLimit=2.0,tileGridSize=(8,8)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])
mask_trnsfrms = list([#transforms.Resize(ImageSize, interpolation),
                      torchvision.transforms.ToTensor()])

### Setting training and validation datasets
MaxTrainingSetSize = 20

ImageNetNorm = False
if True:
    (trainDataset, vldtnDataset) = MakeDatasets(x_train_dir, screen_train_dir, y_train_dir,
                                                MaxTrainingSetSize=MaxTrainingSetSize,
                                                UseGreenOnly=UseGreenOnly,
                                                preprocessing_trnsfrms=preprocessing_trnsfrms,
                                                mask_trnsfrms=mask_trnsfrms)
    print('Training set size: {}, Validation set size: {}'.format(len(trainDataset), len(vldtnDataset)))
    train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    vldtn_loader = DataLoader(dataset=vldtnDataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    visualizeDataset(trainDataset, mean=mean, std=std)
    # visualizeDataset(trainDataset)

# visualizeTransforms(x_train_dir, screen_train_dir, y_train_dir, preprocessing_trnsfrms=preprocessing_trnsfrms,
#                     imageInd=0, UseGreenOnly=UseGreenOnly, mean=mean, std=std)

### Load model and loss function
bn = True
model = UNet_V4(n_class=1, bn=bn, SingleChannel=UseSingleChannel).cuda()
PATH = r'16Run_fullsize_Gamma_Clahe_normDataset_BN_more.pth'
model.load_state_dict(torch.load(PATH))


metric = m_loss_functionals.DiceLoss()
criterion = m_loss_functionals.WCE(weight=torch.tensor(0.600))

###
initial_lr = 0.002 #  1e-2
num_iterations = 100
epochNumFor_lr_Decrease = 100
# optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr = initial_lr, momentum=0.9) # works well
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=initial_lr)
if True:
    losses = list()
    vldtn_losses = list()
    model.train()
    lr = initial_lr
    for epoch in range(num_iterations):
        t0 = time.time()
        Epoch_losses = list()
        model.train()
        for ii, (data, target, screen) in enumerate(train_loader):
            data, target, screen = data.cuda(), target.cuda(), screen.cuda()  # please do so before constructing optimizers for it
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, screen)
            with torch.no_grad():
                metric_val = metric(output, target, screen)
            loss.backward()
            optimizer.step()
            Epoch_losses.append(loss.item())
            del loss

        losses.append(np.mean(Epoch_losses))
        # run validation
        val_Epoch_mean, metric_epoch_mean = eval_epoch_vldtn_loss(model, vldtn_loader, criterion, metric=metric)
        vldtn_losses.append(val_Epoch_mean)

        adjust_learning_rate(lr, optimizer, epoch, ratio=0.5, epochNumForDecrease=epochNumFor_lr_Decrease)
        print('Epoch: {} - Loss: {:.4f} - Metric: {:.3f} , Validation: {:.4f}, Runtime: {:.2f} [s]'.format(epoch + 1,
                                                                                                           np.mean(
                                                                                                               Epoch_losses),
                                                                                                           metric_val.item(),
                                                                                                           val_Epoch_mean,
                                                                                                           time.time() - t0))
        del metric_val

VisulaizePrediction(model, train_loader, ind_in_batch=0, threshold=0.5)

list_metrices = list([m_loss_functionals.specificity, m_loss_functionals.Sensitivity,
                      m_loss_functionals.Accuracy, m_loss_functionals.DiceLoss])
final_vldn_df = eval_final(model, vldtn_loader, list_metrices=list_metrices)
calculate_ROC(model, vldtn_loader)
pseudo_train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
final_train_df = eval_final(model, pseudo_train_loader, list_metrices=list_metrices)
print(final_train_df)
print(final_vldn_df)
print('sd')
Base_filename = '16Run_fullsize_Gamma_Clahe_normDataset_BN_Then_WCE0p6_'

torch.save(model.state_dict(), Base_filename + '.pth')
with open(Base_filename + '_Meta.txt', 'w') as filehandle:
    for listitem in preprocessing_trnsfrms:
        filehandle.write('%s\n' % listitem)
with open(Base_filename + '_Meta.txt', 'a') as filehandle:
    filehandle.write('optimizer:\n')
    filehandle.write('%s\n' % optimizer)
    filehandle.write('Loss function:\n')
    filehandle.write('%s\n' % criterion)
    filehandle.write('Model:\n')
    filehandle.write('%s\n' % model)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# import os
#
# os.system("sudo poweroff")
