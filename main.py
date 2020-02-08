import platform
import os
import time
from m_dataset_aux_functions import *
from m_network_architectures import *
import m_transforms
import m_loss_functionals

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

### setting data preprocessing and transformation
Threshold = 240
ImageSize = (300, 300)
interpolation = 2
# transforms.Resize(ImageSize, interpolation), m_transforms.invert(),
# m_transforms.threshold(240), m_transforms.Clahe_trnsfrm()
# torchvision.transforms.RandomCrop(ImageSize), m_transforms.HistEqualize()
# preprocessing_trnsfrms =list([torchvision.transforms.RandomCrop(ImageSize),
#                               m_transforms.invert(),
#                               m_transforms.threshold(Threshold),
#                               m_transforms.Clahe_trnsfrm()
#                              ])
# mask_trnsfrms = list([torchvision.transforms.RandomCrop(ImageSize)])0
preprocessing_trnsfrms =list([transforms.Resize(ImageSize, interpolation),
                              m_transforms.invert(),
                              m_transforms.threshold(Threshold),
                              m_transforms.Clahe_trnsfrm(),
                              m_transforms.Gamma_cor(gamma=0.5)
                              ])
mask_trnsfrms = list([transforms.Resize(ImageSize, interpolation)])

### Setting training and validation datasets
MaxTrainingSetSize = 1
UseGreenOnly = False
UseSingleChannel = False
ImageNetNorm = False
if True:
    (trainDataset, vldtnDataset) = MakeDatasets(x_train_dir, screen_train_dir, y_train_dir,
                                                MaxTrainingSetSize = MaxTrainingSetSize,
                                                UseGreenOnly=UseGreenOnly,
                                               preprocessing_trnsfrms = preprocessing_trnsfrms,
                                               mask_trnsfrms = mask_trnsfrms)
    print('Training set size: {}, Validation set size: {}'.format(len(trainDataset), len(vldtnDataset)))
    train_loader = DataLoader(dataset = trainDataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    vldtn_loader = DataLoader(dataset = vldtnDataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    visualizeDataset(trainDataset)

visualizeTransforms(x_train_dir, screen_train_dir, y_train_dir, preprocessing_trnsfrms=preprocessing_trnsfrms, imageInd = 0, UseGreenOnly = UseGreenOnly)

### Load model and loss function
model = UNet_V4(n_class=1, bn = True, SingleChannel=UseSingleChannel).cuda()
criterion = m_loss_functionals.WCE(weight=torch.tensor(0.7000))
metric = m_loss_functionals.specificity()

###
initial_lr = 1e-3
num_iterations = 50
epochNumFor_lr_Decrease = 25
# optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr = initial_lr, momentum=0.9) # works well
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr = initial_lr)
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


VisulaizePrediction(model, train_loader, ind_in_batch = 0, threshold = 0.5)
# torch.save(model, str(BaseFolder / 'tmp.pth'))

