import platform
import matplotlib.pyplot as plt
import glob
import re

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

def getSerialStringForTraining():
    """
    Looks up for the latest model saved (.pth file with highest serial). Retunrs the next serial string.

    :params

    :return

      (str) serial string for the next serial model to train.

    """
    filenames = glob.glob('*.pth')
    serial = 0
    serial_string = str(serial).zfill(4)
    for filename in filenames:
        m = re.match(r'(?P<name>\w+)_S(?P<serial>\d+).pth', filename)
        if m:
            if int(m['serial']) >= serial:
                serial = int(m['serial'])
                serial_string = str(serial+1).zfill(4)
    print('Serial number ' + serial_string)
    return serial_string

def visualize_numpy(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask, cmap='gray')
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask, cmap='gray')
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    f.show()
