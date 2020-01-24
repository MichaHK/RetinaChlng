import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps
    def forward(self, output, target, screen = None):    
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here. 
        target.requires_grad_(requires_grad=False) # may be redundant
        Activation = torch.sigmoid(output)
        if isinstance(screen, torch.Tensor):
            screen.requires_grad_(requires_grad=False) # may be redundant
            numerator = (Activation[screen > 0.5] * target[screen > 0.5]).sum()
            denominator = Activation[screen > 0.5].sum() + target[screen > 0.5].sum()
        elif screen == None:
            numerator = (Activation * target).sum()
            denominator = Activation.sum() + target.sum()
        else: 
            print('screen argument type is wrong')
        return 1 - 2 * (numerator / denominator)

        
class WCE(nn.Module):
    """My custom WEIGHTED Cross Entropy loss function. It first runs the output through a Sigmoid, then computes the
    cross entropy. Weight is in range [0,inf]. To decrease the number of false negatives, use a large weight. 
    To decrease the number flase positives, set to small. 
    """
    def __init__(self, eps=1e-7):
        super(WCE, self).__init__()
        self.eps = eps
    def forward(self, output, target, screen = None):  
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here. 
        target.requires_grad_(requires_grad=False) # may be redundant
        if isinstance(screen, torch.Tensor):
            screen.requires_grad_(requires_grad=False) # may be redundant
            return -(weight * target[screen > 0.5] * F.logsigmoid(output[screen > 0.5]) + 
                    (1-target[screen > 0.5]) * F.logsigmoid(1 - output[screen > 0.5])).sum()
        elif screen == None:
            return -(weight * target * F.logsigmoid(output) + (1-target) * F.logsigmoid(1 - output)).sum()
        else: 
            print('screen argument type is wrong')
            return screen


class FocalLoss(nn.Module):
    """My custom Focal Loss function. It runs on an output that was not normalized yet using a sigmoid. 
    For stability, I used the tricks on Lar's Blog: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation 
    """
    def __init__(self, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.eps = eps
    def forward(self, output, target, screen = None, alpha = 1, gamma = 1):  
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here. 
        target.requires_grad_(requires_grad=False) # may be redundant
        p_hat = torch.sigmoid(output)
        if isinstance(screen, torch.Tensor):
            screen.requires_grad_(requires_grad=False) # may be redundant
            return -(weight * target[screen > 0.5] * F.logsigmoid(output[screen > 0.5]) + 
                    (1-target[screen > 0.5]) * F.logsigmoid(1 - output[screen > 0.5])).sum()
        elif screen == None:
            weight_a = alpha * target * (1 - p_hat)**gamma
            weight_b = (1 - alpha)  * (1 - target) * p_hat**gamma
            return -(weight * target * F.logsigmoid(output) + (1-target) * F.logsigmoid(1 - output)).sum()
        else: 
            print('screen argument type is wrong')
            return screen