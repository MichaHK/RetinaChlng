import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
            return None
        return 1 - 2 * (numerator / denominator)

        
class WCE(nn.Module):
    """My custom WEIGHTED Cross Entropy loss function. It first runs the output through a Sigmoid, then computes the
    cross entropy. Weight is in range [0,1]. To decrease the number of false negatives, use a large weight.
    To decrease the number false positives, set to small.
    """
    def __init__(self,  weight = torch.tensor(0.5, requires_grad=False)):
        super().__init__()
        self.weight = weight
    def forward(self, output, target, screen = None):
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here. 
        target.requires_grad_(requires_grad=False) # may be redundant
        weight = self.weight
        assert 0 <= weight <= 1, 'weight out of range:' + str(weight)
        if isinstance(screen, torch.Tensor):
            screen.requires_grad_(requires_grad=False) # may be redundant
            return -(weight * target[screen > 0.5] * F.logsigmoid(output[screen > 0.5]) + 
                    (1- weight)*(1-target[screen > 0.5]) * F.logsigmoid(-output[screen > 0.5])).sum()
        elif screen == None:
            return -(weight * target * F.logsigmoid(output) + (1 - weight)*(1-target) * F.logsigmoid( - output)).sum()
        else: 
            print('screen argument type is wrong')
            return screen
    def __repr__(self):
        return self.__class__.__name__ + '(Weight={0})'.format(self.weight)

class FocalLoss(nn.Module):
    """My custom Focal Loss function. It runs on an output that was not normalized yet using a sigmoid. 
    For stability, I used the tricks on Lar's Blog: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation 
    """
    def __init__(self, alpha=torch.tensor(0.5, requires_grad=False), gamma = 1): # maybe: __init__(self, alpha = torch.tensor(0.5, requires_grad=False), gamma = 1);
        super().__init__() # maybe: super().__init__();
        self.alpha , self.gamma = alpha , gamma
    def forward(self, output, target, screen = None):
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here. 
        target.requires_grad_(requires_grad=False) # may be redundant
        p_hat = torch.sigmoid(output)
#         import pdb; pdb.set_trace()
        weight_a = self.alpha * target[screen > 0.5] * (1 - p_hat)**self.gamma
        weight_b = (1 - self.alpha) * (1 - target) * p_hat**self.gamma
        if isinstance(screen, torch.Tensor):
            screen.requires_grad_(requires_grad=False) # may be redundant
            return -((weight_a*F.logsigmoid(output) + weight_b * F.logsigmoid(-output))[screen > 0.5]).sum()
        elif screen == None:

            return -(weight_a*F.logsigmoid(output) + weight_b * F.logsigmoid(-output)).sum()
        else: 
            print('screen argument type is wrong')
            return screen

class Sensitivity(nn.Module):
    """\frac{TP}{TP+FN}
    This is the continuous version of the Sensitivity metric. Feed with non-activated last layer of classification NN.
    For evaluation, feed with thresholded NN output, where:
    seg_output[torch.sigmoid(seg_output) < threshold] = -1e10
    seg_output[torch.sigmoid(seg_output) > threshold] = +1e10
    """
    def __init__(self):
        super().__init__() # maybe: super().__init__();

    def forward(self, output, target, screen = None):
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here.
        target.requires_grad_(requires_grad=False) # may be redundant
        p_hat = torch.sigmoid(output)
        if isinstance(screen, torch.Tensor):
            TP = (p_hat[screen > 0.5] * target[screen > 0.5]).sum()
            FN = (target[screen > 0.5] * (1-p_hat[screen > 0.5])).sum()
            return TP/(TP+FN)
        elif screen == None:
            TP = (p_hat * target).sum()
            FN = (target * (1 - p_hat)).sum()
            return TP / (TP + FN)
        else:
            print('screen argument type is wrong')
            return screen

class specificity(nn.Module):
    """\frac{TN}{TN+FP}
    This is the continuous version of the specificity metric. Feed with non-activated last layer of classification NN.
    For evaluation, feed with thresholded NN output, where:
    seg_output[torch.sigmoid(seg_output) < threshold] = -1e10
    seg_output[torch.sigmoid(seg_output) > threshold] = +1e10
    """
    def __init__(self):
        super().__init__() # maybe: super().__init__();

    def forward(self, output, target, screen = None):
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here.
        target.requires_grad_(requires_grad=False) # may be redundant
        p_hat = torch.sigmoid(output)
        if isinstance(screen, torch.Tensor):
            TN = ((1-p_hat[screen > 0.5]) * (1-target[screen > 0.5])).sum()
            FP = ((1-target[screen > 0.5]) * p_hat[screen > 0.5]).sum()
            return TN/(TN+FP)
        elif screen == None:
            TN = ((1 - p_hat) * (1 - target)).sum()
            FP = ((1 - target) * p_hat).sum()
            return TN/(TN+FP)
        else:
            print('screen argument type is wrong')
            return screen

class Accuracy(nn.Module):
    """\frac{TP+TN}{All}
    """
    def __init__(self):
        super().__init__() # maybe: super().__init__();

    def forward(self, output, target, screen = None):
        output.requires_grad_(requires_grad=True) # absolutely necessary! Not in Jupyter, but yes in here.
        target.requires_grad_(requires_grad=False) # may be redundant
        p_hat = torch.sigmoid(output)
        if isinstance(screen, torch.Tensor):
            TP = (p_hat[screen > 0.5] * target[screen > 0.5]).sum()
            TN = ((1-p_hat[screen > 0.5]) * (1-target[screen > 0.5])).sum()
            All = screen.sum()
            return (TP+TN)/All
        elif screen == None:
            TP = (p_hat * target).sum()
            TN = ((1-p_hat) * (1-target)).sum()
            All = np.prod(target.shape)
            return (TP+TN)/All
        else:
            print('screen argument type is wrong')
            return screen
