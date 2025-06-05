import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

class GT_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GT_CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, gt_pre, out, target):

        # Compute the primary loss (ce + Dice) for the output
        celoss = self.ce(out, target)

        # Compute losses for ground truth pyramid predictions (with different weights)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (self.ce(gt_pre5, target) * 0.1 +
                   self.ce(gt_pre4, target) * 0.2 +
                   self.ce(gt_pre3, target) * 0.3 +
                   self.ce(gt_pre2, target) * 0.4 +
                   self.ce(gt_pre1, target) * 0.5)

        # Return the combined loss
        return celoss + gt_loss


