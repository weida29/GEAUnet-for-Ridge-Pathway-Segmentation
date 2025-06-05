import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.my_loss.lovasz_losses as L


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

class GT_CrossEntropyLoss(nn.Module):
    def __init__(self,cls_weights, num_classes=21):
        super(GT_CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)

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


class GT_FocalLoss(nn.Module):#对于中间深度监督层使用celoss，尾部输出层使用FocalLoss。
    def __init__(self,cls_weights, num_classes=21):
        super(GT_FocalLoss, self).__init__()
        self.cls_weights = cls_weights
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)

    def forward(self, gt_pre, out, target):
        cls_weights, num_classes = self.cls_weights, self.num_classes
        # Compute the primary loss (ce + Dice) for the output
        focalloss = Focal_Loss(out, target, cls_weights, num_classes)

        # Compute losses for ground truth pyramid predictions (with different weights)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (self.ce(gt_pre5, target) * 0.1 +
                   self.ce(gt_pre4, target) * 0.2 +
                   self.ce(gt_pre3, target) * 0.3 +
                   self.ce(gt_pre2, target) * 0.4 +
                   self.ce(gt_pre1, target) * 0.5)

        # Return the combined loss
        return focalloss + gt_loss


class GT_LovaszLoss(nn.Module):#对于中间深度监督层使用celoss，尾部输出层使用lovaszloss(cvpr2018)。在lovasz论文中也提到，和celoss一起用效果会好
    def __init__(self,cls_weights, num_classes=21):
        super(GT_LovaszLoss, self).__init__()
        self.cls_weights = cls_weights
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)

    def forward(self, gt_pre, out, target):
        cls_weights, num_classes = self.cls_weights, self.num_classes
        # Compute the primary loss (ce + lovase) for the output
        lovaseloss = L.lovasz_softmax(out, target, classes='present', ignore=0)#按作者原文的demo，需要忽略背景
        # Compute losses for ground truth pyramid predictions (with different weights)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (self.ce(gt_pre5, target) * 0.1 +
                   self.ce(gt_pre4, target) * 0.2 +
                   self.ce(gt_pre3, target) * 0.3 +
                   self.ce(gt_pre2, target) * 0.4 +
                   self.ce(gt_pre1, target) * 0.5)

        # Return the combined loss
        return lovaseloss + gt_loss