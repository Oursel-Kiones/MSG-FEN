# 文件路径: /workspace/deep参考1/utils/loss.py

"""
This module defines a collection of loss functions for semantic and panoptic
segmentation tasks, including a robust mechanism for handling class weights
for specific sub-tasks like 'stuff' segmentation.
"""

import torch
import torch.nn as nn
from typing import Optional

CITYSCAPES_STUFF_CLASS_INDICES =[0, 1, 2, 3, 8, 9, 10]

class SegmentationLosses(object):
    """
    A collection of loss functions that can be pre-initialized with class weights
    and other parameters. It handles weight extraction for specific sub-tasks.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, pos_weight: Optional[torch.Tensor] = None,
                 ignore_index: int = 255, cuda: bool = False):
        """
        Initializes the loss functions.
        """
        reduction = 'mean'
        self.ignore_index = ignore_index
        self.weight = weight
        self.pos_weight = pos_weight
        self.cuda = cuda

        stuff_weight = None
        if self.weight is not None:
            print("INFO: Global weights provided. Attempting to extract weights for stuff classes...")
            if len(self.weight) == 19:
                try:
                    stuff_weight = self.weight[CITYSCAPES_STUFF_CLASS_INDICES]
                    print(f"      - Success: Extracted {len(stuff_weight)} weights for stuff loss.")
                except IndexError:
                    print("      - WARNING: Index out of bounds during stuff weight extraction. Stuff loss will proceed without weights.")
                    stuff_weight = None
            else:
                print(f"      - WARNING: Expected global weight of length 19, but got {len(self.weight)}. Stuff loss will proceed without weights.")

        self.stuff_ce = nn.CrossEntropyLoss(weight=stuff_weight, ignore_index=self.ignore_index, reduction=reduction)
        self.objectness_bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')
        self.semantic_ce = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=reduction)
        self.focal_base_ce = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')

        if self.cuda:
            self.stuff_ce = self.stuff_ce.cuda()
            self.objectness_bce = self.objectness_bce.cuda()
            self.semantic_ce = self.semantic_ce.cuda()
            self.focal_base_ce = self.focal_base_ce.cuda()

    def L_Stuff_CE(self, logit_stuff: torch.Tensor, target_stuff: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Cross-Entropy loss for 'stuff' classes.
        """
        return self.stuff_ce(logit_stuff, target_stuff.long())

    def L_Objectness_BCE(self, logit_objectness: torch.Tensor, target_objectness: torch.Tensor,
                         valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Binary Cross-Entropy loss for objectness, with support for
        a validity mask and pre-configured pos_weight.
        """
        valid_mask = valid_mask.bool()
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logit_objectness.device, requires_grad=True)

        unreduced_loss = self.objectness_bce(logit_objectness, target_objectness)
        
        # ===【V2.2 BUGFIX - 高优先级】 Shape Mismatch Correction ===
        if unreduced_loss.shape != valid_mask.shape:
            if len(unreduced_loss.shape) == 4 and len(valid_mask.shape) == 3:
                 valid_mask = valid_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unhandled shape mismatch between loss {unreduced_loss.shape} and mask {valid_mask.shape}")
        # === 结束修复 ===

        masked_loss = unreduced_loss[valid_mask]
        
        return masked_loss.mean()

    def CrossEntropyLoss(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.semantic_ce(logit, target.long())

    # =========================================================================
    # 【核心重构】：满血版 Focal Loss，拔掉背景稀释的保险栓
    # =========================================================================
    def FocalLoss(self, logit: torch.Tensor, target: torch.Tensor, gamma: int = 2, alpha: float = 0.5) -> torch.Tensor:
        # ce_loss 形状为 [B, H, W]，其中 target 为 ignore_index (255) 的地方值为 0
        ce_loss = self.focal_base_ce(logit, target.long())
        
        pt = torch.exp(-ce_loss)
        focal_term = ((1 - pt) ** gamma) * ce_loss
        
        if alpha is not None:
            focal_term *= alpha
            
        # 【关键修复】：只挑选出非 255 的有效像素求均值！绝不除以全图面积！
        valid_mask = (target != self.ignore_index)
        
        if valid_mask.any():
            return focal_term[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=logit.device, requires_grad=True)
    
    def build_loss(self, mode: str = 'ce') -> callable:
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError(f"Loss mode '{mode}' is not implemented.")