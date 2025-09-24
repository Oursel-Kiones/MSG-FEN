# /workspace/deep参考1/utils/loss.py (V2.2 - Bugfix)

"""
This module defines a collection of loss functions for semantic and panoptic
segmentation tasks, including a robust mechanism for handling class weights
for specific sub-tasks like 'stuff' segmentation.
"""

import torch
import torch.nn as nn
from typing import Optional

CITYSCAPES_STUFF_CLASS_INDICES = [0, 1, 2, 3, 8, 9, 10]


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

        if self.cuda:
            self.stuff_ce = self.stuff_ce.cuda()
            self.objectness_bce = self.objectness_bce.cuda()
            self.semantic_ce = self.semantic_ce.cuda()

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
        
        # ===【V2.2 BUGFIX - [高优先级]】 Shape Mismatch Correction ===
        # 描述：模型的预测 (unreduced_loss) 形状为 [B, 1, H, W]，而来自标签的 mask 形状为 [B, H, W]。
        #      这会导致索引错误。我们必须在索引前，通过 unsqueeze(1) 为 mask 增加一个通道维度，
        #      使其形状与 unreduced_loss 保持一致。
        if unreduced_loss.shape != valid_mask.shape:
            if len(unreduced_loss.shape) == 4 and len(valid_mask.shape) == 3:
                 valid_mask = valid_mask.unsqueeze(1)
            else:
                # Handle other potential mismatches if necessary, or raise an error
                raise ValueError(f"Unhandled shape mismatch between loss {unreduced_loss.shape} and mask {valid_mask.shape}")
        # === 结束修复 ===

        masked_loss = unreduced_loss[valid_mask]
        
        return masked_loss.mean()

    def build_loss(self, mode: str = 'ce') -> callable:
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.semantic_ce(logit, target.long())

    def FocalLoss(self, logit: torch.Tensor, target: torch.Tensor, gamma: int = 2, alpha: float = 0.5) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        
        return loss.mean()