# /workspace/deep参考1/utils/loss.py (最终确认版 V2.3)

"""
This module defines a collection of loss functions for semantic and panoptic
segmentation tasks, including a robust mechanism for handling class weights
for specific sub-tasks like 'stuff' segmentation.
"""

import torch
import torch.nn as nn
from typing import Optional

# Hardcoded indices for Cityscapes 'stuff' classes. This is used to extract
# relevant class weights if a global weight tensor is provided.
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

        Args:
            weight (torch.Tensor, optional): A tensor of weights for each class for CE loss.
            pos_weight (torch.Tensor, optional): A weight for positive samples in BCE loss.
            ignore_index (int): The label index to be ignored during loss calculation. Defaults to 255.
            cuda (bool): Whether to move loss functions to the GPU.
        """
        reduction = 'mean'
        self.ignore_index = ignore_index
        self.weight = weight
        self.pos_weight = pos_weight
        self.cuda = cuda

        # --- Initialize Loss Components ---

        # 1. Stuff Cross-Entropy Loss
        stuff_weight = None
        if self.weight is not None:
            print("INFO: Global weights provided. Attempting to extract weights for stuff classes...")
            # Cityscapes has 19 classes in total
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

        # 2. Objectness BCE Loss (with logits)
        self.objectness_bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')
        
        # 3. Standard Semantic Cross-Entropy Loss (used for 'things' in Stage 2 and end-to-end)
        # This will be the loss used by the CrossEntropyLoss method below.
        self.semantic_ce = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=reduction)
        
        # 4. Focal Loss (optional, uses 'none' reduction for manual calculation)
        self.focal_base_ce = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')

        if self.cuda:
            self.stuff_ce = self.stuff_ce.cuda()
            self.objectness_bce = self.objectness_bce.cuda()
            self.semantic_ce = self.semantic_ce.cuda()
            self.focal_base_ce = self.focal_base_ce.cuda()

    def L_Stuff_CE(self, logit_stuff: torch.Tensor, target_stuff: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Cross-Entropy loss for 'stuff' classes (used in Stage 1).
        """
        return self.stuff_ce(logit_stuff, target_stuff.long())

    def L_Objectness_BCE(self, logit_objectness: torch.Tensor, target_objectness: torch.Tensor,
                         valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Binary Cross-Entropy loss for objectness, with support for
        a validity mask and pre-configured pos_weight (used in Stage 1).
        """
        valid_mask = valid_mask.bool()
        
        # If no valid pixels are in the mask, return a zero loss.
        if not valid_mask.any():
            return torch.tensor(0.0, device=logit_objectness.device, requires_grad=True)

        # Calculate unreduced loss for all pixels
        unreduced_loss = self.objectness_bce(logit_objectness, target_objectness)
        
        # [V2.2 BUGFIX] Ensure mask shape matches loss shape before indexing.
        # The model's prediction might be [B, 1, H, W], while the mask is [B, H, W].
        if unreduced_loss.shape != valid_mask.shape:
            if len(unreduced_loss.shape) == 4 and len(valid_mask.shape) == 3:
                 valid_mask = valid_mask.unsqueeze(1) # Add channel dimension
            else:
                raise ValueError(f"Unhandled shape mismatch between loss {unreduced_loss.shape} and mask {valid_mask.shape}")
        
        # Apply the mask and calculate the mean of the valid pixels
        masked_loss = unreduced_loss[valid_mask]
        return masked_loss.mean()

    def CrossEntropyLoss(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the standard Cross-Entropy loss (used in Stage 2 for 'things').
        This is the primary loss for the final segmentation task.
        """
        return self.semantic_ce(logit, target.long())

    def FocalLoss(self, logit: torch.Tensor, target: torch.Tensor, gamma: int = 2, alpha: float = 0.5) -> torch.Tensor:
        """
        Calculates the Focal Loss.
        """
        # Calculate log probabilities using the base CE loss with 'none' reduction
        logpt = -self.focal_base_ce(logit, target.long())
        pt = torch.exp(logpt)

        # Calculate the focal loss term
        focal_term = -((1 - pt) ** gamma) * logpt
        
        # Apply alpha weighting if provided
        if alpha is not None:
            focal_term *= alpha
        
        return focal_term.mean()
    
    def build_loss(self, mode: str = 'ce') -> callable:
        """
        A helper to select the loss function by name.
        """
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError(f"Loss mode '{mode}' is not implemented.")