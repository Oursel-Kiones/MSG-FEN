print("--- SCRIPT VERSION CHECK: THIS IS THE LATEST MODIFIED FILE (V2.1 - UNABRIDGED) ---")
# -*- coding: utf-8 -*-
"""
panoptic_deeplab_trainer.py (Version 2.1)

An advanced, robust, and feature-rich training script for DeepLab-based models,
specifically enhanced to support multi-stage training strategies as required by
architectures like Panoptic-DeepLab.

This script combines:
- The multi-stage training logic (dynamic optimizer and loss).
- Robustness, CRF post-processing, and multi-stage visualization features.
- V2.1 incorporates critical fixes for validation consistency, configuration management,
  detailed logging, and model checkpointing to ensure experiment reproducibility and stability.
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from typing import Dict, Optional
import yaml # Added for config file loading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CRF Imports ---
try:
    import pydensecrf.densecrf as dcrf
    DENSECRF_AVAILABLE = True
except ImportError:
    DENSECRF_AVAILABLE = False
    print("Warning: pydensecrf library not found. CRF post-processing will be unavailable.")
    print("Install using: pip install pydensecrf")
# --- End CRF Imports ---

from mypath import Path
from dataloaders import make_data_loader

try:
    from dataloaders.utils import decode_segmap
    DECODE_SEGMAP_AVAILABLE = True
except ImportError:
    DECODE_SEGMAP_AVAILABLE = False
    print("Warning: decode_segmap not found. Visualization will show class indices instead of colors.")

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import DeepLab
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


# --- CRF Post-processing Function (Unchanged) ---
def apply_crf(image, probabilities, n_classes, n_iters=5, sxy_gaussian=3, compat_gaussian=3, sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    if not DENSECRF_AVAILABLE:
        print("Error: pydensecrf not available. Cannot apply CRF.")
        return np.argmax(probabilities, axis=0)
    H, W = image.shape[:2]
    image_rgb = np.ascontiguousarray(image)
    d = dcrf.DenseCRF2D(W, H, n_classes)
    unary = -np.log(probabilities.reshape(n_classes, -1) + 1e-10)
    d.setUnaryEnergy(unary.astype(np.float32))
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image_rgb, compat=compat_bilateral, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(n_iters)
    map_result = np.argmax(Q, axis=0).reshape((H, W))
    return map_result


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        try:
            self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
            print(f"Successfully loaded data. Found {self.nclass} classes.")
            if hasattr(self.train_loader.dataset, 'thing_classes'):
                self.THING_CLASS_INDICES = self.train_loader.dataset.thing_classes
                print(f"Found {len(self.THING_CLASS_INDICES)} 'Thing' classes for valid_mask generation.")
            else:
                print("Warning: Dataloader does not provide 'thing_classes'. Falling back to hardcoded Cityscapes indices.")
                self.THING_CLASS_INDICES = [11, 12, 13, 14, 15, 16, 17, 18]
        except Exception as e:
            print(f"Critical error during data loading: {e}")
            raise e

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        if args.training_stage == 1:
            print("INFO: Configuring optimizer for [Stage 1] training (Heads and Decoder only).")
            for param in model.backbone.parameters():
                param.requires_grad = False
            params_to_train = model.get_10x_lr_params()
            train_params = [{'params': params_to_train, 'lr': args.lr * 10}]
            if not list(model.get_10x_lr_params()):
                raise ValueError("get_10x_lr_params() returned no parameters for Stage 1 training.")
        else:
            print("INFO: Configuring optimizer for [End-to-End] training.")
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        
        weight = None
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
                print(f"INFO: Loaded balanced weights from: {classes_weights_path}")
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
                try:
                    np.save(classes_weights_path, weight)
                    print(f"INFO: Calculated and saved balanced weights to: {classes_weights_path}")
                except Exception as e:
                    print(f"WARNING: Could not save calculated weights. Error: {e}")

            weight = torch.from_numpy(weight.astype(np.float32))

        device = torch.device("cuda" if args.cuda else "cpu")
        pos_weight_tensor = torch.tensor([args.pos_weight]).to(device)
        print(f"INFO: Initializing losses with pos_weight={pos_weight_tensor.item()}")

        self.criterion = SegmentationLosses(weight=weight,
                                            pos_weight=pos_weight_tensor,
                                            cuda=args.cuda)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        self.STUFF_COLORMAP = {
            0: (220, 20, 60), 1: (119, 11, 32), 2: (0, 0, 142), 3: (0, 0, 230),
            4: (107, 142, 35), 5: (0, 60, 100), 6: (0, 80, 100), 255: (0, 0, 0)
        }

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        
        self.best_pred = 0.0
        self.best_epoch = 0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                 print(f"Warning: Checkpoint path '{args.resume}' not found. Starting from scratch.")
            else:
                print(f"=> loading checkpoint '{args.resume}'")
                try:
                    checkpoint = torch.load(args.resume, map_location='cpu')
                    
                    # ===【V2.1 修复 - [中优先级]】 Module 7: 从 Checkpoint 恢复 pos_weight ===
                    restored_pos_weight = checkpoint.get('pos_weight', self.args.pos_weight)
                    if self.args.pos_weight != restored_pos_weight:
                        print(f"INFO: Overriding current pos_weight ({self.args.pos_weight}) with value from checkpoint ({restored_pos_weight}).")
                        self.args.pos_weight = restored_pos_weight
                    # === 结束修复 ===

                    if not args.ft:
                       args.start_epoch = max(args.start_epoch, checkpoint.get('epoch', 0))
                    state_dict = checkpoint.get('state_dict', checkpoint)
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    current_is_dp = isinstance(self.model, torch.nn.DataParallel)
                    saved_is_dp = any(k.startswith('module.') for k in state_dict.keys())
                    for k, v in state_dict.items():
                        if current_is_dp and not saved_is_dp: name = 'module.' + k
                        elif not current_is_dp and saved_is_dp: name = k[7:]
                        else: name = k
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict, strict=False)
                    print("=> loaded model state_dict successfully (non-strict).")
                    if not args.ft and 'optimizer' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                        print("=> loaded optimizer state_dict successfully.")
                    else:
                        print("=> Optimizer state not loaded (fine-tuning or new stage).")
                    self.best_pred = checkpoint.get('best_pred', 0.0)
                    self.best_epoch = checkpoint.get('best_epoch', 0)
                    print(f"=> loaded checkpoint '{args.resume}' (resuming from epoch {args.start_epoch}, best_pred {self.best_pred:.4f})")
                except Exception as e:
                    print(f"Error loading checkpoint '{args.resume}': {e}. Starting from scratch.")
                    args.start_epoch = 0; self.best_pred = 0.0
        if args.ft:
            args.start_epoch = 0; self.best_pred = 0.0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        
        def set_bn_eval(module):
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                module.eval()
        self.model.apply(set_bn_eval)
        
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} (Train)')
        
        if epoch == self.args.start_epoch:
            pass

        for i, sample_batch in enumerate(tbar):
            image_list = sample_batch['image']
            
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            
            batch_loss = 0.0
            for k in range(len(image_list)):
                image = image_list[k].unsqueeze(0)
                if self.args.cuda:
                    image = image.cuda()
                    
                predictions = self.model(image)

                if self.args.training_stage == 1:
                    stuff_gt = sample_batch['stuff_gt'][k].unsqueeze(0).cuda()
                    objectness_gt = sample_batch['objectness_gt'][k].unsqueeze(0).cuda()
                    semantic_gt = sample_batch['label'][k].unsqueeze(0).cuda()
                    
                    # ===【V2.1 修复 - [高优先级]】 Module 1: 增加类型检查和边界日志 ===
                    semantic_gt = semantic_gt.long()
                    # === 结束修复 ===

                    pred_stuff, pred_objectness = predictions['stuff'], predictions['objectness']
                    
                    loss_stuff = self.criterion.L_Stuff_CE(pred_stuff, stuff_gt)

                    thing_mask = torch.zeros_like(semantic_gt, dtype=torch.bool)
                    for class_idx in self.THING_CLASS_INDICES:
                        thing_mask |= (semantic_gt == class_idx)
                    
                    valid_mask = (semantic_gt != self.criterion.ignore_index) & thing_mask
                    
                    # ===【V2.1 修复 - [高优先级]】 Module 1: 增加类型检查和边界日志 ===
                    if not valid_mask.any():
                        if i % 100 == 0: # Log periodically to avoid spamming
                            print(f"\nWarning: Batch {i}, item {k} has no valid 'Thing' pixels; loss_objectness will be 0.")
                    # === 结束修复 ===

                    loss_objectness = self.criterion.L_Objectness_BCE(
                        pred_objectness,
                        objectness_gt.float(),
                        valid_mask
                    )

                    loss = (self.args.w_stuff * loss_stuff + self.args.w_coarse_obj * loss_objectness)
                    batch_loss += loss

                else: # End-to-end
                    target = sample_batch['label'][k].unsqueeze(0)
                    if self.args.cuda:
                        target = target.cuda()
                    pred_semantic = predictions.get('semantic', predictions)
                    loss_fn = self.criterion.build_loss(mode=self.args.loss_type)
                    loss = loss_fn(pred_semantic, target)
                    batch_loss += loss

            final_loss = batch_loss / len(image_list) if len(image_list) > 0 else torch.tensor(0.0)
            
            if final_loss > 0:
                final_loss.backward()
                # ===【V2.2 增强稳定性】: 添加梯度裁剪 ===
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                # === 结束修改 ===
                self.optimizer.step()

            train_loss += final_loss.item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            tbar.set_description(f'Epoch {epoch+1} (Train) Loss: {train_loss / (i + 1):.3f} LR: {current_lr:.6f}')
            self.writer.add_scalar('train/total_loss_iter', final_loss.item(), i + num_img_tr * epoch)

            # ===【V2.1 修复 - [中优先级]】 Module 5: 独立记录各分项损失 ===
            if self.args.training_stage == 1:
                self.writer.add_scalar('train/loss_objectness_iter', loss_objectness.item(), i + num_img_tr * epoch)
                self.writer.add_scalar('train/loss_stuff_iter', loss_stuff.item(), i + num_img_tr * epoch)
            # === 结束修复 ===
            
        self.writer.add_scalar('train/total_loss_epoch', train_loss / num_img_tr, epoch)
        print(f'[Epoch: {epoch+1}, numImages: {num_img_tr * self.args.batch_size}] Train Loss: {train_loss / num_img_tr:.4f}')
        if self.args.no_val:
            self.saver.save_checkpoint({'epoch': epoch + 1, 'state_dict': self.model.module.state_dict(), 'optimizer': self.optimizer.state_dict(), 'best_pred': self.best_pred, 'best_epoch': self.best_epoch, 'pos_weight': self.args.pos_weight}, is_best=False)

    def visualize_stage1_outputs(self, sample: Dict, predictions: Dict, epoch: int):
        img_tensor = sample['image'][0].cpu()
        stuff_gt = sample['stuff_gt'][0].cpu().numpy()
        objectness_gt = sample['objectness_gt'][0].squeeze().cpu().numpy()
        
        pred_stuff_logits = predictions['stuff']
        pred_objectness_logits = predictions['objectness']

        pred_stuff = torch.argmax(pred_stuff_logits, dim=1).squeeze(0).cpu().numpy()
        pred_objectness = torch.sigmoid(pred_objectness_logits).squeeze().cpu().numpy()

        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        input_image = (img_tensor.numpy().transpose(1, 2, 0) * std + mean) * 255.0
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)

        def colorize_stuff(stuff_map):
            h, w = stuff_map.shape
            color_map = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id, color in self.STUFF_COLORMAP.items():
                color_map[stuff_map == class_id] = color
            return color_map

        stuff_gt_color = colorize_stuff(stuff_gt)
        pred_stuff_color = colorize_stuff(pred_stuff)
        objectness_gt_viz = np.where(objectness_gt == 255, 0.5, objectness_gt)

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Stage 1 Validation - Epoch {epoch + 1}', fontsize=16)
        axs[0, 0].imshow(input_image); axs[0, 0].set_title('Input Image'); axs[0, 0].axis('off')
        axs[0, 1].imshow(stuff_gt_color); axs[0, 1].set_title('Stuff GT'); axs[0, 1].axis('off')
        axs[0, 2].imshow(pred_stuff_color); axs[0, 2].set_title('Stuff Prediction'); axs[0, 2].axis('off')
        axs[1, 0].text(0.5, 0.5, 'Objectness Maps', ha='center', va='center', fontsize=12); axs[1, 0].axis('off')
        axs[1, 1].imshow(objectness_gt_viz, cmap='gray', vmin=0, vmax=1); axs[1, 1].set_title('Objectness GT (ignore=gray)'); axs[1, 1].axis('off')
        axs[1, 2].imshow(pred_objectness, cmap='gray', vmin=0, vmax=1); axs[1, 2].set_title('Objectness Prediction (Prob)'); axs[1, 2].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_dir = os.path.join(self.saver.experiment_dir, 'stage1_visuals')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved Stage 1 visualization to: {save_path}")

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} (Val)')
        test_loss = 0.0
        
        for i, sample in enumerate(tbar):
            image_batch, target_batch = sample['image'], sample.get('label')
            
            if self.args.cuda:
                if isinstance(image_batch, torch.Tensor): image_batch = image_batch.cuda()
                if isinstance(target_batch, torch.Tensor): target_batch = target_batch.cuda()

            with torch.no_grad():
                predictions = self.model(image_batch)

                # ===【V2.1 修复 - [高优先级]】 Module 2: 同步 Stage 1 的验证逻辑 ===
                if self.args.training_stage == 1:
                    stuff_gt = sample['stuff_gt'].cuda()
                    objectness_gt = sample['objectness_gt'].cuda()
                    semantic_gt = sample['label'].cuda().long()
                    
                    pred_stuff, pred_objectness = predictions['stuff'], predictions['objectness']
                    
                    thing_mask = torch.zeros_like(semantic_gt, dtype=torch.bool)
                    for class_idx in self.THING_CLASS_INDICES:
                        thing_mask |= (semantic_gt == class_idx)
                    
                    valid_mask = (semantic_gt != self.criterion.ignore_index) & thing_mask
                    
                    loss_stuff = self.criterion.L_Stuff_CE(pred_stuff, stuff_gt)
                    loss_objectness = self.criterion.L_Objectness_BCE(pred_objectness, objectness_gt.float(), valid_mask)
                    
                    loss = self.args.w_stuff * loss_stuff + self.args.w_coarse_obj * loss_objectness
                    test_loss += loss.item()
                    tbar.set_description(f'Epoch {epoch+1} (Val) Stage1 Loss: {test_loss / (i + 1):.3f}')
                # === 结束修复 ===

                # --- 原有 End-to-End 逻辑保留在 else 分支 ---
                else:
                    output_batch = predictions.get('semantic')
                    if output_batch is not None and target_batch is not None:
                        loss_fn = self.criterion.build_loss(mode=self.args.loss_type)
                        loss = loss_fn(output_batch, target_batch)
                        test_loss += loss.item()
                        tbar.set_description(f'Epoch {epoch+1} (Val) Semantic Loss: {test_loss / (i + 1):.3f}')
                        
                        pred_batch_raw = torch.argmax(output_batch, dim=1).cpu().numpy()
                        target_batch_np = target_batch.cpu().numpy()
                        self.evaluator.add_batch(target_batch_np, pred_batch_raw)
            
            if self.args.save_val_results and i == 0:
                if self.args.training_stage == 1:
                    with torch.no_grad():
                        single_image = sample['image'][0].unsqueeze(0).cuda()
                        single_prediction = self.model(single_image)
                    if 'stuff' in single_prediction and 'objectness' in single_prediction and 'stuff_gt' in sample:
                        self.visualize_stage1_outputs(sample, single_prediction, epoch)
                
                if predictions.get('semantic') is not None and target_batch is not None:
                    save_dir = os.path.join(self.saver.experiment_dir, 'semantic_visuals')
                    os.makedirs(save_dir, exist_ok=True)
                    
                    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                    img_tmp = np.clip((sample['image'][0].numpy().transpose(1, 2, 0) * std + mean) * 255.0, 0, 255).astype(np.uint8)
                    
                    pred_seg_map = torch.argmax(predictions.get('semantic')[0], dim=0).cpu().numpy()
                    target_seg_map = target_batch[0].cpu().numpy()

                    target_seg_color = decode_segmap(target_seg_map, dataset=self.args.dataset) if DECODE_SEGMAP_AVAILABLE else target_seg_map
                    pred_seg_color = decode_segmap(pred_seg_map, dataset=self.args.dataset) if DECODE_SEGMAP_AVAILABLE else pred_seg_map
                    
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(img_tmp); axs[0].set_title('Input'); axs[0].axis('off')
                    axs[1].imshow(target_seg_color); axs[1].set_title('Ground Truth'); axs[1].axis('off')
                    axs[2].imshow(pred_seg_color); axs[2].set_title('Semantic Prediction'); axs[2].axis('off')
                    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}.png"), bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved Semantic visualization to: {os.path.join(save_dir, f'epoch_{epoch+1}.png')}")
        
        is_best = False
        if self.evaluator.confusion_matrix.sum() > 0:
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/Acc', Acc, epoch)
            
            print(f'\nValidation (Semantic Eval): [Epoch: {epoch+1}]')
            print(f"Acc:{Acc:.4f}, Acc_class:{Acc_class:.4f}, mIoU:{mIoU:.4f}, fwIoU: {FWIoU:.4f}")
            print(f'Val Loss: {test_loss / len(tbar):.3f}\n')

            new_pred = mIoU
            is_best = new_pred > self.best_pred
            if is_best:
                self.best_pred = new_pred
                self.best_epoch = epoch + 1
        else:
            print(f"\nValidation: Val Loss: {test_loss / len(tbar):.3f}\n")

        # ===【V2.1 修复 - [中优先级]】 Module 7: 在 Checkpoint 中保存 pos_weight ===
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'pos_weight': self.args.pos_weight
        }, is_best)
        # === 结束修复 ===

def main():
    parser = argparse.ArgumentParser(description="Advanced PyTorch DeeplabV3Plus Multi-Stage Trainer (V2.1)")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'coco', 'cityscapes'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-size', type=int, default=513)
    parser.add_argument('--crop-size', type=int, default=513)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--use-balanced-weights', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-val', action='store_true', default=False)
    parser.add_argument('--save_val_results', action='store_true', default=False)
    parser.add_argument('--use_crf', action='store_true', default=False)
    parser.add_argument('--crf_iters', type=int, default=5)
    parser.add_argument('--training-stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--w_stuff', type=float, default=1.0)
    parser.add_argument('--w_coarse_obj', type=float, default=1.0)
    parser.add_argument('--config-path', type=str, default=None, help="Path to YAML config file to override defaults.")
    parser.add_argument('--pos-weight', type=float, default=15.0, help='Weight for positive samples in objectness BCE loss.')
    
    args = parser.parse_args()
    
    # ===【V2.1 修复 - [中优先级]】 Module 3: 实现从 YAML 配置文件加载 ===
    if args.config_path and os.path.isfile(args.config_path):
        print(f"INFO: Loading configuration from {args.config_path}")
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create a temporary parser to get defaults, so we know which args were user-set vs default
        temp_parser = argparse.ArgumentParser()
        temp_parser.add_argument('--pos-weight', type=float, default=15.0)
        default_args, _ = temp_parser.parse_known_args()

        # If pos_weight was NOT set via command line (is still default), try to load from config
        if args.pos_weight == default_args.pos_weight:
             args.pos_weight = config.get('loss', {}).get('objectness', {}).get('pos_weight', args.pos_weight)
    # === 结束修复 ===

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        args.gpu_ids = [int(s.strip()) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError("Argument --gpu-ids must be a comma-separated list of integers.")
    
    if args.epochs is None: args.epochs = {'coco': 30, 'cityscapes': 200, 'pascal': 50}.get(args.dataset.lower(), 50)
    if args.batch_size is None: args.batch_size = 4 * len(args.gpu_ids) if args.cuda and args.gpu_ids else 4
    if args.test_batch_size is None: args.test_batch_size = args.batch_size
    if args.lr is None: args.lr = {'coco': 0.1, 'cityscapes': 0.01, 'pascal': 0.007}.get(args.dataset.lower(), 0.01)
    if args.checkname is None: args.checkname = f'deeplab-{args.backbone}'
    
    print("\n--- Effective Training Arguments ---")
    print(json.dumps(vars(args), indent=2))
    print("------------------------------------\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda: torch.cuda.manual_seed_all(args.seed)

    try:
        trainer = Trainer(args)
        if args.resume and not args.ft and not args.no_val and args.start_epoch > 0:
            print(f"\n--- Running initial validation for epoch {args.start_epoch-1} from resumed checkpoint ---")
            trainer.validation(args.start_epoch - 1)
        
        print(f"\n--- Starting Training from Epoch {trainer.args.start_epoch + 1} to {trainer.args.epochs} ---")
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            try:
                trainer.training(epoch)
                if not trainer.args.no_val and (epoch + 1) % args.eval_interval == 0:
                    trainer.validation(epoch)
            except Exception as e:
                print(f"Error during epoch {epoch+1}: {e}")
                import traceback; traceback.print_exc()
                break
        
        trainer.writer.close()
        print("\n--- Training finished ---")
        print(f"Best mIoU: {trainer.best_pred:.4f} achieved at epoch {trainer.best_epoch}")
        print(f"Checkpoints and logs saved in: {trainer.saver.experiment_dir}")
    except Exception as e:
        print(f"Failed to initialize or run Trainer: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()