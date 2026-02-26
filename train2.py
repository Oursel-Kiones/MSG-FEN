# 文件路径: /workspace/deep参考1/train2.py (最终修复版 V1.7)

print("--- SCRIPT FOR STAGE 2 TRAINING (MSG-FENet) V1.7 (BN Fix) ---")
# -*- coding: utf-8 -*-
"""
train2.py (Version 1.7 - BN Fix)

This script is specifically adapted for Stage 2 training of the MSG-FENet architecture.
Key fixes in this version:
- [V1.7] CRITICAL FIX: The frozen Stage 1 model's Batch Normalization layers are now
  explicitly set to evaluation mode during training. This prevents the corruption
  of their running statistics and resolves the 'Val Loss: nan' issue.
- [V1.6] Robust label validation for variable-sized images.
- [V1.5] Integrated TensorBoard monitoring and other diagnostics.
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

# --- Imports from existing project structure ---
from mypath import Path
from dataloaders import make_data_loader
from dataloaders.utils import decode_segmap
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import DeepLab
from modeling.msg_fenet import MSG_FENet_Stage2
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        self.labels_verified = False

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        
        print("=> [Stage 2] Instantiating Stage 1 model structure as a base...")
        stage1_base_model = DeepLab(
            backbone=args.backbone, output_stride=args.out_stride,
            num_classes=args.num_total_classes,
            num_stuff_classes=args.num_stuff_classes,
            num_object_classes=args.num_object_classes,
            sync_bn=args.sync_bn, freeze_bn=args.freeze_bn
        )

        print(f"=> [Stage 2] Instantiating MSG_FENet_Stage2 with {args.num_object_classes} 'thing' classes...")
        model = MSG_FENet_Stage2(stage1_model=stage1_base_model,
                                 num_thing_classes=args.num_object_classes)

        if not args.stage1_checkpoint or not os.path.isfile(args.stage1_checkpoint):
            raise FileNotFoundError(f"Required Stage 1 checkpoint not found at: {args.stage1_checkpoint}")
        
        print(f"=> [Stage 2] Loading pretrained weights from Stage 1 checkpoint: '{args.stage1_checkpoint}'")
        checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
        
        model.stage1_model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("   ✅ SUCCESS: Stage 1 weights loaded into the frozen base model successfully (strict check passed).")

        print("=> [Stage 2] Configuring optimizer to train ONLY new Stage 2 modules...")
        trainable_params = [
            {'params': model.low_level_compressor.parameters(), 'lr': args.lr},
            {'params': model.feature_fusion.parameters(), 'lr': args.lr},
            {'params': model.decoder_object.parameters(), 'lr': args.lr}
        ]
        
        num_trainable_params = sum(p.numel() for group in trainable_params for p in group['params'])
        if num_trainable_params == 0:
            raise RuntimeError("FATAL: No trainable parameters found!")
        print(f"   - Found {num_trainable_params / 1e6:.2f}M trainable parameters.")
            
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Assuming the fix is applied in utils/loss.py or the class handles ignore_index
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda, ignore_index=255)
        self.model, self.optimizer = model, optimizer
        
        self.evaluator = Evaluator(self.args.num_object_classes) 
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        
        self.best_pred = 0.0
        self.best_epoch = 0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
            print(f"=> Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            self.args.start_epoch = checkpoint['epoch']
            self.model.module.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.best_epoch = checkpoint['best_epoch']
            print(f"   ...resumed to epoch {self.args.start_epoch} with best mIoU {self.best_pred:.4f} at epoch {self.best_epoch}")

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()

        # --- [V1.7 CRITICAL FIX] ---
        # Force the frozen Stage 1 model's BN layers to be in eval mode.
        # This prevents them from updating their running statistics, which is the
        # root cause of the 'Val Loss: nan' issue.
        def set_bn_eval(module):
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                module.eval()
        
        if hasattr(self.model, 'module'):
            self.model.module.stage1_model.apply(set_bn_eval)
        else:
            self.model.stage1_model.apply(set_bn_eval)
        # --- [END OF FIX] ---

        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} (Train)')
        
        for i, sample_batch in enumerate(tbar):
            image_list, target_things_list = sample_batch['image'], sample_batch['object_gt']

            if not self.labels_verified:
                try:
                    all_labels_flat = torch.cat([t.flatten() for t in target_things_list])
                    unique_labels = torch.unique(all_labels_flat)
                    
                    print("\n--- [标签验证] ---")
                    print(f"Epoch: {epoch}, Batch: {i}")
                    print(f"在当前批次中发现的唯一标签值: {unique_labels}")
                    print(f"预期值应在 [0, ..., {self.args.num_object_classes - 1}] 范围内，外加一个忽略标签 (通常是 255)。")

                    invalid_labels = [
                        label.item() for label in unique_labels 
                        if (label < 0 or label >= self.args.num_object_classes) and label != 255
                    ]
                    
                    if invalid_labels:
                        print(f"\033[91m[严重警告] 发现无效标签值: {invalid_labels}\033[0m")
                        print("请立即停止训练，检查数据预处理流程！")
                    else:
                        print("\033[92m[成功] 标签值范围检查通过。\033[0m")
                    
                    self.labels_verified = True
                    print("---------------------\n")
                except Exception as e:
                    print(f"\n\033[91m[标签验证失败] 检查代码时发生错误: {e}\033[0m\n")
                    self.labels_verified = True

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            batch_loss = 0.0
            for k in range(len(image_list)):
                image = image_list[k].unsqueeze(0)
                target_things = target_things_list[k].unsqueeze(0)

                if self.args.cuda:
                    image = image.cuda()
                    target_things = target_things.cuda()

                output = self.model(image)
                logit_things = output['things']
                
                loss = self.criterion.CrossEntropyLoss(logit_things, target_things)
                batch_loss += loss

            final_loss = batch_loss / len(image_list) if len(image_list) > 0 else torch.tensor(0.0)
            
            if final_loss > 0 and torch.isfinite(final_loss):
                final_loss.backward()
                
                if i % 100 == 0:
                    global_step = epoch * num_img_tr + i
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            tag = name.replace('module.', '').replace('.', '/')
                            self.writer.add_histogram(f'grads/{tag}', param.grad.cpu(), global_step)
                
                torch.nn.utils.clip_grad_norm_((p for group in self.optimizer.param_groups for p in group['params']), max_norm=1.0)
                self.optimizer.step()

            train_loss += final_loss.item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            tbar.set_description(f'Epoch {epoch+1} (Train) Things Loss: {train_loss / (i + 1):.3f} LR: {current_lr:.6f}')
            self.writer.add_scalar('train/things_loss_iter', final_loss.item(), i + num_img_tr * epoch)
            
        self.writer.add_scalar('train/things_loss_epoch', train_loss / num_img_tr, epoch)
        print(f'[Epoch: {epoch+1}] Train Loss: {train_loss / num_img_tr:.4f}')

        if self.args.no_val:
            self.saver.save_checkpoint({
                'epoch': epoch + 1, 'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'best_pred': self.best_pred,
                'best_epoch': self.best_epoch}, is_best=False)

    def visualize_stage2_outputs(self, sample: Dict, predictions: Dict, epoch: int):
        img_tensor = sample['image'][0].cpu()
        target_seg_map = sample['object_gt'][0].cpu().numpy()
        
        single_logit = predictions['things'][0].unsqueeze(0)
        pred_seg_map = torch.argmax(single_logit, dim=1).squeeze(0).cpu().numpy()

        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        input_image = (img_tensor.numpy().transpose(1, 2, 0) * std + mean) * 255.0
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)

        target_seg_color = decode_segmap(target_seg_map, dataset=self.args.dataset, is_thing=True)
        pred_seg_color = decode_segmap(pred_seg_map, dataset=self.args.dataset, is_thing=True)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Stage 2 Validation - Epoch {epoch + 1}', fontsize=16)
        axs[0].imshow(input_image); axs[0].set_title('Input Image'); axs[0].axis('off')
        axs[1].imshow(target_seg_color); axs[1].set_title('Things Ground Truth'); axs[1].axis('off')
        axs[2].imshow(pred_seg_color); axs[2].set_title('Things Prediction'); axs[2].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_dir = os.path.join(self.saver.experiment_dir, 'stage2_visuals')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved Stage 2 visualization to: {save_path}")

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} (Val)')
        test_loss = 0.0
        
        for i, sample_batch in enumerate(tbar):
            image_list, target_things_list = sample_batch['image'], sample_batch['object_gt']
            
            batch_logits = []
            for k in range(len(image_list)):
                image = image_list[k].unsqueeze(0)
                target_things = target_things_list[k].unsqueeze(0)
                if self.args.cuda:
                    image, target_things = image.cuda(), target_things.cuda()

                with torch.no_grad():
                    output = self.model(image)
                    logit_things = output['things']
                    
                    loss = self.criterion.CrossEntropyLoss(logit_things, target_things)
                    test_loss += loss.item()
                    
                    pred_things = torch.argmax(logit_things, dim=1)
                    
                    target_np = target_things.cpu().numpy()
                    pred_np = pred_things.cpu().numpy()
                    self.evaluator.add_batch(target_np, pred_np)
                    
                    if i == 0:
                        batch_logits.append(logit_things.cpu())
            
            tbar.set_description(f'Epoch {epoch+1} (Val) Things Loss: {test_loss / ((i + 1) * len(image_list)):.3f}')

            if self.args.save_val_results and i == 0:
                vis_sample = {'image': image_list, 'object_gt': target_things_list}
                vis_preds = {'things': torch.cat(batch_logits, dim=0)}
                self.visualize_stage2_outputs(vis_sample, vis_preds, epoch)
        
        total_val_samples = len(self.val_loader.dataset)
        final_val_loss = test_loss / total_val_samples if total_val_samples > 0 else 0
        
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        
        self.writer.add_scalar('val/mIoU_things', mIoU, epoch)
        self.writer.add_scalar('val/Acc_things', Acc, epoch)
        self.writer.add_scalar('val/loss_epoch', final_val_loss, epoch)
        
        if np.isnan(final_val_loss):
            print("\n\033[91m[严重问题] Validation Loss is NaN! 训练发散，请检查学习率或数据问题。\033[0m\n")
        
        print(f'\nValidation (Things Eval): [Epoch: {epoch+1}] mIoU: {mIoU:.4f}')
        print(f'  - Other metrics: Acc:{Acc:.4f}, Acc_class:{Acc_class:.4f}, fwIoU: {FWIoU:.4f}')
        print(f'  - Val Loss: {final_val_loss:.3f}\n')

        new_pred = mIoU
        is_best = new_pred > self.best_pred
        if is_best:
            self.best_pred = new_pred
            self.best_epoch = epoch + 1
            print(f"*** New best mIoU (things) found: {self.best_pred:.4f} at epoch {self.best_epoch} ***")

        self.saver.save_checkpoint({
            'epoch': epoch + 1, 'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'best_pred': self.best_pred,
            'best_epoch': self.best_epoch}, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch MSG-FENet Stage 2 Trainer (Corrected)")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'coco', 'cityscapes'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-size', type=int, default=513)
    parser.add_argument('--crop-size', type=int, default=513)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--checkname', type=str, default='msg-fenet-stage2')
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-val', action='store_true', default=False)
    parser.add_argument('--save_val_results', action='store_true', default=True)

    parser.add_argument('--training-stage', type=int, default=2, choices=[2], help="Must be 2 for this script.")
    parser.add_argument('--stage1-checkpoint', type=str, required=True, help="Path to the pretrained Stage 1 checkpoint.")
    parser.add_argument('--resume', type=str, default=None, help="Path to a Stage 2 checkpoint to resume training.")
    parser.add_argument('--ft', action='store_true', default=False, help="Fine-tuning mode.")
    
    args = parser.parse_args()

    if args.dataset == 'cityscapes':
        args.num_total_classes = 19
        args.num_stuff_classes = 7
        args.num_object_classes = 12
    else:
        raise NotImplementedError(f"Class numbers for dataset '{args.dataset}' are not defined.")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    if args.batch_size is None: args.batch_size = 4 * len(args.gpu_ids)
    
    print("\n--- MSG-FENet Stage 2 Training Arguments (Corrected) ---")
    print(json.dumps(vars(args), indent=2))
    print("------------------------------------------------------\n")

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    print(f"--- Starting Training from Epoch {trainer.args.start_epoch + 1} to {trainer.args.epochs} ---")
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and (epoch + 1) % args.eval_interval == 0:
            trainer.validation(epoch)
    
    trainer.writer.close()
    print("\n--- Training finished ---")
    print(f"Best mIoU (things): {trainer.best_pred:.4f} achieved at epoch {trainer.best_epoch}")

if __name__ == "__main__":
    main()