print("--- SCRIPT FOR STAGE 2 TRAINING (MSG-FENet) V1.0 ---")
# -*- coding: utf-8 -*-
"""
train2.py (Version 1.0)

This script is specifically adapted for Stage 2 training of the MSG-FENet architecture.
It performs the following key functions:
1.  Instantiates the composite MSG_FENet_Stage2 model.
2.  Loads pretrained weights from a Stage 1 checkpoint into the model's frozen backbone.
3.  Configures the optimizer to train ONLY the new Stage 2 modules (feature fusion and object decoder).
4.  Calculates loss based on the fine-grained 'thing' class predictions.
5.  Includes visualization for Stage 2 semantic segmentation outputs.
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from typing import Dict
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

# --- Imports from existing project structure (Paths remain unchanged) ---
from mypath import Path
from dataloaders import make_data_loader
from dataloaders.utils import decode_segmap
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

# ### STAGE 2 MODIFICATION ###: Import our new custom network
from modeling.deeplab import DeepLab # Still needed as the base for Stage 1
from modeling.msg_fenet import MSG_FENet_Stage2


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        
        # ### STAGE 2 MODIFICATION ###: Model Definition and Weight Loading
        # 1. Instantiate Stage 1 model structure as a placeholder
        print("=> [Stage 2] Instantiating Stage 1 model structure as a base...")
        stage1_base_model = DeepLab(num_classes=args.num_classes_stage1, # Stage 1 has its own output classes
                                    backbone=args.backbone,
                                    output_stride=args.out_stride,
                                    sync_bn=args.sync_bn,
                                    freeze_bn=args.freeze_bn)

        # 2. Instantiate the complete MSG_FENet_Stage2 model
        print(f"=> [Stage 2] Instantiating MSG_FENet_Stage2 with {args.num_classes_stage2} 'thing' classes...")
        model = MSG_FENet_Stage2(stage1_model=stage1_base_model,
                                 num_thing_classes=args.num_classes_stage2)

        # 3. Load pretrained weights from the Stage 1 checkpoint
        if not args.stage1_checkpoint or not os.path.isfile(args.stage1_checkpoint):
            raise FileNotFoundError(f"Required Stage 1 checkpoint not found at: {args.stage1_checkpoint}")
        
        print(f"=> [Stage 2] Loading pretrained weights from Stage 1 checkpoint: '{args.stage1_checkpoint}'")
        checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
        stage1_state_dict = checkpoint['state_dict']

        # CRITICAL STEP: Add the 'stage1_model.' prefix to all keys to match the submodule name in MSG_FENet_Stage2
        new_state_dict = {}
        for k, v in stage1_state_dict.items():
            name = 'stage1_model.' + k
            new_state_dict[name] = v
            
        # Load with strict=False because our Stage 2 model has new layers (fusion, decoder) that are not in the checkpoint.
        loading_report = model.load_state_dict(new_state_dict, strict=False)
        
        print("   --- Weight Loading Report ---")
        if not loading_report.missing_keys:
            print("   ✅ SUCCESS: All keys from Stage 1 checkpoint were successfully matched and loaded.")
        else:
            # This should ideally not happen if the architectures match
            print(f"   ⚠️ WARNING: The following keys were expected but are missing in the model: {loading_report.missing_keys}")
        
        # This is expected and confirms that our new layers are correctly identified.
        print(f"   ℹ️ INFO: The following keys were present in the model but not in the checkpoint (This is correct for Stage 2):")
        for key in loading_report.unexpected_keys:
            print(f"      - {key}")
        print("   ---------------------------\n")


        # ### STAGE 2 MODIFICATION ###: Optimizer Configuration
        # We only want to train the new parameters (feature_fusion and decoder_object).
        # The MSG_FENet_Stage2 __init__ already set requires_grad=False for stage1_model.
        print("=> [Stage 2] Configuring optimizer to train ONLY new modules (feature_fusion, decoder_object)...")
        train_params = [p for p in model.parameters() if p.requires_grad]
        
        if not train_params:
            raise RuntimeError("FATAL: No trainable parameters found! Check model's requires_grad settings.")
            
        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # ### STAGE 2 MODIFICATION ###: Loss and Evaluator setup
        # For Stage 2, the primary loss is CrossEntropy on 'thing' classes.
        # We can reuse the existing SegmentationLosses class.
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda) # We don't need balanced weights or pos_weight here
        self.model, self.optimizer = model, optimizer
        
        # The Evaluator now works on the N 'thing' classes
        self.evaluator = Evaluator(self.args.num_classes_stage2) 
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        
        self.best_pred = 0.0
        self.best_epoch = 0
        # The 'resume' logic is for resuming a STAGE 2 training, not for loading Stage 1 weights.
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                 print(f"Warning: Stage 2 resume checkpoint '{args.resume}' not found. Starting from scratch.")
            else:
                # Standard resume logic for a Stage 2 run
                print(f"=> Resuming Stage 2 training from checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location='cpu')
                self.model.load_state_dict(checkpoint['state_dict']) # strict=True is fine here
                if not args.ft:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint.get('best_pred', 0.0)
                args.start_epoch = checkpoint.get('epoch', 0)
                print(f"=> Resumed successfully from epoch {args.start_epoch}")


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} (Train)')
        
        for i, sample in enumerate(tbar):
            image, target_things = sample['image'], sample['label'] # 'label' now corresponds to 'thing' GT
            if self.args.cuda:
                image, target_things = image.cuda(), target_things.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            
            # ### STAGE 2 MODIFICATION ###: Forward pass and Loss Calculation
            output = self.model(image)
            logit_things = output['things']
            
            # The primary loss for Stage 2 is the cross-entropy loss on the 'thing' classes
            loss = self.criterion.CrossEntropyLoss(logit_things, target_things)
            
            if loss > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            train_loss += loss.item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            tbar.set_description(f'Epoch {epoch+1} (Train) Things Loss: {train_loss / (i + 1):.3f} LR: {current_lr:.6f}')
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            
        self.writer.add_scalar('train/total_loss_epoch', train_loss / num_img_tr, epoch)
        print(f'[Epoch: {epoch+1}, numImages: {num_img_tr * self.args.batch_size}] Train Loss: {train_loss / num_img_tr:.4f}')

        if self.args.no_val:
            # Note: best_pred is not updated if no_val is true
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'best_epoch': self.best_epoch
            }, is_best=False)

    # ### STAGE 2 MODIFICATION ###: New visualization function for 'thing' segmentation
    def visualize_stage2_outputs(self, sample: Dict, predictions: Dict, epoch: int):
        img_tensor = sample['image'][0].cpu()
        target_seg_map = sample['label'][0].cpu().numpy()
        
        pred_seg_map = torch.argmax(predictions['things'][0], dim=0).cpu().numpy()

        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        input_image = (img_tensor.numpy().transpose(1, 2, 0) * std + mean) * 255.0
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)

        # Use the same decode_segmap for coloring, but now for 'thing' classes
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
        
        for i, sample in enumerate(tbar):
            image, target_things = sample['image'], sample['label']
            if self.args.cuda:
                image, target_things = image.cuda(), target_things.cuda()

            with torch.no_grad():
                # ### STAGE 2 MODIFICATION ###: Validation logic
                output = self.model(image)
                logit_things = output['things']
                
                # Calculate loss on validation set
                loss = self.criterion.CrossEntropyLoss(logit_things, target_things)
                test_loss += loss.item()
                tbar.set_description(f'Epoch {epoch+1} (Val) Things Loss: {test_loss / (i + 1):.3f}')
                
                # Get predictions for metrics calculation
                pred_things = torch.argmax(logit_things, dim=1)
                
                target_np = target_things.cpu().numpy()
                pred_np = pred_things.cpu().numpy()
                self.evaluator.add_batch(target_np, pred_np)
            
            # Visualize the first sample of the validation batch
            if self.args.save_val_results and i == 0:
                self.visualize_stage2_outputs(sample, {'things': logit_things}, epoch)
        
        # Calculate metrics
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/loss_epoch', test_loss / len(self.val_loader), epoch)
        
        print(f'\nValidation (Things Segmentation Eval): [Epoch: {epoch+1}]')
        print(f"Acc:{Acc:.4f}, Acc_class:{Acc_class:.4f}, mIoU:{mIoU:.4f}, fwIoU: {FWIoU:.4f}")
        print(f'Val Loss: {test_loss / len(tbar):.3f}\n')

        new_pred = mIoU
        is_best = new_pred > self.best_pred
        if is_best:
            self.best_pred = new_pred
            self.best_epoch = epoch + 1
        
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
        }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch MSG-FENet Stage 2 Trainer")
    # --- General arguments (mostly unchanged) ---
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'coco', 'cityscapes'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-size', type=int, default=513)
    parser.add_argument('--crop-size', type=int, default=513)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--epochs', type=int, default=50) # Default epochs for Stage 2
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.01) # Default LR for Stage 2
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--checkname', type=str, default='msg-fenet-stage2')
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-val', action='store_true', default=False)
    parser.add_argument('--save_val_results', action='store_true', default=False, help="Save val results for debugging")

    # ### STAGE 2 MODIFICATION ###: New and modified arguments
    parser.add_argument('--training-stage', type=int, default=2, choices=[2], help="Set to 2 for Stage 2 training.")
    parser.add_argument('--stage1-checkpoint', type=str, required=True, help="Path to the pretrained Stage 1 checkpoint (.pth.tar file).")
    parser.add_argument('--num-classes-stage1', type=int, default=2, help="Number of output classes for Stage 1 model (e.g., stuff and objectness).")
    parser.add_argument('--num-classes-stage2', type=int, default=8, help="Number of 'thing' classes for Stage 2 segmentation (e.g., 8 for Cityscapes).")
    parser.add_argument('--resume', type=str, default=None, help="Path to a Stage 2 checkpoint to resume training.")
    parser.add_argument('--ft', action='store_true', default=False, help="Fine-tuning: reset optimizer and epoch count.")
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    if args.batch_size is None: args.batch_size = 4 * len(args.gpu_ids)
    
    print("\n--- MSG-FENet Stage 2 Training Arguments ---")
    print(json.dumps(vars(args), indent=2))
    print("------------------------------------------\n")

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    print(f"--- Starting Training from Epoch {trainer.args.start_epoch + 1} to {trainer.args.epochs} ---")
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and (epoch + 1) % args.eval_interval == 0:
            trainer.validation(epoch)
    
    trainer.writer.close()
    print("\n--- Training finished ---")
    print(f"Best mIoU: {trainer.best_pred:.4f} achieved at epoch {trainer.best_epoch}")

if __name__ == "__main__":
    main()