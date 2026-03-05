# 文件路径: /workspace/deep参考1/train3.py
print("--- SCRIPT VERSION CHECK: THIS IS THE LATEST MODIFIED FILE (STAGE 2 - ENTERPRISE E2E + FOCAL LOSS) ---")

import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from modeling.msg_fenet import MSG_FENet_Stage2 
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses # 引入 Loss 库

class Trainer_Stage2(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # ==========================================
        # 1. 架构组装与权重加载
        # ==========================================
        print("INFO: Initializing MSG-FENet Stage 2 Architecture...")
        stage1_base = DeepLab(num_classes=19, backbone=args.backbone, output_stride=args.out_stride, 
                              sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
        
        self.model = MSG_FENet_Stage2(stage1_model=stage1_base, num_object_classes=12)

        # 【核心步骤 A】：加载 Stage 1 黄金先验权重
        if not args.stage1_checkpoint or not os.path.isfile(args.stage1_checkpoint):
            raise FileNotFoundError(f"致命错误: 必须提供有效的 Stage 1 权重路径！找不到 {args.stage1_checkpoint}")
        
        print(f"=> Loading Stage 1 Base weights from: {args.stage1_checkpoint}")
        checkpoint_s1 = torch.load(args.stage1_checkpoint, map_location='cpu')
        self.model.stage1_model.load_state_dict(checkpoint_s1['state_dict'], strict=True)
        print("=> [成功] Stage 1 引擎权重严苛加载完毕！")

        # ==========================================
        # 2. 配置双速优化器 (端到端微调的核心)
        # ==========================================
        print("INFO: Configuring optimizer for Stage 2 (End-to-End Fine-Tuning).")
        train_params =[
            # 引擎部分用极小的学习率 (lr * 0.1) 进行微调，保护先验知识
            {'params': self.model.stage1_model.backbone.parameters(), 'lr': args.lr * 0.1},
            {'params': self.model.stage1_model.aspp.parameters(), 'lr': args.lr * 0.1},
            # 新模块用正常学习率 (lr)
            {'params': self.model.feature_fusion.parameters(), 'lr': args.lr},
            {'params': self.model.decoder_object.parameters(), 'lr': args.lr},
            {'params': self.model.low_level_compressor.parameters(), 'lr': args.lr}
        ]
        self.optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                         weight_decay=args.weight_decay, nesterov=args.nesterov)

        # ==========================================
        # 3. 损失函数与评估器设定 (Focal Loss 大杀器)
        # ==========================================
        print("INFO: Activating Focal Loss for hard examples mining!")
        loss_engine = SegmentationLosses(weight=None, ignore_index=255, cuda=args.cuda)
        self.criterion = loss_engine.FocalLoss
        
        self.evaluator = Evaluator(num_class=12) # 仅评估 12 类
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        self.best_pred = 0.0
        self.best_epoch = 0

        # 【核心步骤 B】：Stage 2 断点续训 (解决服务器崩溃问题)
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                print(f"Warning: Checkpoint path '{args.resume}' not found. Starting Stage 2 from scratch.")
            else:
                print(f"=> loading Stage 2 checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint.get('epoch', 0)
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint.get('best_pred', 0.0)
                self.best_epoch = checkpoint.get('best_epoch', 0)
                print(f"=> loaded checkpoint (resuming from epoch {args.start_epoch}, best_pred {self.best_pred:.4f})")

    def _pad_and_stack(self, tensor_list, pad_value=0):
        if not isinstance(tensor_list, list): return tensor_list
        max_h = max([t.shape[-2] for t in tensor_list])
        max_w = max([t.shape[-1] for t in tensor_list])
        padded_list =[]
        for t in tensor_list:
            pad_h = max_h - t.shape[-2]
            pad_w = max_w - t.shape[-1]
            if pad_h > 0 or pad_w > 0: padded = F.pad(t, (0, pad_w, 0, pad_h), value=pad_value)
            else: padded = t
            padded_list.append(padded)
        return torch.stack(padded_list, dim=0)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        
        # 冻结 BN 层统计量，防止微调时 Batch Size 较小导致统计量剧烈波动
        def set_bn_eval(module):
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                module.eval()
        self.model.apply(set_bn_eval)
        
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} (Train Stage 2 E2E)')

        for i, sample_batch in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            
            image_batch = self._pad_and_stack(sample_batch['image'], pad_value=0.0)
            target_object = self._pad_and_stack(sample_batch['object_gt'], pad_value=255).long()
            
            if self.args.cuda:
                image_batch, target_object = image_batch.cuda(), target_object.cuda()
            
            predictions = self.model(image_batch)
            pred_object = predictions['object'] 

            loss = self.criterion(pred_object, target_object)
            
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description(f'Epoch {epoch+1} (Train) Loss: {train_loss / (i + 1):.4f}')
            self.writer.add_scalar('train/stage2_object_loss_iter', loss.item(), i + num_img_tr * epoch)
            
        self.writer.add_scalar('train/stage2_object_loss_epoch', train_loss / num_img_tr, epoch)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} (Val Stage 2)')
        test_loss = 0.0
        
        for i, sample in enumerate(tbar):
            image_batch = self._pad_and_stack(sample['image'], pad_value=0.0)
            target_object = self._pad_and_stack(sample['object_gt'], pad_value=255).long()
            
            if self.args.cuda:
                image_batch, target_object = image_batch.cuda(), target_object.cuda()

            with torch.no_grad():
                predictions = self.model(image_batch)
                pred_object = predictions['object']
                
                loss = self.criterion(pred_object, target_object)
                test_loss += loss.item()
                
                pred_map = torch.argmax(pred_object, dim=1).cpu().numpy()
                target_np = target_object.cpu().numpy()
                self.evaluator.add_batch(target_np, pred_map)
                tbar.set_description(f'Epoch {epoch+1} (Val) Loss: {test_loss / (i + 1):.4f}')
            
            # 【工程化补齐】：保存验证集可视化图片
            if self.args.save_val_results and i == 0:
                save_dir = os.path.join(self.saver.experiment_dir, 'stage2_object_visuals')
                os.makedirs(save_dir, exist_ok=True)
                
                mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                img_tmp = np.clip((image_batch[0].cpu().numpy().transpose(1, 2, 0) * std + mean) * 255.0, 0, 255).astype(np.uint8)
                
                target_tmp = target_np[0]
                pred_tmp = pred_map[0]

                # 简单映射 12 类的颜色方便观察
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(img_tmp); axs[0].set_title('Input'); axs[0].axis('off')
                axs[1].imshow(target_tmp, cmap='tab20', vmin=0, vmax=11); axs[1].set_title('Object GT (12 classes)'); axs[1].axis('off')
                axs[2].imshow(pred_tmp, cmap='tab20', vmin=0, vmax=11); axs[2].set_title('Object Prediction'); axs[2].axis('off')
                plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}.png"), bbox_inches='tight')
                plt.close(fig)

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        self.writer.add_scalar('val/stage2_mIoU_object', mIoU, epoch)
        
        print(f'\nValidation Stage 2:[Epoch: {epoch+1}]')
        print(f"mIoU (Objects 12 classes): {mIoU:.4f}")
        print(f'Val Loss: {test_loss / len(tbar):.4f}\n')

        is_best = False
        if mIoU > self.best_pred:
            is_best = True
            self.best_pred = mIoU
            self.best_epoch = epoch + 1
            print(f"*** New global best mIoU {self.best_pred:.4f} (epoch {self.best_epoch}) ***")

        # 【工程化补齐】：标准的 checkpoint 保存逻辑
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
        }, is_best)

def main():
    parser = argparse.ArgumentParser(description="MSG-FENet Stage 2 Trainer (E2E Enterprise Version + Focal Loss)")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-size', type=int, default=513)
    parser.add_argument('--crop-size', type=int, default=513)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=0.005) 
    parser.add_argument('--lr-scheduler', type=str, default='poly')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--checkname', type=str, default='msg_fenet_stage2')
    parser.add_argument('--resume', type=str, default=None, help="Stage 2 断点续训权重")
    parser.add_argument('--stage1-checkpoint', type=str, required=True, help="必须提供 Stage 1 训练好的 checkpoint 路径")
    parser.add_argument('--save_val_results', action='store_true', default=False, help="是否保存验证集可视化结果")
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        args.gpu_ids =[int(s.strip()) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError("Argument --gpu-ids must be a comma-separated list of integers.")
    
    trainer = Trainer_Stage2(args)
    for epoch in range(trainer.args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.writer.close()

if __name__ == "__main__":
    main()