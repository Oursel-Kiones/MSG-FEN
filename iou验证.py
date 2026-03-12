# 文件路径: /workspace/deep参考1/iou验证.py
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np  # 确保导入了 numpy

from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from modeling.msg_fenet import MSG_FENet_Stage2
from utils.metrics import Evaluator

def _pad_and_stack(tensor_list, pad_value=0):
    if not isinstance(tensor_list, list): return tensor_list
    max_h = max([t.shape[-2] for t in tensor_list])
    max_w = max([t.shape[-1] for t in tensor_list])
    padded_list = []
    for t in tensor_list:
        pad_h = max_h - t.shape[-2]
        pad_w = max_w - t.shape[-1]
        if pad_h > 0 or pad_w > 0: padded = F.pad(t, (0, pad_w, 0, pad_h), value=pad_value)
        else: padded = t
        padded_list.append(padded)
    return torch.stack(padded_list, dim=0)

def main():
    parser = argparse.ArgumentParser(description="MSG-FENet Stage 2 IoU Validation")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-size', type=int, default=513)
    parser.add_argument('--crop-size', type=int, default=513)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=2) 
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--stage1-checkpoint', type=str, required=True, help="Stage 1 weights")
    parser.add_argument('--resume', type=str, required=True, help="Stage 2 weights to evaluate")
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        args.gpu_ids = [int(s.strip()) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError("Argument --gpu-ids must be a comma-separated list of integers.")

    # 仅加载验证集
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    _, val_loader, _, _ = make_data_loader(args, **kwargs)

    print("INFO: Initializing MSG-FENet Stage 2 Architecture for Validation...")
    stage1_base = DeepLab(num_classes=19, backbone=args.backbone, output_stride=args.out_stride, 
                          sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model = MSG_FENet_Stage2(stage1_model=stage1_base, num_object_classes=12)

    # 1. 加载 Stage 1 引擎权重
    print(f"=> Loading Stage 1 weights from: {args.stage1_checkpoint}")
    if not os.path.isfile(args.stage1_checkpoint):
        raise RuntimeError(f"Error: Stage 1 weights not found at {args.stage1_checkpoint}")
    checkpoint_s1 = torch.load(args.stage1_checkpoint, map_location='cpu')
    model.stage1_model.load_state_dict(checkpoint_s1['state_dict'], strict=True)
    
    # 2. 加载 Stage 2 待评估权重
    print(f"=> Loading Stage 2 weights from: {args.resume}")
    if not os.path.isfile(args.resume):
        raise RuntimeError(f"Error: Stage 2 weights not found at '{args.resume}'")
    checkpoint_s2 = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint_s2['state_dict'], strict=False)
    print("=> All weights loaded successfully!")

    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        model = model.cuda()

    model.eval()
    evaluator = Evaluator(num_class=12)
    evaluator.reset()
    
    tbar = tqdm(val_loader, desc='Validating')
    for i, sample in enumerate(tbar):
        image_batch = _pad_and_stack(sample['image'], pad_value=0.0)
        target_object = _pad_and_stack(sample['object_gt'], pad_value=255).long()
        
        if args.cuda:
            image_batch, target_object = image_batch.cuda(), target_object.cuda()

        with torch.no_grad():
            predictions = model(image_batch)
            pred_object = predictions['object']
            
            pred_map = torch.argmax(pred_object, dim=1).cpu().numpy()
            target_np = target_object.cpu().numpy()
            evaluator.add_batch(target_np, pred_map)

    # 计算平均 mIoU
    mIoU = evaluator.Mean_Intersection_over_Union()
    
    # ==============================================================
    # 【核心修复】：直接从底层混淆矩阵手撕单类 IoU 公式，绕过 API 限制
    # ==============================================================
    conf_matrix = evaluator.confusion_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        # 单类 IoU 公式 = 对角线 / (行和 + 列和 - 对角线)
        class_ious = np.diag(conf_matrix) / (
            np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)
        )
    
    object_class_names = ['fence', 'pole', 'traffic_light', 'traffic_sign', 
                          'person', 'rider', 'car', 'truck', 'bus', 
                          'train', 'motorcycle', 'bicycle']

    print("\n" + "="*60)
    print("========== 单类 IoU 诊断报告 ==========")
    print("="*60)
    for i, name in enumerate(object_class_names):
        if i < len(class_ious):
            val = class_ious[i]
            if np.isnan(val):
                print(f"{name.ljust(15)}: NaN")
            else:
                print(f"{name.ljust(15)}: {val * 100:.2f}%")
    print("-" * 60)
    print(f"Overall mIoU (12 classes) : {mIoU * 100:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()