import os
import shutil
import torch
from collections import OrderedDict
import glob
from datetime import datetime

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.base_dir = "/workspace/deep参考1/result"
        self.directory = os.path.join(self.base_dir, args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            print(f"Creating new experiment directory: {self.experiment_dir}")
            os.makedirs(self.experiment_dir)
        else:
            print(f"Using existing experiment directory: {self.experiment_dir}")

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filepath = os.path.join(self.experiment_dir, filename)
        try:
            torch.save(state, filepath)
            # print(f"   Checkpoint saved to: {filepath}") # 可以减少打印信息
        except Exception as e:
            print(f"Error saving checkpoint {filepath}: {e}")
            return

        if is_best:
            best_epoch = state.get('epoch', 'N/A')
            best_pred = state.get('best_pred', 0.0)
            try:
                with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                    f.write(f"{best_pred:.4f}\n")
                    f.write(f"Epoch: {best_epoch}\n")
                # print(f"   Best prediction info saved (mIoU: {best_pred:.4f}, Epoch: {best_epoch}).")
            except Exception as e:
                print(f"Error saving best_pred.txt: {e}")

            try:
                previous_miou = [0.0]
                all_runs_dirs = glob.glob(os.path.join(self.directory, 'experiment_*'))
                for run_dir in all_runs_dirs:
                    path = os.path.join(run_dir, 'best_pred.txt')
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                miou_line = f.readline()
                                miou = float(miou_line.strip())
                                previous_miou.append(miou)
                        except (ValueError, IndexError):
                             continue
                max_miou = max(previous_miou) if previous_miou else 0.0
                global_best_path = os.path.join(self.directory, 'model_best.pth.tar')

                # 修改比较逻辑，确保等于也更新（如果需要最新的 best）或严格大于
                # 这里使用严格大于 >
                if best_pred > max_miou:
                    print(f"   New global best mIoU {best_pred:.4f} (epoch {best_epoch}) > previous max {max_miou:.4f}. Updating global best model.")
                    shutil.copyfile(filepath, global_best_path)
                    print(f"   Global best model updated at: {global_best_path}")
                # elif best_pred == max_miou: # 如果等于也想更新（比如保存最新的同分模型）
                #    print(f"   Current best mIoU {best_pred:.4f} equals previous max. Updating global best model to latest.")
                #    shutil.copyfile(filepath, global_best_path)
                #    print(f"   Global best model updated at: {global_best_path}")

            except Exception as e:
                 print(f"Error comparing/copying global best model: {e}")

    # ======================== 修改开始 ========================
    # 修改 save_experiment_config 以记录精简的最佳参数
    def save_experiment_config(self, current_lr, best_epoch=None, best_pred=None):
        # 仅当 best_epoch 和 best_pred 有效时才执行保存（确保只在找到新最佳时写入）
        if best_epoch is None or best_pred is None:
             # print("   Skipping config save: best_epoch or best_pred is None.") # 可以取消注释以调试
             return

        # 文件名保持不变，但内容会被覆盖
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')

        # 使用 'w' 覆盖模式
        try:
             log_file = open(logfile, 'w')
             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
             log_file.write(f"--- Best Configuration Found at {timestamp} ---\n")

             p = OrderedDict()
             # === 选择你认为最重要的超参数记录 ===
             p['dataset'] = self.args.dataset
             p['backbone'] = self.args.backbone
             p['out_stride'] = self.args.out_stride
             p['initial_lr'] = self.args.lr # 初始学习率
             p['lr_scheduler'] = self.args.lr_scheduler
             p['loss_type'] = self.args.loss_type
             p['total_epochs_trained'] = best_epoch # 记录达到最佳时的 epoch
             p['base_size'] = self.args.base_size
             p['crop_size'] = self.args.crop_size
             p['batch_size'] = self.args.batch_size
             p['weight_decay'] = getattr(self.args, 'weight_decay', 'N/A')
             p['momentum'] = getattr(self.args, 'momentum', 'N/A')
             # 添加 sync_bn, freeze_bn 等其他你关心的参数
             p['sync_bn'] = getattr(self.args, 'sync_bn', 'N/A')
             p['freeze_bn'] = getattr(self.args, 'freeze_bn', 'N/A')

             # === 记录最佳结果 ===
             p['best_mIoU'] = f"{best_pred:.4f}" # 格式化最佳 mIoU
             p['best_epoch'] = best_epoch

             # 写入 p 字典中的关键参数摘要
             for key, val in p.items():
                 log_file.write(key + ': ' + str(val) + '\n')

             log_file.close()
             print(f"   Best experiment parameters saved/updated to: {logfile}")
        except Exception as e:
             print(f"Error saving best experiment config: {e}")
    # ======================== 修改结束 ========================