from dataloaders.utils import decode_seg_map_sequence
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid # 导入make_grid
import os
class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        from torchvision.utils import make_grid # 在函数内部重新导入make_grid
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False) # 移除 range 参数
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False) # 移除 range 参数
        writer.add_image('Groundtruth label', grid_image, global_step)