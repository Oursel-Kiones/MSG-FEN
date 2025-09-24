import os
from tqdm import tqdm
import numpy as np
from mypath import Path

def calculate_weigths_labels(dataset, dataloader, num_classes):
    z = np.zeros((num_classes,))
    print("Calculating classes weights...")
    tqdm_batch = tqdm(dataloader, total=len(dataloader))
    
    for sample_batch in tqdm_batch:
        # <<< 核心修改: sample_batch['label'] 现在是一个列表 >>>
        labels_list = sample_batch['label']
        for label_tensor in labels_list:
            target = label_tensor.cpu().numpy().astype(np.uint8)
            mask = (target >= 0) & (target < num_classes)
            hist = np.bincount(target[mask], minlength=num_classes)
            z += hist
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret