# /workspace/deep参考1/dataloaders/__init__.py

from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
# <<< 核心修改 1: 导入我们自定义的 collate_fn >>>
from dataloaders.utils import custom_collate_fn

def make_data_loader(args, **kwargs):
    """
    Creates train, validation, and test data loaders.
    """

    if args.dataset == 'cityscapes':
        # 保持使用 RandomScaleCrop
        train_transform = tr.ExtCompose([
            tr.RandomScaleCrop(base_size=args.base_size, crop_size=args.crop_size),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        val_transform = tr.ExtCompose([
            tr.FixScaleCrop(crop_size=args.crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        train_set = cityscapes.CityscapesSegmentation(args, split='train', transform=train_transform)
        val_set = cityscapes.CityscapesSegmentation(args, split='val', transform=val_transform)
        test_set = cityscapes.CityscapesSegmentation(args, split='test', transform=val_transform)

        num_class = train_set.NUM_CLASSES

        # <<< 核心修改 2: 在创建 DataLoader 时传入 collate_fn >>>
        # 注意：只对 train_loader 使用，因为只有它用了 RandomScaleCrop
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=custom_collate_fn, **kwargs)
        # val_loader 和 test_loader 的尺寸是固定的，可以继续使用默认的 collate_fn
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'pascal':
        train_transform = tr.ExtCompose([
            tr.RandomScaleCrop(base_size=args.base_size, crop_size=args.crop_size, scale_range=(0.5, 2.0)),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        val_transform = tr.ExtCompose([
            tr.FixScaleCrop(crop_size=args.crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        train_set = pascal.VOCSegmentation(args, split='train', transform=train_transform)
        val_set = pascal.VOCSegmentation(args, split='val', transform=val_transform)
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'], transform=train_transform)
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        # <<< 核心修改 2: 在创建 DataLoader 时传入 collate_fn >>>
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=custom_collate_fn, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    # (对 coco 也进行类似修改)
    elif args.dataset == 'coco':
        # ...
        # <<< 核心修改 2: 在创建 DataLoader 时传入 collate_fn >>>
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=custom_collate_fn, **kwargs)
        # ...
    
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not supported.")