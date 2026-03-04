# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import json # Added for printing args

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # For softmax

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
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


# --- CRF Post-processing Function ---
def apply_crf(image, probabilities, n_classes, n_iters=5, sxy_gaussian=3, compat_gaussian=3, sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    """
    Applies Dense CRF post-processing to segmentation probability maps.

    Args:
        image (np.ndarray): Original input image (H, W, 3), RGB format, values 0-255. Must be C contiguous.
        probabilities (np.ndarray): Model output probability map (C, H, W), where C is the number of classes.
        n_classes (int): Number of segmentation classes.
        n_iters (int): Number of CRF inference iterations.
        sxy_gaussian (int): Standard deviation for the spatial Gaussian kernel (smoothness term).
        compat_gaussian (int): Label compatibility factor for the Gaussian kernel (smoothness term weight).
        sxy_bilateral (int): Standard deviation for the spatial component of the bilateral kernel (appearance term).
        srgb_bilateral (int): Standard deviation for the color component of the bilateral kernel (appearance term).
        compat_bilateral (int): Label compatibility factor for the bilateral kernel (appearance term weight).

    Returns:
        np.ndarray: CRF-optimized segmentation map (H, W) with class indices.
    """
    if not DENSECRF_AVAILABLE:
        print("Error: pydensecrf not available. Cannot apply CRF.")
        # Fallback: return argmax of probabilities
        return np.argmax(probabilities, axis=0)

    H, W = image.shape[:2]
    C = n_classes

    # Ensure image is C contiguous (required by pydensecrf)
    image_rgb = np.ascontiguousarray(image)

    # 1. Create CRF object
    d = dcrf.DenseCRF2D(W, H, C)

    # 2. Set Unary Potentials
    # Probabilities shape is (C, H, W). Need (C, H*W) or (H*W, C).
    # Unary energy expects (n_classes, n_pixels).
    # Use negative log probabilities. Add epsilon for stability.
    unary = -np.log(probabilities.reshape(C, -1) + 1e-10)
    d.setUnaryEnergy(unary.astype(np.float32))

    # 3. Add Pairwise Potentials

    # Gaussian Smoothness Kernel (spatial proximity)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Bilateral Appearance Kernel (spatial proximity + color similarity)
    # Requires the image in HxWx3 format
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral,
                           rgbim=image_rgb, compat=compat_bilateral,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 4. Perform Inference
    Q = d.inference(n_iters)

    # 5. Get MAP (Maximum A Posteriori) segmentation
    map_result = np.argmax(Q, axis=0).reshape((H, W))

    return map_result
# --- End CRF Function ---


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        # ================== Corrected Dataloader Section ==================
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        try:
            # Call make_data_loader ONCE and get all outputs
            data_loaders_output = make_data_loader(args, **kwargs)

            # Check how many items were returned and unpack accordingly
            if len(data_loaders_output) == 4:
                self.train_loader, self.val_loader, self.test_loader, self.nclass = data_loaders_output
                print(f"Found {self.nclass} classes. Train, validation, and test loaders created.")
                if self.test_loader is None:
                    print("Note: Test loader was returned as None by make_data_loader.")
            elif len(data_loaders_output) == 3:
                # Assume the order is train, val, nclass (test loader is implicitly None)
                # Verify this assumption based on make_data_loader's implementation
                self.train_loader, self.val_loader, self.nclass = data_loaders_output
                self.test_loader = None
                print(f"Found {self.nclass} classes. Train and validation loaders created. Test loader not provided.")
                # Safety check: Ensure the third element really looks like nclass
                if not isinstance(self.nclass, int):
                     print(f"Warning: make_data_loader returned 3 items, but the third item ({type(self.nclass)}) doesn't seem to be the class count (int). Check make_data_loader logic.")
                     # Handle error appropriately, maybe set nclass to 0 or raise error
                     # self.nclass = 0 # Example fallback
                     raise TypeError("Failed to determine number of classes from make_data_loader output.")
            else:
                 # Handle unexpected number of return values
                 raise ValueError(f"make_data_loader returned an unexpected number of items: {len(data_loaders_output)}. Expected 3 or 4.")

            # === Final Check: Ensure nclass is a valid integer ===
            if not isinstance(self.nclass, int) or self.nclass <= 0:
                # If loading failed, nclass might be 0 here, but we need a valid num_classes for the model
                # Raise error if it's still not a valid integer after attempting to load data
                raise TypeError(f"Error: self.nclass must be a positive integer after data loading, but got: {self.nclass} (type: {type(self.nclass)})")

        except Exception as e:
            print(f"Error creating data loaders or processing output: {e}")
            # Set safe defaults to potentially allow further execution or easier debugging
            self.train_loader, self.val_loader, self.test_loader, self.nclass = None, None, None, 0
            # It might be better to re-raise the exception if dataloaders are critical
            print("Critical error during data loading. Exiting.")
            raise e # Re-raise the exception to stop execution if data loading fails
        # ================== End Corrected Dataloader Section ==================

        # Define network
        # Now self.nclass should be a correct integer
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
                print(f"Loaded balanced weights from: {classes_weights_path}")
            else:
                # Ensure train_loader exists before calculating weights
                if self.train_loader:
                    print("Calculating balanced weights...")
                    weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
                    # Optionally save the calculated weights
                    # np.save(classes_weights_path, weight)
                    # print(f"Saved calculated weights to: {classes_weights_path}")
                else:
                    print("Warning: Train loader not available, cannot calculate balanced weights.")
                    weight = None
            if weight is not None:
                weight = torch.from_numpy(weight.astype(np.float32))
                print(f"Using balanced weights: {weight}")
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler (ensure train_loader exists)
        if self.train_loader:
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                          args.epochs, len(self.train_loader))
        else:
            self.scheduler = None
            print("Warning: Train loader not available, LR scheduler not initialized.")


        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model) # Ensure SyncBN works if used
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_epoch = 0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                 # Simplified warning
                 print(f"Warning: Checkpoint path '{args.resume}' not found. Starting from scratch.")
                 args.resume = None # Set to None so the else block below handles 'starting from scratch'

            if args.resume: # Check again, as it might have been set to None above
                print(f"=> loading checkpoint '{args.resume}'")
                try:
                    # Load onto CPU first to avoid GPU memory issues if model was saved on different devices
                    checkpoint = torch.load(args.resume, map_location='cpu')
                    # Override start_epoch from command line ONLY if not finetuning
                    if not args.ft:
                       # Prioritize checkpoint epoch if available, otherwise use command-line start_epoch
                       # Command-line start_epoch acts as a minimum start if checkpoint is older
                       ckpt_epoch = checkpoint.get('epoch', 0)
                       args.start_epoch = max(args.start_epoch, ckpt_epoch)
                       print(f"   Resuming training from epoch {args.start_epoch} (Checkpoint epoch: {ckpt_epoch}, Command line: {self.args.start_epoch})")

                    state_dict = checkpoint.get('state_dict', checkpoint)
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()

                    # Check if the current model is DataParallel (mult-GPU)
                    current_is_data_parallel = isinstance(self.model, torch.nn.DataParallel)
                    # Detect if the saved model state_dict has 'module.' prefix (was saved using DataParallel)
                    saved_starts_with_module = any(k.startswith('module.') for k in state_dict.keys())

                    for k, v in state_dict.items():
                        is_module_key = k.startswith('module.')
                        if current_is_data_parallel and not is_module_key:
                            # Add 'module.' prefix if current model is DataParallel but saved is not
                            name = 'module.' + k
                        elif not current_is_data_parallel and is_module_key:
                            # Remove 'module.' prefix if current model is not DataParallel but saved is
                            name = k[7:]
                        else:
                            # Otherwise, keep the key as is
                            name = k
                        new_state_dict[name] = v

                    # Load model state with flexibility
                    try:
                        self.model.load_state_dict(new_state_dict, strict=True)
                        print("=> loaded model state_dict successfully (strict).")
                    except RuntimeError as e:
                         print(f"Warning: Strict loading failed ({e}). Trying non-strict loading.")
                         try:
                             self.model.load_state_dict(new_state_dict, strict=False)
                             print("=> loaded model state_dict successfully (non-strict).")
                         except Exception as load_err:
                             print(f"Error: Non-strict loading also failed: {load_err}")
                             print("Model weights could not be loaded. Check model architecture and checkpoint compatibility.")
                             # Decide whether to raise error or continue with initialized weights
                             # raise load_err


                    # Load optimizer state if not fine-tuning and present in checkpoint
                    if not args.ft and 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                        try:
                            self.optimizer.load_state_dict(checkpoint['optimizer'])
                            print("=> loaded optimizer state_dict successfully.")
                        except Exception as e:
                            print(f"Warning: Could not load optimizer state: {e}. Optimizer will start from scratch.")
                    elif not args.ft:
                         print("Warning: Optimizer state not found in checkpoint or is None. Optimizer will start from scratch.")
                    else: # If fine-tuning, explicitly state optimizer starts fresh
                         print("=> Fine-tuning mode: Optimizer state not loaded, starting optimizer from scratch.")


                    # Load best prediction score and its epoch
                    self.best_pred = checkpoint.get('best_pred', 0.0) # Get best score achieved so far
                    self.best_epoch = checkpoint.get('best_epoch', 0) # Get epoch where best score was achieved

                    print("=> loaded checkpoint '{}' (Target start epoch {}, best_pred: {:.4f} at epoch {})".format(
                          args.resume, args.start_epoch, self.best_pred, self.best_epoch))

                except Exception as e:
                    print(f"Error loading checkpoint '{args.resume}': {e}")
                    print("Starting from scratch.")
                    args.resume = None # Prevent further attempts if loading fails
                    args.start_epoch = 0
                    self.best_pred = 0.0 # Reset best score
                    self.best_epoch = 0

        else:
             print("=> No checkpoint specified or found, starting from scratch.")
             args.start_epoch = 0 # Ensure start_epoch is 0 if not resuming
             self.best_pred = 0.0
             self.best_epoch = 0

        # Clear start epoch if fine-tuning, regardless of resume status
        if args.ft:
            print("=> Fine-tuning mode, start_epoch reset to 0.")
            args.start_epoch = 0
            self.best_pred = 0.0 # Reset best score for fine-tuning
            self.best_epoch = 0


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        if not self.train_loader:
            print(f"Error: Train loader is not initialized for epoch {epoch+1}. Skipping training.")
            return
        num_img_tr = len(self.train_loader)
        if num_img_tr == 0:
            print(f"Error: Train loader has length 0 for epoch {epoch+1}. Skipping training.")
            return

        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} (Train)')

        # Save config at the beginning of the first *actual* training epoch
        if epoch == self.args.start_epoch:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"--- Saving initial experiment config at epoch {epoch+1} ---")
            self.saver.save_experiment_config(current_lr, self.best_epoch, self.best_pred)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            # Update LR before optimizer step
            if self.scheduler:
                self.scheduler(self.optimizer, i, epoch, self.best_pred) # Pass best_pred for potential ReduceLROnPlateau

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            tbar.set_description(f'Epoch {epoch+1}/{self.args.epochs} (Train) Loss: {train_loss / (i + 1):.3f} LR: {current_lr:.6f}')
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Visualize training images less frequently (optional)
            # Add your visualization code here if needed

        avg_train_loss = train_loss / num_img_tr
        self.writer.add_scalar('train/total_loss_epoch', avg_train_loss, epoch)
        # Estimate total images processed in this epoch
        total_images_epoch = num_img_tr * self.args.batch_size
        print(f'[Epoch: {epoch+1}, Est. numImages: {total_images_epoch}]')
        print(f'Train Loss: {avg_train_loss:.3f}')

        # Save checkpoint if skipping validation
        if self.args.no_val:
            is_best = False # Cannot determine best without validation
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"--- Saving checkpoint (no_val mode) after epoch {epoch+1} ---")
            # Use self.saver.save_checkpoint, passing is_best=False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred, # Store the current best_pred (might be 0.0)
                'best_epoch': self.best_epoch, # Store the current best_epoch
            }, is_best, filename=f'checkpoint_epoch{epoch+1}.pth.tar')


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()

        if not self.val_loader:
            print(f"Error: Validation loader is not initialized for epoch {epoch+1}. Skipping validation.")
            return
        num_img_val = len(self.val_loader)
        if num_img_val == 0:
            print(f"Error: Validation loader has length 0 for epoch {epoch+1}. Skipping validation.")
            return

        tbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} (Val)  ')
        test_loss = 0.0
        crf_params = None
        processing_type = "Raw" # Default processing type

        # --- Setup CRF Parameters if enabled ---
        if self.args.use_crf:
            if DENSECRF_AVAILABLE:
                 crf_params = {
                     'n_iters': self.args.crf_iters,
                     'sxy_gaussian': self.args.crf_sxy_g,
                     'compat_gaussian': self.args.crf_compat_g,
                     'sxy_bilateral': self.args.crf_sxy_b,
                     'srgb_bilateral': self.args.crf_srgb_b,
                     'compat_bilateral': self.args.crf_compat_b,
                     'n_classes': self.nclass # Use the class number determined at init
                 }
                 processing_type = "CRF" # Update processing type
                 # Print CRF params only once at the start of validation loop
                 if tbar.n == 0: # Check if it's the first iteration
                     print("\n--- Applying CRF Post-processing with params: ---")
                     print(crf_params)
                     print("-------------------------------------------------")
            else:
                # Print warning only once if CRF requested but unavailable
                if tbar.n == 0:
                     print("\n--- CRF requested (--use_crf) but pydensecrf not found. Using raw predictions. ---")
                # processing_type remains "Raw"

        for i, sample in enumerate(tbar):
            image_batch, target_batch = sample['image'], sample['label']
            # --- Keep original tensor for unnormalization in CRF/visualization ---
            image_batch_original_tensor = image_batch.clone() # Clone before moving to GPU

            if self.args.cuda:
                image_batch, target_batch = image_batch.cuda(), target_batch.cuda()

            with torch.no_grad():
                output_batch = self.model(image_batch)
                # Calculate loss based on model output BEFORE CRF
                loss = self.criterion(output_batch, target_batch)

                # --- Get Probabilities (always needed for potential CRF) ---
                # Use torch.softmax, ensure output is on CPU as numpy array
                probabilities_batch = F.softmax(output_batch, dim=1).cpu().numpy() # (B, C, H, W)

            test_loss += loss.item()
            tbar.set_description(f'Epoch {epoch+1}/{self.args.epochs} (Val) Loss: {test_loss / (i + 1):.3f}')

            # --- Get Raw Prediction (Argmax) for comparison & default ---
            pred_batch_raw = torch.argmax(output_batch, dim=1).cpu().numpy() # (B, H, W)
            target_batch_np = target_batch.cpu().numpy() # (B, H, W)

            # --- Apply Post-processing (CRF if enabled and available) ---
            if processing_type == "CRF" and crf_params is not None:
                # Apply CRF per image in the batch
                refined_pred_batch = np.zeros_like(target_batch_np)
                for b_idx in range(image_batch_original_tensor.size(0)):
                    # 1. Get single probability map
                    probabilities_single = probabilities_batch[b_idx] # (C, H, W)

                    # 2. Get single original image and unnormalize
                    img_tensor = image_batch_original_tensor[b_idx] # Use the original CPU tensor
                    # Assuming ImageNet mean/std - **ADJUST IF YOUR DATALOADER USES DIFFERENT VALUES**
                    # TODO: Consider getting mean/std from dataloader/args if they vary
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_unnormalized = img_tensor.numpy().transpose(1, 2, 0) # H, W, C
                    img_unnormalized = (img_unnormalized * std + mean) * 255.0
                    img_unnormalized = np.clip(img_unnormalized, 0, 255).astype(np.uint8)
                    # Ensure it's RGB if needed by CRF (usually is)
                    image_for_crf = np.ascontiguousarray(img_unnormalized) # Make C contiguous

                    # 3. Apply CRF
                    refined_pred_map = apply_crf(image_for_crf, probabilities_single, **crf_params)
                    refined_pred_batch[b_idx] = refined_pred_map

                # Use CRF results for evaluation
                pred_to_eval = refined_pred_batch

            else:
                # Use raw predictions if CRF is not enabled or not available
                pred_to_eval = pred_batch_raw
            # ------------------------------------------------

            
                        # --- Add results to evaluator ---
            # Ensure both target and prediction are integer types before evaluation
            try:
                # Convert ground truth to int64 (IMPORTANT!)
                target_batch_np_int = target_batch_np.astype(np.int64)

                # Ensure prediction is also int64 (argmax/CRF should return int, but let's be sure)
                pred_to_eval_int = pred_to_eval.astype(np.int64)

                # Print types and unique values AFTER casting for debugging (Optional)
                # print(f"DEBUG: Target Int - dtype={target_batch_np_int.dtype}, unique={np.unique(target_batch_np_int)}")
                # print(f"DEBUG: Pred Int   - dtype={pred_to_eval_int.dtype}, unique={np.unique(pred_to_eval_int)}")

                # Pass integer arrays to the evaluator
                self.evaluator.add_batch(target_batch_np_int, pred_to_eval_int)

            except Exception as eval_err:
                print(f"Error during evaluator.add_batch: {eval_err}")
                # Print details of the integer arrays passed
                if 'target_batch_np_int' in locals():
                     print(f"  target_batch_np_int info: dtype={target_batch_np_int.dtype}, shape={target_batch_np_int.shape}, unique={np.unique(target_batch_np_int)}")
                if 'pred_to_eval_int' in locals():
                     print(f"  pred_to_eval_int info: dtype={pred_to_eval_int.dtype}, shape={pred_to_eval_int.shape}, unique={np.unique(pred_to_eval_int)}")
                # Re-raise the error or handle it as appropriate
                raise eval_err

            # --- Validation Visualization (Compare Raw vs. Processed) ---
            # Visualize only the first batch of each validation run if requested
            if self.args.save_val_results and i == 0:
                save_dir = os.path.join(self.saver.experiment_dir, f'validation_visuals_epoch_{epoch+1}')
                os.makedirs(save_dir, exist_ok=True)

                img_idx_to_vis = 0 # Visualize the first image in the batch
                print(f"\n--- Saving validation visualization ({processing_type}) for epoch {epoch+1}, batch {i} ---")

                # Get unnormalized image for visualization
                img_tensor = image_batch_original_tensor[img_idx_to_vis]
                mean = np.array([0.485, 0.456, 0.406]) # TODO: Get from args/dataloader?
                std = np.array([0.229, 0.224, 0.225])  # TODO: Get from args/dataloader?
                img_tmp = img_tensor.numpy().transpose(1, 2, 0)
                img_tmp = (img_tmp * std + mean) * 255.0
                img_tmp = np.clip(img_tmp, 0, 255).astype(np.uint8)

                target_tmp = target_batch_np[img_idx_to_vis]
                baseline_pred_tmp = pred_batch_raw[img_idx_to_vis]    # Always show raw prediction
                processed_pred_tmp = pred_to_eval[img_idx_to_vis] # Show processed (CRF or Raw)

                if DECODE_SEGMAP_AVAILABLE:
                    # Attempt to decode, fallback to raw indices if dataset name unknown
                    try:
                        target_segmap = decode_segmap(target_tmp, dataset=self.args.dataset)
                    except ValueError:
                        print(f"Warning: decode_segmap does not support dataset '{self.args.dataset}'. Showing class indices.")
                        target_segmap = target_tmp
                    try:
                        baseline_segmap = decode_segmap(baseline_pred_tmp, dataset=self.args.dataset)
                    except ValueError:
                         baseline_segmap = baseline_pred_tmp
                    try:
                        processed_segmap = decode_segmap(processed_pred_tmp, dataset=self.args.dataset)
                    except ValueError:
                        processed_segmap = processed_pred_tmp
                else:
                    # If decode_segmap itself is unavailable
                    target_segmap = target_tmp
                    baseline_segmap = baseline_pred_tmp
                    processed_segmap = processed_pred_tmp

                fig, axs = plt.subplots(1, 4, figsize=(20, 5)) # Adjusted figsize
                axs[0].imshow(img_tmp)
                axs[0].set_title('Input Image')
                axs[0].axis('off')
                axs[1].imshow(target_segmap)
                axs[1].set_title('Ground Truth')
                axs[1].axis('off')
                axs[2].imshow(baseline_segmap)
                axs[2].set_title('Raw Prediction') # Simpler title
                axs[2].axis('off')
                axs[3].imshow(processed_segmap)
                # Dynamic title based on processing type
                title = f'{processing_type} Prediction'
                if processing_type == "CRF" and crf_params:
                    title += f' (iters={crf_params["n_iters"]})'
                axs[3].set_title(title)
                axs[3].axis('off')

                plt.tight_layout()
                # Include processing type in filename for clarity
                save_filename = f"epoch_{epoch+1}_batch_{i}_idx_{img_idx_to_vis}_{processing_type.lower()}_vs_raw.png"
                save_path = os.path.join(save_dir, save_filename)
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved visualization to: {save_path}")
            # --- End Visualization ---

        # --- Calculate and Log Metrics ---
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        avg_test_loss = test_loss / num_img_val
        # Add metrics to TensorBoard, tag with processing type
        tag_suffix = f"_{processing_type}"
        self.writer.add_scalar(f'val/total_loss_epoch', avg_test_loss, epoch)
        self.writer.add_scalar(f'val/mIoU{tag_suffix}', mIoU, epoch)
        self.writer.add_scalar(f'val/Acc{tag_suffix}', Acc, epoch)
        self.writer.add_scalar(f'val/Acc_class{tag_suffix}', Acc_class, epoch)
        self.writer.add_scalar(f'val/fwIoU{tag_suffix}', FWIoU, epoch)

        print(f'Validation ({processing_type} Evaluation):') # Indicate which result is printed
        # Estimate total validation images processed
        total_images_val = num_img_val * self.args.test_batch_size
        print(f'[Epoch: {epoch+1}, Est. numImages: {total_images_val}]')
        print("Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwIoU: {:.4f}".format(Acc, Acc_class, mIoU, FWIoU))
        print(f'Val Loss: {avg_test_loss:.3f}') # Note: Loss is calculated *before* post-processing

        # --- Save Model Checkpoint ---
        new_pred = mIoU # The metric used to determine "best" model (always mIoU from processed results)
        is_best = False
        # Compare against the absolute best prediction seen so far across all epochs/processing types
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.best_epoch = epoch + 1 # Current epoch is epoch + 1
            print(f"*** New best mIoU ({processing_type}) found at epoch {self.best_epoch}: {self.best_pred:.4f} ***")

        # Save checkpoint (contains the raw model weights)
        # Note: The "best" designation is based on the potentially post-processed mIoU,
        # but the saved model is always the one *before* post-processing.
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred, # Store the best *evaluated* mIoU achieved so far
            'best_epoch': self.best_epoch, # Store the epoch where best_pred was achieved
            'nclass': self.nclass # Save number of classes for potential future use
        }
        # Optionally save best CRF params if this is the best model AND CRF was used
        if is_best and processing_type == "CRF" and crf_params:
             checkpoint_data['best_crf_params'] = crf_params
             print("   Saved best CRF parameters in checkpoint.")

        self.saver.save_checkpoint(checkpoint_data, is_best)

        # Optionally save config when best model found
        if is_best:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"--- Saving experiment config for new best model at epoch {epoch+1} ---")
            self.saver.save_experiment_config(current_lr, self.best_epoch, self.best_pred)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training with Optional CRF Post-processing")

    # === Standard Arguments ===
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'], help='backbone name')
    parser.add_argument('--out-stride', type=int, default=16, help='network output stride')
    parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'cityscapes'], help='dataset name (pascal, coco, cityscapes)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513, help='base image size')
    parser.add_argument('--crop-size', type=int, default=513, help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters')
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (ce, focal)')
    parser.add_argument('--epochs', type=int, default=None, metavar='N', help='number of epochs to train (default: auto based on dataset)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (override with checkpoint unless fine-tuning)')
    parser.add_argument('--batch-size', type=int, default=None, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None, metavar='N', help='input batch size for validation/testing (default: same as training batch size)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False, help='whether to use balanced weights based on class frequency')
    # Learning Rate Arguments
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (default: auto based on dataset)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help='lr scheduler mode: (poly, step, cos)')
    # Optimizer Arguments
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov momentum')
    # CUDA, Seed and Logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default: 0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # Checkpoint settings
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint (default: None)')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name prefix (default: deeplab-<backbone>)')
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset (resets start_epoch and optimizer state)')
    # Evaluation options
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (run validation every N epochs)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--save_val_results', action='store_true', default=False, help='Save visual segmentation results during validation')

    # === CRF Post-processing Arguments ===
    parser.add_argument('--use_crf', action='store_true', default=False,
                        help='Apply Dense CRF post-processing during validation.')
    parser.add_argument('--crf_iters', type=int, default=5,
                        help='Number of iterations for CRF inference.')
    parser.add_argument('--crf_sxy_g', type=int, default=3,
                        help='CRF spatial Gaussian kernel std dev (smoothness).')
    parser.add_argument('--crf_compat_g', type=int, default=3,
                        help='CRF spatial Gaussian kernel compatibility (smoothness weight).')
    parser.add_argument('--crf_sxy_b', type=int, default=80,
                        help='CRF spatial Bilateral kernel std dev (appearance).')
    parser.add_argument('--crf_srgb_b', type=int, default=13,
                        help='CRF color Bilateral kernel std dev (appearance).')
    parser.add_argument('--crf_compat_b', type=int, default=10,
                        help='CRF Bilateral kernel compatibility (appearance weight).')
    # ========================================

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            gpu_list = [int(s.strip()) for s in args.gpu_ids.split(',')]
            if not gpu_list: # Handle empty string case
                 raise ValueError
            args.gpu_ids = gpu_list
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a non-empty comma-separated list of integers only (e.g., 0 or 0,1)')

    # --- Set Auto Defaults ---
    # Default number of epochs based on dataset
    if args.epochs is None:
        epoches = {'coco': 30, 'cityscapes': 200, 'pascal': 50}
        args.epochs = epoches.get(args.dataset.lower(), 50) # Default 50 if dataset not in dict

    # Default learning rate based on dataset
    if args.lr is None:
        lrs = {'coco': 0.1, 'cityscapes': 0.01, 'pascal': 0.007}
        # Adjust LR based on batch size? (Common practice)
        # Example: base_lr = lrs.get(args.dataset.lower(), 0.01)
        # args.lr = base_lr * (args.batch_size / 16) # If base batch size was 16
        args.lr = lrs.get(args.dataset.lower(), 0.01) # Keep simple for now

    # Default batch size (adjust based on GPUs)
    if args.batch_size is None:
        # Heuristic: 4 per GPU seems reasonable for 513x513 on ~11GB cards
        args.batch_size = 4 * len(args.gpu_ids) if args.cuda and args.gpu_ids else 4
    # Default validation/test batch size
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    # Default SyncBN behavior
    if args.sync_bn is None:
        # Use SyncBN if using CUDA and more than one GPU
        args.sync_bn = args.cuda and len(args.gpu_ids) > 1

    # Default checkpoint name prefix
    if args.checkname is None:
        args.checkname = f'deeplab-{args.backbone}'

    # --- Print effective arguments ---
    print("--- Effective Training Arguments ---")
    # Sort keys for consistent printing
    args_dict = vars(args)
    sorted_args = {k: args_dict[k] for k in sorted(args_dict)}
    print(json.dumps(sorted_args, indent=2))
    print("------------------------------------")

    # --- Seed setting for reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed) # Seed numpy as well
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed) # Seed all GPUs
        # Optional: uncomment for potentially more deterministic results at the cost of performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False # Benchmark True is usually faster but less deterministic


    # --- Create Trainer and Start ---
    try:
        trainer = Trainer(args)
    except Exception as e:
        print(f"Failed to initialize Trainer: {e}")
        # Potentially log the full traceback here
        import traceback
        traceback.print_exc()
        return # Exit if trainer fails to initialize

    # --- Initial Validation ---
    # Run validation before starting training only if resuming from an epoch > 0, not fine-tuning, and not skipping validation
    should_run_initial_val = (trainer.args.resume is not None and
                              not trainer.args.ft and
                              not trainer.args.no_val and
                              trainer.args.start_epoch > 0)

    if should_run_initial_val:
        initial_val_epoch = trainer.args.start_epoch - 1 # Validate the state *before* the start_epoch
        print(f"\n--- Running initial validation for epoch {initial_val_epoch} from resumed checkpoint ---")
        try:
            trainer.validation(initial_val_epoch)
        except Exception as e:
            print(f"Error during initial validation: {e}")
            # Optionally decide whether to continue training or stop
            # return

    # --- Training Loop ---
    print(f"\n--- Starting Training from Epoch: {trainer.args.start_epoch + 1} ---") # Display 1-based epoch
    print(f"--- Total Epochs: {trainer.args.epochs} ---")

    # Loop from start_epoch up to (but not including) args.epochs
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{trainer.args.epochs} =====")
        # Perform training for the current epoch
        if trainer.train_loader: # Check if trainer initialized properly
            try:
                trainer.training(epoch)
            except Exception as e:
                 print(f"Error during training epoch {epoch+1}: {e}")
                 import traceback
                 traceback.print_exc()
                 # Decide whether to break or continue
                 break # Stop training on error
        else:
            print(f"Skipping training for epoch {epoch+1} due to missing train_loader.")

        # Perform validation if not skipped and interval is met
        if not trainer.args.no_val and (epoch + 1) % args.eval_interval == 0:
            if trainer.val_loader: # Check if validation loader exists
                try:
                    trainer.validation(epoch)
                except Exception as e:
                    print(f"Error during validation epoch {epoch+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Decide whether to break or continue
            else:
                print(f"Epoch {epoch+1}: Skipping validation as validation loader is not available.")
        elif not trainer.args.no_val:
             print(f"Epoch {epoch+1}: Skipping validation (eval_interval not met).")


    # --- End of Training ---
    print("\n--- Training finished ---")
    print(f"Best mIoU: {trainer.best_pred:.4f} achieved at epoch {trainer.best_epoch}")
    if hasattr(trainer, 'saver') and trainer.saver:
        print(f"Checkpoints and logs saved in: {trainer.saver.experiment_dir}")

    # Close tensorboard writer
    if trainer.writer is not None:
        try:
            trainer.writer.close()
        except Exception as e:
            print(f"Warning: Error closing Tensorboard writer: {e}")


if __name__ == "__main__":
   main()