import argparse
import logging
import os
import sys
from timeit import default_timer as timer

import numpy as np
import torch
from datasets import load_dataset
from monai import config
from monai.data import DataLoader, Dataset
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler, TensorBoardStatsHandler, CheckpointSaver,
    ValidationHandler, LrScheduleHandler
)
from monai.networks.nets import resnet10
from monai.transforms import (
    Compose, EnsureType, EnsureTyped, ScaleIntensityd,
    RandRotate90d, RandFlipd, Activationsd, AsDiscrete,
    Resized
)
from ignite.metrics import Accuracy
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# 1. Hugging Face Dataset Adapter
# ==============================================================================
class ODELIAHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        
        img_pre = np.array(item['Image_Pre'], dtype=np.float32).squeeze()
        img_post1 = np.array(item['Image_Post_1'], dtype=np.float32).squeeze()
        img_t2 = np.array(item['Image_T2'], dtype=np.float32).squeeze()
        
        # Subtraction image
        img_sub1 = img_post1 - img_pre
        
        # 3-modality stacking: (3, 32, 256, 256)
        image_stacked = np.stack([img_pre, img_sub1, img_t2], axis=0)
        
        # Label 0=Normal,1=Benign,2=Malignant
        label = int(item['Lesion'])
        
        data_dict = {
            "image": image_stacked,
            "label": np.array(label, dtype=np.int64)
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict

# ==============================================================================
# 2. Main Classification Model
# ==============================================================================
class ODELIAClassifier:
    train_epochs = 50
    n_classes = 3
    batch_size = 8
    img_size = (32, 256, 256)  # Z x H x W

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    keys = ("image", "label")
    
    # Training data augmentation
    train_tform = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=img_size),
        RandRotate90d(keys="image", prob=0.5, spatial_axes=(1, 2)),
        RandFlipd(keys="image", prob=0.5, spatial_axis=1),
        EnsureTyped(keys=keys, dtype=(torch.float32, torch.int64))
    ])

    val_tform = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=img_size),
        EnsureTyped(keys=keys, dtype=(torch.float32, torch.int64))
    ])

    # Softmax output (requirement: calibrated probability distribution)
    post_pred = Compose([EnsureTyped(keys="pred"), Activationsd(keys="pred", softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=n_classes)])

    # 3D ResNet model
    model = resnet10(
        spatial_dims=3,
        n_input_channels=3,
        num_classes=n_classes
    ).to(device)

    trainer = None

    # --------------------------
    # Data Loading
    # --------------------------
    @classmethod
    def load_odelia_hf_data(cls, split_type='train'):
        logging.info(f"Loading {split_type} split...")
        dataset = load_dataset("ODELIA-AI/ODELIA-Challenge-2025", name="unilateral")
        hf_split = dataset[split_type]
        tform = cls.train_tform if split_type == 'train' else cls.val_tform
        monai_ds = ODELIAHuggingFaceDataset(hf_split, transform=tform)

        num_workers = 4 if torch.cuda.is_available() else 0
        loader = DataLoader(
            monai_ds,
            batch_size=cls.batch_size,
            shuffle=(split_type == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        return loader

    # --------------------------
    # Main Training Function
    # --------------------------
    @classmethod
    def train(cls):
        config.print_config()
        train_loader = cls.load_odelia_hf_data('train')
        val_loader = cls.load_odelia_hf_data('val')

        net = cls.model
        loss_function = CrossEntropyLoss()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)

        logdir = "./runs/odelia_breast_classifier"
        os.makedirs(logdir, exist_ok=True)

        # Evaluator
        val_metrics = {
            # Restore the most reasonable usage: get pred and label directly from the dictionary
            "val_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))
        }

        evaluator = SupervisedEvaluator(
            device=cls.device,
            val_data_loader=val_loader,
            network=net,
            decollate=False,  # Critical fix: disable auto-decollate, keep dict format!
            key_val_metric=val_metrics,
        )

        # Trainer
        trainer = SupervisedTrainer(
            device=cls.device,
            max_epochs=cls.train_epochs,
            train_data_loader=train_loader,
            network=net,
            optimizer=opt,
            loss_function=loss_function,
            amp=True
        )
        cls.trainer = trainer

        # Learning rate scheduler
        lr_scheduler = StepLR(opt, step_size=20, gamma=0.1)
        LrScheduleHandler(lr_scheduler).attach(trainer)

        # ========================
        # Fix: MONAI 1.5.2 dedicated CheckpointSaver
        # ========================
        CheckpointSaver(
            save_dir=logdir,
            save_dict={"net": net},
            save_interval=5,  # Use save_interval instead of save_freq, supported in 1.5.2
            epoch_level=True
        ).attach(trainer)

        # Logging
        StatsHandler(output_transform=lambda x: x[0]['loss']).attach(trainer)
        tb_writer = SummaryWriter(log_dir=logdir)
        TensorBoardStatsHandler(summary_writer=tb_writer, tag_name="train_loss").attach(trainer)

        # Validation every 5 epochs
        ValidationHandler(validator=evaluator, interval=5, epoch_level=True).attach(trainer)

        logging.info("Starting training...")
        trainer.run()
        logging.info("Training completed!")

# ==============================================================================
# 3. Command Line Interface
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='ODELIA Breast 3-Class Classification')
    subparsers = parser.add_subparsers(dest='mode', required=True)
    subparsers.add_parser('train', help="Train the model")
    return parser.parse_args()

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    start = timer()
    args = parse_args()

    if args.mode == 'train':
        ODELIAClassifier.train()

    end = timer()
    print(f"\n======================================")
    print(f"Total computing time: {end - start:.2f} seconds")
    print(f"======================================")

if __name__ == "__main__":
    main()