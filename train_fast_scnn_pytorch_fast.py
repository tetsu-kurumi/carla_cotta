"""
Train Fast-SCNN in PyTorch using CARLA TFRecord data.
Optimized version with preloading for fast training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Import our PyTorch Fast-SCNN
import sys
sys.path.insert(0, '/home/tetsuk/Downloads/CARLA_0.9.16/cotta_proj')
from fast_scnn_pytorch import FastSCNN


class CARLATFRecordDataset(Dataset):
    """
    Dataset that reads CARLA TFRecord files with preloading for fast access.
    All data is loaded into CPU RAM during initialization.
    """
    def __init__(self, tfrecord_pattern, input_size=(256, 512)):
        import tensorflow as tf
        import cv2

        self.input_size = input_size
        self.tfrecord_files = sorted(tf.io.gfile.glob(tfrecord_pattern))

        if len(self.tfrecord_files) == 0:
            raise ValueError(f"No files found matching: {tfrecord_pattern}")

        print(f"Found {len(self.tfrecord_files)} TFRecord files")

        # Preload all data into memory for fast access
        self.rgb_data = []
        self.label_data = []

        for tfr_file in tqdm(self.tfrecord_files, desc="Loading TFRecords"):
            dataset = tf.data.TFRecordDataset(tfr_file)
            for raw_record in dataset:
                example = tf.io.parse_single_example(
                    raw_record,
                    {'raw_image': tf.io.FixedLenFeature([600 * 800 * 4], tf.int64)}
                )
                image = tf.reshape(example['raw_image'], [600, 800, 4]).numpy().astype(np.float32)

                # Split and resize
                rgb = image[:, :, :3]
                label = image[:, :, 3]

                rgb = cv2.resize(rgb, (input_size[1], input_size[0]),
                                interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (input_size[1], input_size[0]),
                                  interpolation=cv2.INTER_NEAREST)

                # Normalize and store
                self.rgb_data.append((rgb / 255.0).astype(np.float32))
                self.label_data.append(label.astype(np.int64))

        print(f"Total samples: {len(self.rgb_data)}")

        # Print memory usage
        rgb_mem = sum(arr.nbytes for arr in self.rgb_data) / (1024**3)
        label_mem = sum(arr.nbytes for arr in self.label_data) / (1024**3)
        print(f"Memory usage: RGB={rgb_mem:.2f} GB, Labels={label_mem:.2f} GB, Total={rgb_mem+label_mem:.2f} GB")

    def __len__(self):
        return len(self.rgb_data)

    def __getitem__(self, idx):
        rgb = torch.from_numpy(self.rgb_data[idx]).permute(2, 0, 1).float()
        label = torch.from_numpy(self.label_data[idx]).long()
        return rgb, label


def train_model(
    tfrecord_pattern: str,
    output_dir: str = './checkpoints_pytorch',
    num_classes: int = 13,
    batch_size: int = 2,
    epochs: int = 1,
    learning_rate: float = 0.001,
    input_size: tuple = (480, 640),
    device: str = 'cuda',
    num_workers: int = 4
):
    """
    Train Fast-SCNN on CARLA data.
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Fast-SCNN PyTorch Training (Optimized)")
    print("=" * 60)
    print(f"TFRecord pattern: {tfrecord_pattern}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Input size: {input_size}")
    print(f"Device: {device}")
    print(f"Num workers: {num_workers}")
    print()

    # Create dataset
    print("Loading dataset...")
    dataset = CARLATFRecordDataset(tfrecord_pattern, input_size=input_size)

    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Create model
    print("Creating Fast-SCNN model...")
    model = FastSCNN(num_classes=num_classes)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    # Training loop
    best_miou = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for rgb, label in pbar:
            rgb = rgb.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(rgb)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        confusion_matrix = torch.zeros(num_classes, num_classes)

        with torch.no_grad():
            for rgb, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                rgb = rgb.to(device)
                label = label.to(device)

                output = model(rgb)
                loss = criterion(output, label)
                val_loss += loss.item()

                # Compute confusion matrix for mIoU
                pred = output.argmax(dim=1)
                for t, p in zip(label.view(-1), pred.view(-1)):
                    if t >= 0 and t < num_classes:
                        confusion_matrix[t, p] += 1

        val_loss /= len(val_loader)

        # Compute mIoU
        iou_per_class = []
        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp

            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
                iou_per_class.append(iou.item())

        miou = np.mean(iou_per_class) if iou_per_class else 0.0

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val mIoU: {miou:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/mIoU', miou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if miou > best_miou:
            best_miou = miou
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model (mIoU: {best_miou:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'miou': miou
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        print()

    # Save final model
    final_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to: {final_path}")
    print(f"Best mIoU: {best_miou:.4f}")

    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to: {os.path.join(output_dir, 'logs')}")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Fast-SCNN in PyTorch (Optimized)')
    parser.add_argument('--tfrecord-pattern', type=str,
                       default='/home/tetsuk/Downloads/CARLA_0.9.16/cotta_proj/data_train/segmentation_images_*.tfrecords',
                       help='Pattern to match TFRecord files')
    parser.add_argument('--output-dir', type=str, default='./checkpoints_pytorch',
                       help='Directory to save checkpoints')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')

    args = parser.parse_args()

    train_model(
        tfrecord_pattern=args.tfrecord_pattern,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        num_workers=args.num_workers
    )
