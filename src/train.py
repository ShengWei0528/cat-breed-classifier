"""Train and evaluate the ResNet50 cat-breed classifier.

Command-line version of ``notebooks/cat_breed_classifier.ipynb``. The model,
data split, augmentation and hyper-parameters are the same as in the executed
notebook; this script only adds CLI arguments, an output folder and two extra
artefacts (``class_names.txt`` and ``training_log.csv``).

Dataset: Cat Breeds Dataset by ma7555 on Kaggle
https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset
Expected layout: <data-dir>/<Breed name>/<image>.jpg

Example:
    python src/train.py --data-dir images --output-dir outputs --epochs 30
"""
import argparse
import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import CatBreedClassifier, eval_transform, train_transform

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ==================== Dataset ====================
class CatBreedsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, IOError):
            # A few files in the dataset are unreadable; use a black placeholder
            # instead of crashing the epoch.
            print(f"\nWarning: failed to load image {img_path}, using a black placeholder.")
            image = Image.new("RGB", (128, 128), color="black")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ==================== Data preparation ====================
def prepare_data(data_dir, test_size=0.2, val_size=0.1, min_images_per_class=10, seed=42):
    """Scan breed folders, drop rare breeds, and make a stratified 70/10/20 split."""
    print("Loading data...")
    class_image_counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTS)]
            class_image_counts[class_name] = len(imgs)

    valid_classes = []
    for class_name, count in class_image_counts.items():
        if count >= min_images_per_class:
            valid_classes.append(class_name)
        else:
            print(f"Warning: {class_name} has only {count} images, skipped "
                  f"(minimum required: {min_images_per_class})")

    image_paths, labels, class_names = [], [], []
    for class_idx, class_name in enumerate(sorted(valid_classes)):
        class_dir = os.path.join(data_dir, class_name)
        class_names.append(class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(IMAGE_EXTS):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)

    print(f"\nFound {len(class_names)} valid breeds (out of {len(class_image_counts)} total)")
    print(f"Total images: {len(image_paths)}")
    if len(class_names) < 2:
        raise ValueError("At least 2 breeds are required for training")

    # Stratified split keeps every class represented; fall back to a random split
    # if a class is too small to stratify.
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=test_size, stratify=labels, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (1 - test_size), stratify=y_temp, random_state=seed)
    except ValueError:
        print("\nWarning: unable to use stratified sampling, using random split")
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=test_size, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (1 - test_size), random_state=seed)

    print("\nDataset split:")
    print(f"  Training set:   {len(X_train)} images")
    print(f"  Validation set: {len(X_val)} images")
    print(f"  Test set:       {len(X_test)} images")

    print("\nBreed distribution in the training set (top 10):")
    for idx, count in Counter(y_train).most_common(10):
        print(f"  {class_names[idx]}: {count} images")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


# ==================== Training ====================
def train_model(model, train_loader, val_loader, num_epochs, lr, device, output_dir):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_path = os.path.join(output_dir, "best_model.pth")

    history = []
    best_acc = 0.0
    print("\nStarting training...")
    print("=" * 50)
    for epoch in range(num_epochs):
        # ---- training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # ---- validation phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss = running_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        # Halve the learning rate when validation loss stops improving.
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        saved = val_acc > best_acc
        if saved:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (Accuracy: {val_acc:.2f}%)")
        history.append((epoch + 1, train_loss, train_acc, val_loss, val_acc, saved))

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    with open(os.path.join(output_dir, "training_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc_pct", "val_loss", "val_acc_pct", "saved_best"])
        for ep, tl, ta, vl, va, s in history:
            writer.writerow([ep, f"{tl:.4f}", f"{ta:.2f}", f"{vl:.4f}", f"{va:.2f}", "yes" if s else "no"])

    plot_curves(history, os.path.join(output_dir, "training_curves.png"))
    return model, best_path


def plot_curves(history, path):
    train_losses = [h[1] for h in history]
    train_accs = [h[2] for h in history]
    val_losses = [h[3] for h in history]
    val_accs = [h[4] for h in history]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Training curves saved as {path}")


# ==================== Evaluation ====================
def test_model(model, test_loader, class_names, device, output_dir):
    model.eval()
    model = model.to(device)
    all_preds, all_labels, all_probs = [], [], []

    print("\nTesting model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean() * 100
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_acc = np.mean([label in top5_preds[i] for i, label in enumerate(all_labels)]) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    report = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                   target_names=class_names, digits=3, zero_division=0)
    print("\nClassification Report:")
    print("=" * 80)
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write("=" * 80 + "\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved as {cm_path}")

    print("\nMost confused breed pairs (count > 3):")
    print("-" * 50)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 3:
                print(f"{class_names[i]} -> {class_names[j]}: {cm[i, j]} times")

    return accuracy, top5_acc


# ==================== Entry point ====================
def parse_args():
    p = argparse.ArgumentParser(description="Train a ResNet50 cat-breed classifier.")
    p.add_argument("--data-dir", default="images", help="folder with one sub-folder per breed")
    p.add_argument("--output-dir", default="outputs", help="where weights, plots and reports go")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256, help="lower this if the GPU runs out of memory")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-images", type=int, default=10, help="drop breeds with fewer images than this")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: 0 on Windows, 4 elsewhere)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Cat Breed Classification System")
    print("Dataset: Cat Breeds Dataset (Kaggle)")
    print("Source: https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = prepare_data(
            args.data_dir, min_images_per_class=args.min_images, seed=args.seed)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")
        print(f"Please download the dataset and extract it so that '{args.data_dir}' "
              f"contains one folder per breed.")
        return

    with open(os.path.join(args.output_dir, "class_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

    train_dataset = CatBreedsDataset(X_train, y_train, transform=train_transform)
    val_dataset = CatBreedsDataset(X_val, y_val, transform=eval_transform)
    test_dataset = CatBreedsDataset(X_test, y_test, transform=eval_transform)

    batch_size = min(args.batch_size, len(X_train))
    # Multiprocess data loading is fragile on Windows; default to the main process there.
    num_workers = args.num_workers if args.num_workers is not None else (0 if os.name == "nt" else 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = CatBreedClassifier(num_classes=len(class_names))
    print("\nModel: ResNet50")
    print(f"Number of classes: {len(class_names)}")
    print(f"Batch size: {batch_size}")

    if device == "cuda":
        torch.cuda.empty_cache()
    model, best_path = train_model(model, train_loader, val_loader, args.epochs, args.lr,
                                   device, args.output_dir)

    # Evaluate the checkpoint with the best validation accuracy.
    model.load_state_dict(torch.load(best_path, map_location=device))
    accuracy, top5_acc = test_model(model, test_loader, class_names, device, args.output_dir)

    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Final test accuracy: {accuracy:.2f}%")
    print(f"Top-5 accuracy: {top5_acc:.2f}%")
    print(f"All artefacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
