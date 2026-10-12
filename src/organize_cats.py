#!/usr/bin/env python3
"""Sort a folder of unsorted cat photos into per-breed sub-folders.

Each image is classified with a checkpoint trained by ``train.py`` (or the
notebook). Predictions below ``--threshold`` confidence go to ``unknown/``.
Files are copied by default; pass ``--move`` to move them instead.

Example:
    python src/organize_cats.py \
        --model outputs/best_model.pth \
        --classes outputs/class_names.txt \
        --input-dir ./messy_cats \
        --output-dir ./sorted_cats \
        --threshold 0.5
"""
import argparse
import os
import shutil
from typing import List

import torch
import torch.nn as nn
from PIL import Image

from model import CatBreedClassifier, eval_transform

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def load_class_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_model(num_classes: int, weights_path: str, device: str) -> nn.Module:
    # Rebuild the exact training architecture so every weight is restored;
    # strict loading fails loudly if the checkpoint does not match.
    model = CatBreedClassifier(num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def is_image_file(fname: str) -> bool:
    return fname.lower().endswith(IMAGE_EXTS)


def iter_images(input_dir: str, output_dir: str):
    out_abs = os.path.abspath(output_dir)
    for root, dirs, files in os.walk(input_dir):
        # Never descend into the output folder if it lives inside the input folder.
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != out_abs]
        for file in files:
            if is_image_file(file):
                yield os.path.join(root, file)


def classify_and_sort(model: nn.Module, class_names: List[str], input_dir: str, output_dir: str,
                      threshold: float, move: bool, device: str):
    os.makedirs(output_dir, exist_ok=True)
    unknown_dir = os.path.join(output_dir, "unknown")
    os.makedirs(unknown_dir, exist_ok=True)

    paths = list(iter_images(input_dir, output_dir))
    total = len(paths)
    for processed, src_path in enumerate(paths, start=1):
        file = os.path.basename(src_path)
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception:
            print(f"[skip unreadable] {src_path}")
            continue

        tensor = eval_transform(img).unsqueeze(0).to(device)  # (1, C, H, W)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)
            conf, idx = probs.max(dim=1)
            conf, idx = conf.item(), idx.item()

        if conf < threshold:
            label, dest_dir = "unknown", unknown_dir
        else:
            label = class_names[idx]
            dest_dir = os.path.join(output_dir, label)
            os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, file)
        if move:
            shutil.move(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
        print(f"[{processed}/{total}] {file} -> {label} ({conf:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Sort cat photos into breed folders with a trained model.")
    parser.add_argument("--model", required=True, help="checkpoint (.pth) saved by train.py")
    parser.add_argument("--classes", required=True, help="class_names.txt, one breed per line")
    parser.add_argument("--input-dir", required=True, help="folder with unsorted images")
    parser.add_argument("--output-dir", required=True, help="destination folder")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="minimum confidence; lower-confidence images go to unknown/ (default 0.5)")
    parser.add_argument("--move", action="store_true", help="move files instead of copying them")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = load_class_names(args.classes)
    model = build_model(len(class_names), args.model, device)
    classify_and_sort(model, class_names, args.input_dir, args.output_dir,
                      args.threshold, args.move, device)


if __name__ == "__main__":
    main()
