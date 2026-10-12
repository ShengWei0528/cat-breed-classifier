"""Model definition and image preprocessing shared by training and inference.

The architecture, attribute names and transforms are identical to the ones used
in ``notebooks/cat_breed_classifier.ipynb``, so a ``best_model.pth`` produced by
the notebook loads into :class:`CatBreedClassifier` without key remapping.
"""
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 128   # crop fed to the network
RESIZE_SIZE = 144  # shorter side before the centre crop (evaluation only)

# Training: random-resized crop + horizontal flip for augmentation.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation / test / inference: deterministic resize + centre crop.
eval_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CatBreedClassifier(nn.Module):
    """ImageNet-pretrained ResNet50 with a two-layer MLP head.

    Only ``layer4`` and the new head are trainable; ``conv1`` to ``layer3`` stay
    frozen as a generic feature extractor.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # IMAGENET1K_V1 is the checkpoint that the legacy `pretrained=True` flag loaded.
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)

        # Freeze everything except the last residual stage and the classifier.
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        num_features = self.resnet.fc.in_features  # 2048 for ResNet50
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
