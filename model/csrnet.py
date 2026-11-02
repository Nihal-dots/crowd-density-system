"""CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly
Congested Scenes (Li et al., CVPR 2018).

A VGG-16 front-end (conv1_1..conv4_3) followed by a dilated convolution
back-end. The network maps an image (N, 3, H, W) to a density map
(N, 1, H/8, W/8) whose sum is the estimated crowd count.

Usage (from the project root):
    python model/csrnet.py                  # summary + self-test
    python model/csrnet.py --no-pretrained  # skip the ImageNet weight download
"""

import argparse
import sys

import torch
import torch.nn as nn
from torchvision import models

# VGG-16 up to conv4_3: numbers are output channels, "M" is a 2x2 max-pool
FRONTEND_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
BACKEND_CFG = [512, 512, 512, 256, 128, 64]


def make_layers(cfg, in_channels=3, dilation=1):
    """Build a stack of conv3x3 + ReLU layers (and max-pools for "M")."""
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, pretrained=True, backend_dilation=2):
        """pretrained: initialise the front-end from torchvision VGG-16 ImageNet weights.

        backend_dilation: dilation rate of the back-end (2 in the paper; 1 gives
        the plain-convolution variant used for ablation).
        """
        super().__init__()
        self.frontend = make_layers(FRONTEND_CFG)
        self.backend = make_layers(BACKEND_CFG, in_channels=512, dilation=backend_dilation)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()
        if pretrained:
            self._load_vgg16_frontend()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _load_vgg16_frontend(self):
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # The front-end has the same layer order as the start of vgg.features,
        # so the state dict keys line up one to one.
        self.frontend.load_state_dict(vgg.features[: len(self.frontend)].state_dict())

    def summary(self):
        """Print parameter counts per block and in total."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        blocks = [
            ("Front-end (VGG-16 conv1_1..conv4_3)", self.frontend),
            ("Back-end (6 dilated conv3x3)", self.backend),
            ("Output (conv1x1)", self.output_layer),
        ]
        total = count(self)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("CSRNet")
        for name, module in blocks:
            print(f"  {name:<38}: {count(module):>12,}")
        print(f"  {'Total':<38}: {total:>12,} ({total / 1e6:.2f}M)")
        print(f"  {'Trainable':<38}: {trainable:>12,}")
        return total


def self_test(pretrained=True):
    """Check output shape, gradient flow and weight initialisation."""
    torch.manual_seed(0)
    model = CSRNet(pretrained=pretrained)
    model.summary()
    ok = True

    def check(passed, message):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {message}")

    print("\nSelf-test")
    x = torch.randn(1, 3, 480, 640)
    out = model(x)
    check(tuple(out.shape) == (1, 1, 60, 80), f"forward {tuple(x.shape)} -> {tuple(out.shape)} (expected (1, 1, 60, 80))")

    loss = nn.functional.mse_loss(out, torch.rand_like(out))
    loss.backward()
    without_grad = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0
    ]
    num_tensors = len(list(model.parameters()))
    check(
        not without_grad,
        f"gradients reach {num_tensors - len(without_grad)}/{num_tensors} parameter tensors"
        + (f"; missing: {without_grad}" if without_grad else ""),
    )

    backend_std = torch.cat([p.detach().flatten() for n, p in model.backend.named_parameters() if n.endswith("weight")]).std()
    check(abs(backend_std - 0.01) < 1e-3, f"back-end weight std {backend_std:.4f} (expected 0.01)")

    if pretrained:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        same = all(torch.equal(p, vgg.state_dict()[n]) for n, p in model.frontend.state_dict().items())
        check(same, "front-end weights equal torchvision VGG-16 ImageNet weights")
    else:
        print("  [SKIP] pretrained weight check (--no-pretrained)")

    print("PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSRNet summary and self-test")
    parser.add_argument("--no-pretrained", action="store_true", help="do not load VGG-16 ImageNet weights")
    args = parser.parse_args()
    sys.exit(0 if self_test(pretrained=not args.no_pretrained) else 1)
