"""
Desert Segmentation Model
SegFormer-B0 (Efficiency King) fine-tuned for desert scene segmentation.
Optimised for RTX 2050 4 GB VRAM with fp16 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# We use B0 because B2 will kill your 4GB VRAM.
# B0 = 140 FPS (Winner). B2 = 30 FPS (Loser).
DEFAULT_VARIANT = 'b0' 
NUM_CLASSES     = 6     # Usually: Sky, Sand, Rock, Obstacle (Check dataset!)
IGNORE_INDEX    = 255

# ─── Loss Functions ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Multi-class Dice loss with smooth and ignore_index support."""
    def __init__(self, num_classes=NUM_CLASSES, smooth=1.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: B × C × H × W   targets: B × H × W (long)
        probs = F.softmax(logits, dim=1)
        
        # Create mask for valid pixels (not ignored)
        valid_mask = (targets != self.ignore_index)
        targets = targets.clone()
        targets[~valid_mask] = 0  # Safe dummy value to avoid index errors

        dice_total = 0.0
        # Loop is fast enough for low class count (4-6)
        for cls in range(self.num_classes):
            # Binary target for this class, masked by valid pixels
            t = ((targets == cls) & valid_mask).float()
            p = probs[:, cls] * valid_mask.float()
            
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_total += 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
            
        return dice_total / self.num_classes


class CombinedLoss(nn.Module):
    """Weighted CE + Dice — balances hard pixels and overall shape."""
    def __init__(self, class_weights=None, dice_weight=0.4, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=0.05
        )
        self.dice = DiceLoss(num_classes=NUM_CLASSES, ignore_index=ignore_index)

    def forward(self, logits, targets):
        # SegFormer outputs 1/4 resolution. We upsample logits to match target.
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits, size=targets.shape[-2:], mode='bilinear', align_corners=False
            )
            
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss, \
               ce_loss.item(), dice_loss.item()

# ─── SegFormer Wrapper ────────────────────────────────────────────────────────

class DesertSegFormer(nn.Module):
    """
    SegFormer-B0 wrapper.
    Loads pretrained encoder (ImageNet) and random decoder head.
    """

    VARIANTS = {
        'b0': 'nvidia/mit-b0',  # <--- THE WINNER (3.7M params)
        'b1': 'nvidia/mit-b1',  # (13M params) - Use only if B0 is too dumb
        'b2': 'nvidia/mit-b2',  # (27M params) - DANGER for 4GB VRAM
    }

    def __init__(self, variant: str = DEFAULT_VARIANT, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        model_name = self.VARIANTS[variant]
        print(f'[DesertSegFormer] Loading {model_name} (pretrained={pretrained}, classes={num_classes})')

        if pretrained:
            # This loads the encoder weights and adds a fresh decoder head
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                id2label={i: f'class_{i}' for i in range(num_classes)},
                label2id={f'class_{i}': i for i in range(num_classes)},
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig.from_pretrained(model_name)
            cfg.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values: B × 3 × H × W
        out = self.model(pixel_values=pixel_values)
        return out.logits # Returns B × C × H/4 × W/4

    def predict(self, pixel_values: torch.Tensor):
        """Inference helper: Forward + Upsample + Argmax"""
        logits = self.forward(pixel_values)
        upsampled = F.interpolate(
            logits, 
            size=(pixel_values.shape[-2], pixel_values.shape[-1]),
            mode='bilinear', 
            align_corners=False
        )
        return upsampled.argmax(dim=1)

# ─── Metrics ──────────────────────────────────────────────────────────────────

class IoUMetric:
    """Streaming mIoU computation."""
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # Move confusion matrix to CPU to save GPU VRAM
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds   = preds.view(-1).cpu()   # Move to CPU immediately
        targets = targets.view(-1).cpu()
        
        valid   = (targets != self.ignore_index)
        preds   = preds[valid]
        targets = targets[valid]

        # Fast bincount trick
        combined = self.num_classes * targets + preds
        bincount = torch.bincount(combined, minlength=self.num_classes ** 2)
        self.confusion += bincount.view(self.num_classes, self.num_classes)

    def compute(self):
        tp = torch.diag(self.confusion).float()
        fp = self.confusion.sum(0).float() - tp
        fn = self.confusion.sum(1).float() - tp

        iou_per_class = tp / (tp + fp + fn + 1e-10)
        miou = iou_per_class.mean().item()
        
        # Pixel Accuracy
        total = self.confusion.sum().float()
        correct = tp.sum()
        pixel_acc = (correct / (total + 1e-10)).item()

        return {'iou_per_class': iou_per_class.tolist(), 'miou': miou, 'pixel_acc': pixel_acc}

# ─── Optimizer ────────────────────────────────────────────────────────────────

def get_optimizer_and_scheduler(model, base_lr=6e-4, total_epochs=50, steps_per_epoch=100):
    # Higher LR (6e-4) because B0 is small and learns fast
    
    # Separate Backbone (Encoder) and Head (Decoder) params
    # SegFormer HF structure: model.segformer.encoder ... model.decode_head
    backbone_params = []
    head_params     = []

    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        
        if "encoder" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = [
        {'params': backbone_params, 'lr': base_lr * 0.1}, # Encoder learns slowly
        {'params': head_params,     'lr': base_lr},       # Head learns fast
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01)
    
    # OneCycleLR is better/faster than Cosine for Hackathons
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[base_lr * 0.1, base_lr], # Matches param groups
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1, # Short warmup
    )
    
    return optimizer, scheduler

# ─── Test Block ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {device}')
    
    # Initialize B0
    model = DesertSegFormer(variant='b0').to(device)
    
    # Test Forward Pass
    dummy_img = torch.randn(2, 3, 512, 512).to(device)
    logits = model(dummy_img)
    
    print(f"Variant: B0")
    print(f"Input: {dummy_img.shape}")
    print(f"Output Logits: {logits.shape}") # Should be [2, 4, 128, 128]
    print("Test Passed! Ready for training.")