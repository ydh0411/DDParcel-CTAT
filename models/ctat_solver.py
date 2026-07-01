
# models/ctat_solver.py
"""CTAT training loop with alpha-entmax annealing schedule."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.losses import DiceLoss, CrossEntropy2D
from models.ctat_network import CTAT


class CombinedLoss(nn.Module):
    """Dice + weighted CE, copied from DDParcel losses.py pattern."""
    def __init__(self, weight_dice=1, weight_ce=1):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = CrossEntropy2D()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, pred, target, weight_map):
        target = target.long().to(pred.device)
        if weight_map is None:
            weight_map = torch.ones_like(target, dtype=torch.float32)
        soft_pred = F.softmax(pred, dim=1)
        dice_val = self.dice_loss(soft_pred, target).mean()
        ce_val = (self.ce_loss(pred, target) * weight_map).mean()
        return self.weight_dice * dice_val + self.weight_ce * ce_val, dice_val, ce_val


class AlphaScheduler:
    """
    Linear alpha annealing: alpha_start -> alpha_end over n_steps.
    alpha=1.0 = softmax (dense), alpha=2.0 = sparsemax (competitive).
    """
    def __init__(self, alpha_start=1.0, alpha_end=2.0, total_epochs=100, steps_per_epoch=1):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_steps = total_epochs * steps_per_epoch
        self.current_step = 0

    def step(self):
        self.current_step += 1
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

    def get_alpha(self):
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress


class CTATSolver:
    """Training loop for CTAT with deep supervision and alpha annealing."""
    def __init__(self, model, train_loader, val_loader=None, lr=1e-4, weight_decay=0.05,
                 alpha_start=1.0, alpha_end=2.0, total_epochs=100,
                 ds_weights=None,  # aux loss weights (deep->shallow)
                 device='cuda', exp_dir='./experiments/ctat'):
        if ds_weights is None:
            ds_weights = [0.25, 0.5, 0.75]
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.exp_dir = exp_dir
        self.total_epochs = total_epochs
        self.ds_weights = ds_weights

        os.makedirs(exp_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6)
        self.alpha_scheduler = AlphaScheduler(
            alpha_start,
            alpha_end,
            total_epochs,
            steps_per_epoch=len(train_loader),
        )
        self.loss_fn = CombinedLoss(weight_dice=1, weight_ce=1)
        self.best_val_dice = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            weights = batch['weight'].to(self.device) if 'weight' in batch else None

            # Alpha annealing step
            alpha = self.alpha_scheduler.step()
            self.model.set_alpha(alpha)

            self.optimizer.zero_grad()
            main_logits, aux_logits = self.model(images, return_aux=True)

            # Main loss
            main_loss, dice_val, ce_val = self.loss_fn(main_logits, labels, weights)
            loss = main_loss

            # Deep supervision auxiliary losses
            for i, aux_logit in enumerate(aux_logits):
                aux_loss, _, _ = self.loss_fn(aux_logit, labels, weights)
                loss = loss + self.ds_weights[i] * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler_lr.step()
        avg_loss = total_loss / len(self.train_loader)
        current_alpha = self.alpha_scheduler.get_alpha()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{self.total_epochs} | Loss: {avg_loss:.4f} | "
              f"Alpha: {current_alpha:.2f} | LR: {current_lr:.2e}")
        return avg_loss

    def validate(self):
        self.model.eval()
        # Set alpha=2.0 for validation (full competition)
        self.model.set_alpha(2.0)
        total_dice = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                main_logits = self.model(images, return_aux=False)
                soft_pred = F.softmax(main_logits, dim=1)
                # Per-class dice
                pred = soft_pred.argmax(dim=1)
                dice_scores = []
                for c in range(1, soft_pred.size(1)):  # skip background
                    pred_c = (pred == c).float()
                    target_c = (labels == c).float()
                    inter = (pred_c * target_c).sum()
                    union = pred_c.sum() + target_c.sum()
                    if union > 0:
                        dice_scores.append((2 * inter / union).item())
                if dice_scores:
                    total_dice += sum(dice_scores) / len(dice_scores)
                n_batches += 1
        avg_dice = total_dice / max(n_batches, 1)
        print(f"Validation Dice: {avg_dice:.4f}")
        return avg_dice

    def train(self):
        for epoch in range(self.total_epochs):
            self.train_epoch(epoch)
            if self.val_loader and (epoch + 1) % 5 == 0:
                val_dice = self.validate()
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_dice': val_dice,
                    }, os.path.join(self.exp_dir, 'best_model.pkl'))
                    print(f"Saved best model (Dice: {val_dice:.4f})")
        # Save final checkpoint
        torch.save({
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.exp_dir, 'final_model.pkl'))
