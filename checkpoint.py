import torch
from pathlib import Path
import logging


def save_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        ema,
        epoch,
        global_step,
        best_val_loss,
        config=None,
):
    """
    Save complete training state for resuming.

    Args:
        path: Where to save checkpoint
        model: The model
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        ema: EMA wrapper
        epoch: Current epoch number
        global_step: Total training steps so far
        best_val_loss: Best validation loss achieved
        config: Optional dict of config values
    """
    checkpoint = {
        # Model
        'model_state_dict': model.state_dict(),

        # Optimizer
        'optimizer_state_dict': optimizer.state_dict(),

        # Scaler (for AMP)
        'scaler_state_dict': scaler.state_dict(),

        # EMA
        'ema_shadow': ema.shadow,
        'ema_step_count': ema.step_count,
        'ema_decay': ema.decay,

        # Training state
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,

        # Config (for verification)
        'config': config,
    }

    torch.save(checkpoint, path)
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer=None, scaler=None, ema=None, device='cuda'):
    """
    Load checkpoint and restore training state.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scaler: Optional GradScaler to restore
        ema: Optional EMA to restore
        device: Device to load to

    Returns:
        dict with epoch, global_step, best_val_loss
    """
    logging.info(f"Loading checkpoint from {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not "model_state_dict" in list(checkpoint.keys()):
        logging.info("  Loaded only model weights")
        model.load_state_dict(checkpoint)
        return {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'config': checkpoint.get('config', None),
        }

    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("  ✓ Loaded model weights")

    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("  ✓ Loaded optimizer state")

    # Load scaler
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logging.info("  ✓ Loaded scaler state")

    # Load EMA
    if ema is not None and 'ema_shadow' in checkpoint:
        ema.shadow = checkpoint['ema_shadow']
        ema.step_count = checkpoint['ema_step_count']
        ema.decay = checkpoint['ema_decay']
        logging.info("  ✓ Loaded EMA state")

    # Return training state
    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'config': checkpoint.get('config', None),
    }
