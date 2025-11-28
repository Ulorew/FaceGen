import logging
import math
import sys
import time

import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm

from checkpoint import save_checkpoint, load_checkpoint
from config import *
from dataset import setup_dataloaders
from ema import EMA
from model import FlowUNet


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler("training.log", mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def get_config_dict():
    """Get current config as dict for saving."""
    return {
        'IMAGE_SIZE': IMAGE_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'LR': LR,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BASE_CHANNELS': BASE_CHANNELS,
        'TRUNC_DATASET': TRUNC_DATASET,
    }


def cosine_schedule(step, warmup, total, min_lr=0.05):
    """Cosine schedule with warmup."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """Remove old checkpoints, keeping only the last N."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoint files (not best_model)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"))

    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            ckpt.unlink()
            logging.info(f"  Removed old checkpoint: {ckpt.name}")


@torch.no_grad()
def generate_samples(model, ema, num=16, steps=50, epoch=0):
    ema.apply_shadow()
    model.eval()

    z = torch.randn(num, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=DEVICE)

    dt = 1.0 / steps
    for i in range(steps):
        t_val = torch.full((num,), i / steps, device=DEVICE)
        v = model(z, t_val)
        z = z + v * dt

    images = (z.clamp(-1, 1) + 1) / 2
    save_image(images, CHECKPOINT_PATH / f"samples_epoch_{epoch:03d}.png", nrow=4)

    ema.restore()
    model.train()


def validate(model, ema, crit, val_loader):
    ema.apply_shadow()
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for X_1, _ in val_loader:
            X_1 = X_1.to(DEVICE)
            batch_size = X_1.size(0)

            X_0 = torch.randn_like(X_1)
            v_true = X_1 - X_0
            t = torch.rand(batch_size, device=DEVICE)
            X_t = X_0 * (1 - t).view(-1, 1, 1, 1) + X_1 * t.view(-1, 1, 1, 1)

            v_pred = model(X_t, t)
            loss = crit(v_pred, v_true)
            total_loss += loss.item()

    ema.restore()
    model.train()
    return total_loss / len(val_loader)


def train_epoch(model, optimizer, scaler, crit, train_loader, ema, epoch, lr_fn, global_step):
    """
    Train one epoch.

    Returns:
        tuple: (average_loss, new_global_step)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    epoch_start = time.time()
    samples_processed = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_id, (X_1, _) in enumerate(pbar):
        # Update LR based on global step
        lr_mult = lr_fn(global_step)
        lr = LR * lr_mult
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        X_1 = X_1.to(DEVICE)
        batch_size = X_1.size(0)

        # Flow matching
        X_0 = torch.randn_like(X_1)
        v_true = X_1 - X_0
        t = torch.rand(batch_size, device=DEVICE)
        X_t = X_0 * (1 - t).view(-1, 1, 1, 1) + X_1 * t.view(-1, 1, 1, 1)

        # Forward + backward
        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            v_pred = model(X_t, t)
            loss = crit(v_pred, v_true)

        if not torch.isfinite(loss):
            logging.warning(f"Non-finite loss at step {global_step}")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
            ema.update()

        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        samples_processed += batch_size
        global_step += 1

        # Throughput
        elapsed = time.time() - epoch_start
        throughput = samples_processed / elapsed if elapsed > 0 else 0

        pbar.set_postfix({
            'loss': f'{total_loss / num_batches:.4f}',
            'lr': f'{lr:.1e}',
            'img/s': f'{throughput:.0f}'
        })

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / max(num_batches, 1)
    throughput = samples_processed / epoch_time

    logging.info(f"  Time: {epoch_time:.1f}s | Throughput: {throughput:.0f} img/s")

    return avg_loss, global_step


def train(override_dataset_size=None):
    setup_logging()

    logging.info("=" * 60)
    logging.info("Flow Matching Training")
    logging.info("=" * 60)

    # =====================================================
    # Setup data
    # =====================================================
    train_loader, val_loader, test_loader = setup_dataloaders(override_dataset_size)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS

    logging.info(f"Dataset: {TRUNC_DATASET} images")
    logging.info(f"Batch size: {BATCH_SIZE}, Steps/epoch: {steps_per_epoch}")
    logging.info(f"Total epochs: {NUM_EPOCHS}, Total steps: {total_steps}")

    # =====================================================
    # Setup model, optimizer, etc.
    # =====================================================
    model = FlowUNet(base_ch=BASE_CHANNELS).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY
    )

    scaler = GradScaler('cuda')
    crit = nn.MSELoss()
    ema = EMA(model, decay=EMA_DECAY, warmup_steps=EMA_WARMUP_STEPS)

    # LR schedule function (takes global step, returns multiplier)
    lr_fn = lambda step: cosine_schedule(step, LR_WARMUP_STEPS, total_steps, min_lr=0.05)

    # =====================================================
    # Resume from checkpoint if specified
    # =====================================================
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if RESUME_FROM is not None and Path(RESUME_FROM).exists():
        logging.info(f"Resuming from checkpoint: {RESUME_FROM}")

        state = load_checkpoint(
            RESUME_FROM,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=DEVICE
        )

        start_epoch = state['epoch'] + 1  # Start from next epoch
        global_step = state['global_step']
        best_val_loss = state['best_val_loss']

        logging.info(f"Resuming from epoch {start_epoch}, step {global_step}")
        logging.info(f"Best val loss so far: {best_val_loss:.5f}")

        # Verify config matches (optional but recommended)
        saved_config = state.get('config', {})
        if saved_config:
            current_config = get_config_dict()
            for key in ['IMAGE_SIZE', 'BASE_CHANNELS']:
                if saved_config.get(key) != current_config.get(key):
                    logging.warning(
                        f"Config mismatch: {key} was {saved_config.get(key)}, now {current_config.get(key)}")

    elif RESUME_FROM is not None:
        logging.warning(f"Checkpoint not found: {RESUME_FROM}, starting fresh")

    # =====================================================
    # Training loop
    # =====================================================
    training_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train one epoch
        train_loss, global_step = train_epoch(
            model, optimizer, scaler, crit, train_loader,
            ema, epoch, lr_fn, global_step
        )

        # Validate
        val_loss = validate(model, ema, crit, val_loader)

        logging.info(f"Epoch {epoch:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ema.apply_shadow()
            torch.save(model.state_dict(), CHECKPOINT_PATH / "best_model.pth")
            ema.restore()
            logging.info(f"  → Saved best model (val: {val_loss:.5f})")

        # Save periodic checkpoint (full state for resuming)
        if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
            ckpt_path = CHECKPOINT_PATH / f"checkpoint_{epoch:03d}.pth"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                config=get_config_dict(),
            )

            # Cleanup old checkpoints
            cleanup_old_checkpoints(CHECKPOINT_PATH, keep_last_n=KEEP_LAST_N_CHECKPOINTS)

        # Generate samples
        if (epoch + 1) % SAMPLE_EVERY_EPOCH == 0:
            generate_samples(model, ema, num=16, steps=50, epoch=epoch)

    # =====================================================
    # Training complete
    # =====================================================
    total_time = time.time() - training_start

    # Save final checkpoint
    save_checkpoint(
        path=CHECKPOINT_PATH / "final_checkpoint.pth",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        epoch=NUM_EPOCHS - 1,
        global_step=global_step,
        best_val_loss=best_val_loss,
        config=get_config_dict(),
    )

    logging.info("=" * 60)
    logging.info(f"Training complete!")
    logging.info(f"Total time: {total_time / 3600:.2f} hours")
    logging.info(f"Best val loss: {best_val_loss:.5f}")
    logging.info(f"Final checkpoint: {CHECKPOINT_PATH / 'final_checkpoint.pth'}")


if __name__ == "__main__":
    train()
