from pathlib import Path
import torch

SEED = 42
torch.manual_seed(SEED)

DATASET_PATH = Path("dataset/")
CHECKPOINT_PATH = Path("checkpoints/")
CHECKPOINT_PATH.mkdir(exist_ok=True)

# ★ Resume settings
RESUME_FROM = None #Path("amazing_checkpoints_2/")/"checkpoint_149.pth"  # Set to checkpoint path to resume, e.g., "checkpoints/checkpoint_020.pth"
# RESUME_FROM = CHECKPOINT_PATH / "checkpoint_020.pth"  # Example

# Data
IMAGE_SIZE = 128
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
TRUNC_DATASET = 80000

# Training
DEVICE = "cuda"
BATCH_SIZE = 128
ACCUMULATION_STEP = 1

NUM_EPOCHS = 100
LR = 3e-4
LR_WARMUP_STEPS = 300
WEIGHT_DECAY = 0.01

# EMA
EMA_DECAY = 0.9999
EMA_WARMUP_STEPS = 500

# Model
BASE_CHANNELS = 128

# Data split
TRAIN_DATA_RATIO = 0.90
VAL_DATA_RATIO = 0.05
TEST_DATA_RATIO = 0.05

# Checkpointing
SAVE_EVERY_EPOCH = 10
SAMPLE_EVERY_EPOCH = 5
KEEP_LAST_N_CHECKPOINTS = 3  # Delete old checkpoints to save space


