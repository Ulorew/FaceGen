import numpy as np
import torch
import torch.nn as nn
import math
from pathlib import Path
from typing import Optional, List, Union

from torchvision.io import read_image
from torchvision.transforms import transforms
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from inversion import FlowMatchingInverter
from model import FlowUNet
from config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, BASE_CHANNELS, CHECKPOINT_PATH


class FlowMatchingInference:
    """
    Inference class for Flow Matching models.

    Supports:
        - Euler sampling (fast)
        - Heun sampling (better quality)
        - Adaptive step sampling
        - Batch generation
        - Interpolation between samples
    """

    def __init__(
            self,
            checkpoint_path: Union[str, Path],
            device: str = "cuda",
            base_channels: int = 64,
    ):
        """
        Initialize inference.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on
            base_channels: Model base channels (must match training)
        """
        self.device = device

        # Load model
        self.model = FlowUNet(base_ch=base_channels).to(device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {device}")

    def _load_checkpoint(self, path: Union[str, Path]):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume it's directly the state dict
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

    @torch.no_grad()
    def sample_euler(
            self,
            num_samples: int = 16,
            steps: int = 50,
            seed: Optional[int] = None,
            show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using Euler method.

        This is the simplest ODE solver:
            z_{t+dt} = z_t + v(z_t, t) * dt

        Args:
            num_samples: Number of images to generate
            steps: Number of integration steps (more = better quality)
            seed: Random seed for reproducibility
            show_progress: Show progress bar

        Returns:
            Tensor of shape (num_samples, 3, H, W) in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Start from noise
        z = torch.randn(num_samples, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=self.device)

        dt = 1.0 / steps
        timesteps = range(steps)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling (Euler)")

        for i in timesteps:
            t = torch.full((num_samples,), i / steps, device=self.device)
            v = self.model(z, t)
            z = z + v * dt

        # Clamp and convert to [0, 1]
        images = (z.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def sample_heun(
            self,
            num_samples: int = 16,
            steps: int = 50,
            seed: Optional[int] = None,
            show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using Heun's method (2nd order).

        This is more accurate than Euler:
            v1 = v(z_t, t)
            z_pred = z_t + v1 * dt
            v2 = v(z_pred, t + dt)
            z_{t+dt} = z_t + (v1 + v2) / 2 * dt

        Args:
            num_samples: Number of images to generate
            steps: Number of integration steps
            seed: Random seed
            show_progress: Show progress bar

        Returns:
            Tensor of shape (num_samples, 3, H, W) in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(num_samples, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=self.device)

        dt = 1.0 / steps
        timesteps = range(steps)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling (Heun)")

        for i in timesteps:
            t = i / steps
            t_tensor = torch.full((num_samples,), t, device=self.device)
            t_next_tensor = torch.full((num_samples,), min(t + dt, 1.0), device=self.device)

            # Predictor (Euler step)
            v1 = self.model(z, t_tensor)
            z_pred = z + v1 * dt

            # Corrector
            v2 = self.model(z_pred, t_next_tensor)

            # Final update (average of velocities)
            z = z + (v1 + v2) * 0.5 * dt

        images = (z.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def sample_rk4(
            self,
            num_samples: int = 16,
            steps: int = 50,
            seed: Optional[int] = None,
            show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using RK4 (4th order Runge-Kutta).

        Most accurate but slowest (4 model evaluations per step).

        Args:
            num_samples: Number of images to generate
            steps: Number of integration steps
            seed: Random seed
            show_progress: Show progress bar

        Returns:
            Tensor of shape (num_samples, 3, H, W) in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(num_samples, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=self.device)

        dt = 1.0 / steps
        timesteps = range(steps)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling (RK4)")

        for i in timesteps:
            t = i / steps

            t1 = torch.full((num_samples,), t, device=self.device)
            t2 = torch.full((num_samples,), t + dt / 2, device=self.device)
            t3 = torch.full((num_samples,), t + dt / 2, device=self.device)
            t4 = torch.full((num_samples,), min(t + dt, 1.0), device=self.device)

            k1 = self.model(z, t1)
            k2 = self.model(z + k1 * dt / 2, t2)
            k3 = self.model(z + k2 * dt / 2, t3)
            k4 = self.model(z + k3 * dt, t4)

            z = z + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        images = (z.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def sample_adaptive(
            self,
            num_samples: int = 16,
            steps: int = 50,
            seed: Optional[int] = None,
            step_schedule: str = "uniform",
            show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples with adaptive time stepping.

        Different schedules concentrate steps at different parts of the trajectory.

        Args:
            num_samples: Number of images to generate
            steps: Number of integration steps
            seed: Random seed
            step_schedule: One of "uniform", "quadratic", "cosine"
            show_progress: Show progress bar

        Returns:
            Tensor of shape (num_samples, 3, H, W) in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(num_samples, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=self.device)

        # Generate time schedule
        if step_schedule == "uniform":
            times = torch.linspace(0, 1, steps + 1, device=self.device)
        elif step_schedule == "quadratic":
            # More steps near t=1 (where details emerge)
            times = torch.linspace(0, 1, steps + 1, device=self.device) ** 2
        elif step_schedule == "cosine":
            # Smooth transition
            t = torch.linspace(0, 1, steps + 1, device=self.device)
            times = 0.5 * (1 - torch.cos(t * math.pi))
        else:
            raise ValueError(f"Unknown schedule: {step_schedule}")

        timesteps = range(steps)
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"Sampling ({step_schedule})")

        for i in timesteps:
            t_curr = times[i]
            t_next = times[i + 1]
            dt = t_next - t_curr

            t_tensor = torch.full((num_samples,), t_curr.item(), device=self.device)
            v = self.model(z, t_tensor)
            z = z + v * dt

        images = (z.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def interpolate(
            self,
            num_interpolations: int = 8,
            steps: int = 50,
            seed: Optional[int] = None,
            method: str = "spherical",
            z1=None,
            z2=None
    ) -> torch.Tensor:
        """
        Generate interpolations between random samples.

        Creates smooth transitions between generated images.

        Args:
            num_interpolations: Number of interpolation steps
            steps: ODE integration steps
            seed: Random seed
            method: "linear" or "spherical" interpolation in noise space

        Returns:
            Tensor of shape (num_interpolations, 3, H, W) in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Sample two noise vectors
        if z1 is None:
            z1 = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=self.device)
        if z2 is None:
            z2 = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=self.device)

        # Interpolate in noise space
        alphas = torch.linspace(0, 1, num_interpolations, device=self.device)

        if method == "linear":
            z_interp = torch.stack([
                (1 - a) * z1 + a * z2 for a in alphas
            ]).squeeze(1)
        elif method == "spherical":
            # Spherical interpolation (slerp) - better for high dimensions
            z1_norm = z1 / z1.norm()
            z2_norm = z2 / z2.norm()
            omega = torch.acos((z1_norm * z2_norm).sum().clamp(-1, 1))

            z_interp = torch.stack([
                (torch.sin((1 - a) * omega) * z1 + torch.sin(a * omega) * z2) / torch.sin(omega)
                for a in alphas
            ]).squeeze(1)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Run ODE for each interpolated noise
        dt = 1.0 / steps

        for i in tqdm(range(steps), desc="Interpolating"):
            t = torch.full((num_interpolations,), i / steps, device=self.device)
            v = self.model(z_interp, t)
            z_interp = z_interp + v * dt

        images = (z_interp.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def sample_batch(
            self,
            total_samples: int,
            batch_size: int = 64,
            steps: int = 50,
            method: str = "euler",
            seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate large number of samples in batches.

        Useful for generating many samples without OOM.

        Args:
            total_samples: Total number of samples to generate
            batch_size: Samples per batch
            steps: ODE integration steps
            method: "euler", "heun", or "rk4"
            seed: Random seed

        Returns:
            Tensor of shape (total_samples, 3, H, W) in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        all_images = []
        num_batches = (total_samples + batch_size - 1) // batch_size

        sample_fn = {
            "euler": self.sample_euler,
            "heun": self.sample_heun,
            "rk4": self.sample_rk4,
        }[method]

        for i in tqdm(range(num_batches), desc="Generating batches"):
            current_batch = min(batch_size, total_samples - i * batch_size)
            images = sample_fn(num_samples=current_batch, steps=steps, show_progress=False)
            all_images.append(images.cpu())

        return torch.cat(all_images, dim=0)

    def save_samples(
            self,
            images: torch.Tensor,
            path: Union[str, Path],
            nrow: int = 4,
    ):
        """Save images to file."""
        save_image(images, path, nrow=nrow, padding=2)
        print(f"Saved {len(images)} images to {path}")

    def visualize(
            self,
            images: torch.Tensor,
            nrow: int = 4,
            figsize: tuple = (12, 12),
            title: str = "Generated Samples",
    ):
        """Display images using matplotlib."""
        grid = make_grid(images, nrow=nrow, padding=2)
        grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=figsize)
        plt.imshow(grid)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def visualize_trajectory(
            self,
            num_snapshots: int = 10,
            steps: int = 50,
            seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Visualize the generation trajectory from noise to image.

        Args:
            num_snapshots: Number of intermediate states to capture
            steps: Total ODE steps
            seed: Random seed

        Returns:
            Tensor of shape (num_snapshots, 3, H, W) showing evolution
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=self.device)

        snapshot_steps = torch.linspace(0, steps - 1, num_snapshots).long().tolist()
        snapshots = []

        dt = 1.0 / steps

        for i in tqdm(range(steps), desc="Generating trajectory"):
            if i in snapshot_steps:
                # Save snapshot
                img = (z.clamp(-1, 1) + 1) / 2
                snapshots.append(img.squeeze(0).cpu())

            t = torch.full((1,), i / steps, device=self.device)
            v = self.model(z, t)
            z = z + v * dt

        # Add final image
        img = (z.clamp(-1, 1) + 1) / 2
        snapshots.append(img.squeeze(0).cpu())

        return torch.stack(snapshots)


# ==============================================================================
# Convenience functions
# ==============================================================================

def quick_sample(
        checkpoint_path: Union[str, Path] = None,
        num_samples: int = 16,
        steps: int = 50,
        method: str = "euler",
        seed: int = 42,
        save_path: Optional[str] = None,
        show: bool = True,
):
    """
    Quick function to generate and optionally display/save samples.

    Args:
        checkpoint_path: Path to checkpoint (uses best_model.pth if None)
        num_samples: Number of samples
        steps: ODE steps
        method: "euler", "heun", or "rk4"
        seed: Random seed
        save_path: Optional path to save images
        show: Whether to display images
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH / "best_model.pth"

    inferencer = FlowMatchingInference(checkpoint_path)

    if method == "euler":
        images = inferencer.sample_euler(num_samples, steps, seed)
    elif method == "heun":
        images = inferencer.sample_heun(num_samples, steps, seed)
    elif method == "rk4":
        images = inferencer.sample_rk4(num_samples, steps, seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_path:
        inferencer.save_samples(images, save_path)

    if show:
        inferencer.visualize(images, title=f"{method.upper()} - {steps} steps")

    return images


def compare_methods(
        checkpoint_path: Union[str, Path] = None,
        steps_list: List[int] = [10, 25, 50, 100],
        seed: int = 42,
        save_dir: Optional[str] = None,
):
    """
    Compare different sampling methods and step counts.

    Args:
        checkpoint_path: Path to checkpoint
        steps_list: List of step counts to compare
        seed: Random seed (same for fair comparison)
        save_dir: Optional directory to save comparisons
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH / "best_model.pth"

    inferencer = FlowMatchingInference(checkpoint_path)

    methods = ["euler", "heun"]
    results = {}

    for method in methods:
        for steps in steps_list:
            print(f"\n{method.upper()} - {steps} steps:")

            sample_fn = getattr(inferencer, f"sample_{method}")
            images = sample_fn(num_samples=4, steps=steps, seed=seed)

            key = f"{method}_{steps}"
            results[key] = images

            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(exist_ok=True)
                inferencer.save_samples(images, save_dir / f"{key}.png", nrow=2)

    # Create comparison grid
    fig, axes = plt.subplots(len(methods), len(steps_list), figsize=(4 * len(steps_list), 4 * len(methods)))

    for i, method in enumerate(methods):
        for j, steps in enumerate(steps_list):
            key = f"{method}_{steps}"
            grid = make_grid(results[key], nrow=2, padding=1)
            grid = grid.permute(1, 2, 0).cpu().numpy()

            ax = axes[i, j] if len(methods) > 1 else axes[j]
            ax.imshow(grid)
            ax.set_title(f"{method.upper()} - {steps} steps")
            ax.axis('off')

    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    return results


image_dir = Path("images")


def load_image(image_name):
    image = read_image(image_dir / image_name)
    image = image.to(torch.float32) / 127.5 - 1.0  # Scale to [-1, 1]
    image = image[:3]
    transform = transforms.Compose([
        transforms.CenterCrop(min(image.shape[-2], image.shape[-1])),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])
    image = transform(image)
    return image


def show_image(image):
    # Convert from [-1, 1] back to [0, 255]
    nump = ((image.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255.0
    nump = nump.astype(np.uint8)  # Use uint8, not int8
    plt.imshow(nump)
    plt.show()

base_channels=64
device="cuda"

def _load_checkpoint(path: Union[str, Path]):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    model = FlowUNet(base_ch=base_channels).to(device)
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume it's directly the state dict
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model
