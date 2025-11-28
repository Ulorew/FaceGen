from pathlib import Path
from typing import Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

from config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, CHECKPOINT_PATH
from model import FlowUNet


class FlowMatchingInverter:
    """
    Invert images to find their corresponding noise vectors.

    Supports:
        - Direct ODE inversion (reverse time integration)
        - Optimization-based inversion (more accurate)
        - DDIM-style inversion
        - Noise optimization with reconstruction loss
    """

    def __init__(
            self,
            checkpoint_path: Union[str, Path],
            device: str = "cuda",
            base_channels: int = 64,
    ):
        self.device = device
        self.model = FlowUNet(base_ch=base_channels).to(device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        print(f"Model loaded from {checkpoint_path}")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

    # =========================================================================
    # Method 1: Direct ODE Inversion (Reverse Time)
    # =========================================================================

    @torch.no_grad()
    def invert_ode(
            self,
            image: torch.Tensor,
            steps: int = 100,
            show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Invert image by running ODE backwards in time.

        This integrates from t=1 (image) to t=0 (noise).

        Args:
            image: Target image tensor (B, 3, H, W) in range [-1, 1]
            steps: Number of integration steps
            show_progress: Show progress bar

        Returns:
            Inverted noise tensor (B, 3, H, W)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        z = image.clone().to(self.device)

        dt = 1.0 / steps
        timesteps = range(steps)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Inverting (ODE)")

        for i in timesteps:
            # Go backwards: t goes from 1 to 0
            t = 1.0 - i / steps
            t_tensor = torch.full((z.size(0),), t, device=self.device)

            v = self.model(z, t_tensor)
            z = z - v * dt  # Subtract instead of add (reverse direction)

        return z

    @torch.no_grad()
    def invert_ode_heun(
            self,
            image: torch.Tensor,
            steps: int = 100,
            show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Invert using Heun's method (more accurate).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        z = image.clone().to(self.device)

        dt = 1.0 / steps
        timesteps = range(steps)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Inverting (Heun)")

        for i in timesteps:
            t = 1.0 - i / steps
            t_next = 1.0 - (i + 1) / steps

            t_tensor = torch.full((z.size(0),), t, device=self.device)
            t_next_tensor = torch.full((z.size(0),), max(t_next, 0), device=self.device)

            # Predictor
            v1 = self.model(z, t_tensor)
            z_pred = z - v1 * dt

            # Corrector
            v2 = self.model(z_pred, t_next_tensor)
            z = z - (v1 + v2) * 0.5 * dt

        return z

    # =========================================================================
    # Method 2: Optimization-Based Inversion
    # =========================================================================

    def invert_optimize(
            self,
            image: torch.Tensor,
            steps: int = 50,
            num_iterations: int = 500,
            lr: float = 0.1,
            show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Find noise by optimizing reconstruction loss.

        More accurate than ODE inversion but slower.

        Args:
            image: Target image (B, 3, H, W) in [-1, 1]
            steps: ODE steps for forward pass
            num_iterations: Optimization iterations
            lr: Learning rate
            show_progress: Show progress bar

        Returns:
            Tuple of (optimized noise, loss history)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Initialize noise (can start from ODE inversion for better init)
        z_init = self.invert_ode(image, steps=steps, show_progress=False)
        z = z_init.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

        loss_history = []

        iterations = range(num_iterations)
        if show_progress:
            iterations = tqdm(iterations, desc="Optimizing")

        for i in iterations:
            optimizer.zero_grad()

            # Forward pass: noise -> reconstructed image
            recon = self._forward_ode(z, steps)

            # Reconstruction loss
            loss = F.mse_loss(recon, image)

            # Optional: Add regularization to keep noise Gaussian-like
            # reg_loss = 0.01 * (z.pow(2).mean() - 1).pow(2)
            # loss = loss + reg_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())

            if show_progress:
                iterations.set_postfix({'loss': f'{loss.item():.6f}'})

        return z.detach(), loss_history

    def _forward_ode(self, z: torch.Tensor, steps: int) -> torch.Tensor:
        """Forward ODE pass (for use in optimization)."""
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((z.size(0),), i / steps, device=self.device)
            v = self.model(z, t)
            z = z + v * dt

        return z

    # =========================================================================
    # Method 3: Encode-Decode (Fixed Point Iteration)
    # =========================================================================

    @torch.no_grad()
    def invert_fixed_point(
            self,
            image: torch.Tensor,
            steps: int = 50,
            num_iterations: int = 10,
            show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Fixed-point iteration for inversion.

        Iteratively refines the noise estimate.

        Args:
            image: Target image
            steps: ODE steps
            num_iterations: Number of refinement iterations

        Returns:
            Inverted noise
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Initial estimate from direct inversion
        z = self.invert_ode(image, steps=steps, show_progress=False)

        iterations = range(num_iterations)
        if show_progress:
            iterations = tqdm(iterations, desc="Fixed-point iteration")

        for _ in iterations:
            # Forward: z -> reconstructed
            recon = self.forward(z, steps=steps, show_progress=False)

            # Compute error
            error = image - recon

            # Invert the error
            error_inverted = self.invert_ode(error, steps=steps, show_progress=False)

            # Update z
            z = z + error_inverted * 0.5  # Damped update

        return z

    # =========================================================================
    # Forward Pass (for verification)
    # =========================================================================

    @torch.no_grad()
    def forward(
            self,
            z: torch.Tensor,
            steps: int = 50,
            show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Forward ODE: noise -> image.
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)

        z = z.clone().to(self.device)

        dt = 1.0 / steps
        timesteps = range(steps)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Forward")

        for i in timesteps:
            t = torch.full((z.size(0),), i / steps, device=self.device)
            v = self.model(z, t)
            z = z + v * dt

        return z

    # =========================================================================
    # High-Level Functions
    # =========================================================================

    def encode(
            self,
            image: torch.Tensor,
            method: str = "ode",
            steps: int = 100,
            **kwargs
    ) -> torch.Tensor:
        """
        Encode image to noise space.

        Args:
            image: Image tensor in [-1, 1]
            method: "ode", "heun", "optimize", or "fixed_point"
            steps: Number of steps

        Returns:
            Noise tensor
        """
        if method == "ode":
            return self.invert_ode(image, steps, **kwargs)
        elif method == "heun":
            return self.invert_ode_heun(image, steps, **kwargs)
        elif method == "optimize":
            z, _ = self.invert_optimize(image, steps, **kwargs)
            return z
        elif method == "fixed_point":
            return self.invert_fixed_point(image, steps, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def decode(
            self,
            z: torch.Tensor,
            steps: int = 50,
    ) -> torch.Tensor:
        """
        Decode noise to image space.
        """
        return self.forward(z, steps)

    def reconstruct(
            self,
            image: torch.Tensor,
            method: str = "ode",
            steps: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode then decode an image.

        Returns:
            Tuple of (noise, reconstruction)
        """
        z = self.encode(image, method, steps)
        recon = self.decode(z, steps)
        return z, recon

    # =========================================================================
    # Interpolation Between Real Images
    # =========================================================================

    def interpolate_images(
            self,
            image1: torch.Tensor,
            image2: torch.Tensor,
            num_steps: int = 8,
            inversion_steps: int = 100,
            generation_steps: int = 50,
            method: str = "spherical",
    ) -> torch.Tensor:
        """
        Interpolate between two real images.

        1. Invert both images to noise
        2. Interpolate in noise space
        3. Decode interpolated noises

        Args:
            image1: First image
            image2: Second image
            num_steps: Number of interpolation steps
            inversion_steps: Steps for inversion
            generation_steps: Steps for generation
            method: "linear" or "spherical"

        Returns:
            Tensor of interpolated images
        """
        # Invert both images
        print("Inverting image 1...")
        z1 = self.encode(image1, method="heun", steps=inversion_steps)
        print("Inverting image 2...")
        z2 = self.encode(image2, method="heun", steps=inversion_steps)

        # Interpolate
        alphas = torch.linspace(0, 1, num_steps, device=self.device)

        if method == "linear":
            z_interp = torch.stack([
                (1 - a) * z1 + a * z2 for a in alphas
            ]).squeeze(1)
        elif method == "spherical":
            # Spherical interpolation
            z1_flat = z1.flatten(1)
            z2_flat = z2.flatten(1)

            z1_norm = z1_flat / z1_flat.norm(dim=1, keepdim=True)
            z2_norm = z2_flat / z2_flat.norm(dim=1, keepdim=True)

            dot = (z1_norm * z2_norm).sum(dim=1).clamp(-1, 1)
            omega = torch.acos(dot)

            interps = []
            for a in alphas:
                if omega.abs() < 1e-6:
                    interp = (1 - a) * z1_flat + a * z2_flat
                else:
                    interp = (torch.sin((1 - a) * omega) / torch.sin(omega)).unsqueeze(1) * z1_flat + \
                             (torch.sin(a * omega) / torch.sin(omega)).unsqueeze(1) * z2_flat
                interps.append(interp.view_as(z1))
            z_interp = torch.cat(interps, dim=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Decode all
        print("Generating interpolations...")
        images = self.decode(z_interp, steps=generation_steps)

        return images.clamp(-1, 1)

    # =========================================================================
    # Image Editing
    # =========================================================================

    def edit_image(
            self,
            image: torch.Tensor,
            edit_direction: torch.Tensor,
            strength: float = 1.0,
            inversion_steps: int = 100,
            generation_steps: int = 50,
    ) -> torch.Tensor:
        """
        Edit image by moving in noise space.

        Args:
            image: Source image
            edit_direction: Direction to move in noise space
            strength: How far to move
            inversion_steps: Steps for inversion
            generation_steps: Steps for generation

        Returns:
            Edited image
        """
        z = self.encode(image, method="heun", steps=inversion_steps)
        z_edited = z + strength * edit_direction.to(self.device)
        edited = self.decode(z_edited, steps=generation_steps)
        return edited.clamp(-1, 1)


# ==============================================================================
# Utility Functions
# ==============================================================================

def load_image(path: Union[str, Path], size: Tuple[int, int] = None) -> torch.Tensor:
    """Load and preprocess image."""
    if size is None:
        size = (IMAGE_HEIGHT, IMAGE_WIDTH)

    image = read_image(str(path))
    image = image.to(torch.float32) / 127.5 - 1.0
    image = image[:3]  # Remove alpha if present

    transform = transforms.Compose([
        transforms.CenterCrop(min(image.shape[-2], image.shape[-1])),
        transforms.Resize(size, antialias=True),
    ])

    return transform(image)


def show_images(images: List[torch.Tensor], titles: List[str] = None, figsize: Tuple = None):
    """Display multiple images side by side."""
    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]
        img = (img.clamp(-1, 1) + 1) / 2
        img = img.permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


def compute_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute reconstruction metrics."""
    original = original.to(reconstructed.device)

    mse = F.mse_loss(reconstructed, original).item()
    psnr = 10 * np.log10(4 / mse)  # 4 because range is [-1, 1], so max diff is 2, squared is 4

    # Compute LPIPS if available (optional)
    lpips = None

    return {
        'mse': mse,
        'psnr': psnr,
        'lpips': lpips,
    }


# ==============================================================================
# Demo
# ==============================================================================

def demo_inversion():
    """Demonstrate image inversion capabilities."""

    print("=" * 60)
    print("Flow Matching Image Inversion Demo")
    print("=" * 60)

    # Load model
    checkpoint_path = CHECKPOINT_PATH / "best_model.pth"
    inverter = FlowMatchingInverter(checkpoint_path)

    # Load test image
    # You can replace this with any image
    test_image_path = "test_image.jpg"  # Replace with your image

    if Path(test_image_path).exists():
        image = load_image(test_image_path)
    else:
        # Generate a random image from the model for testing
        print("No test image found, generating one...")
        torch.manual_seed(42)
        z_random = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=DEVICE)
        image = inverter.decode(z_random, steps=50)[0].cpu()

    print(f"\nImage shape: {image.shape}")

    # Test different inversion methods
    print("\n--- Testing Inversion Methods ---\n")

    results = {}

    # Method 1: ODE inversion
    print("1. ODE Inversion (Euler)...")
    z_ode = inverter.encode(image, method="ode", steps=100)
    recon_ode = inverter.decode(z_ode, steps=100)
    results['ODE'] = (z_ode, recon_ode)

    # Method 2: Heun inversion
    print("\n2. ODE Inversion (Heun)...")
    z_heun = inverter.encode(image, method="heun", steps=100)
    recon_heun = inverter.decode(z_heun, steps=100)
    results['Heun'] = (z_heun, recon_heun)

    # Method 3: Optimization-based
    print("\n3. Optimization-based Inversion...")
    z_opt, loss_history = inverter.invert_optimize(image, steps=50, num_iterations=200, lr=0.05)
    recon_opt = inverter.decode(z_opt, steps=50)
    results['Optimize'] = (z_opt, recon_opt)

    # Show results
    print("\n--- Results ---\n")

    images_to_show = [image]
    titles = ['Original']

    for name, (z, recon) in results.items():
        images_to_show.append(recon[0].cpu())
        titles.append(f'{name} Recon')

        metrics = compute_reconstruction_error(image, recon[0].cpu())
        print(f"{name}: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f} dB")

    show_images(images_to_show, titles)

    # Plot optimization loss
    if loss_history:
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.title('Optimization-based Inversion Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    return inverter, results


def demo_interpolation():
    """Demonstrate interpolation between real images."""

    print("=" * 60)
    print("Image Interpolation Demo")
    print("=" * 60)

    checkpoint_path = CHECKPOINT_PATH / "best_model.pth"
    inverter = FlowMatchingInverter(checkpoint_path)

    # Generate two random images for demo
    torch.manual_seed(42)
    z1 = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=DEVICE)
    torch.manual_seed(123)
    z2 = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=DEVICE)

    image1 = inverter.decode(z1, steps=50)[0].cpu()
    image2 = inverter.decode(z2, steps=50)[0].cpu()

    # Or load real images:
    # image1 = load_image("image1.jpg")
    # image2 = load_image("image2.jpg")

    print("Interpolating between images...")
    interpolated = inverter.interpolate_images(
        image1, image2,
        num_steps=8,
        inversion_steps=100,
        generation_steps=50,
        method="spherical"
    )

    # Show interpolation
    interp_list = [interpolated[i].cpu() for i in range(len(interpolated))]

    fig, axes = plt.subplots(1, len(interp_list), figsize=(2 * len(interp_list), 2))
    for i, img in enumerate(interp_list):
        img_np = (img.clamp(-1, 1) + 1) / 2
        img_np = img_np.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f'α={i / (len(interp_list) - 1):.2f}')

    plt.suptitle('Image Interpolation in Noise Space')
    plt.tight_layout()
    plt.show()

    # Save as grid
    save_image((interpolated + 1) / 2, "interpolation_result.png", nrow=len(interpolated))
    print("Saved interpolation to interpolation_result.png")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flow Matching Image Inversion")
    parser.add_argument("--image", type=str, help="Image to invert")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--method", type=str, default="heun", choices=["ode", "heun", "optimize"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--interpolate", action="store_true", help="Run interpolation demo")

    args = parser.parse_args()

    if args.demo:
        demo_inversion()
    elif args.interpolate:
        demo_interpolation()
    elif args.image:
        # Load and invert specific image
        checkpoint_path = args.checkpoint or CHECKPOINT_PATH / "best_model.pth"
        #inverter = FlowMatchingInverter(checkpoint_path)

        image = load_image(args.image)
        print(f"Loaded image: {args.image}")

        # Invert
        z = inverter.encode(image, method=args.method, steps=args.steps)

        # Reconstruct
        recon = inverter.decode(z, steps=args.steps)

        # Show
        show_images([image, recon[0].cpu()], ['Original', 'Reconstructed'])

        # Compute metrics
        metrics = compute_reconstruction_error(image, recon[0].cpu())
        print(f"Reconstruction: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f} dB")

        # Save noise
        torch.save(z.cpu(), "inverted_noise.pt")
        print("Saved inverted noise to inverted_noise.pt")
    else:
        parser.print_help()
