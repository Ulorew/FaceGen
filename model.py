import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_groups(ch):
    for g in [32, 16, 8, 4, 2, 1]:
        if ch % g == 0:
            return g
    return 1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.net(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(get_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(get_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(get_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        head_dim = C // self.heads

        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, head_dim, H * W)
        q, k, v = qkv.unbind(1)

        q = q.permute(0, 1, 3, 2)  # B, heads, HW, head_dim
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class FlowUNet(nn.Module):
    """
    UNet for 96x96 flow matching.

    Architecture:
        Encoder: 96 → 48 → 24 → 12
        Channels: 64 → 128 → 256 → 256
        Decoder mirrors encoder with skip connections
    """

    def __init__(self, base_ch=64, use_attention=True):
        super().__init__()

        ch1 = base_ch  # 64
        ch2 = base_ch * 2  # 128
        ch3 = base_ch * 4  # 256
        emb_dim = base_ch * 4

        self.time_emb = TimeEmbedding(base_ch, emb_dim)

        # =====================
        # Encoder
        # =====================
        self.conv_in = nn.Conv2d(3, ch1, 3, 1, 1)

        # Level 1: 96x96, 64 channels
        self.enc1_block1 = ResBlock(ch1, ch1, emb_dim)
        self.enc1_block2 = ResBlock(ch1, ch1, emb_dim)
        self.down1 = Downsample(ch1)  # 96 → 48

        # Level 2: 48x48, 128 channels
        self.enc2_block1 = ResBlock(ch1, ch2, emb_dim)
        self.enc2_block2 = ResBlock(ch2, ch2, emb_dim)
        self.down2 = Downsample(ch2)  # 48 → 24

        # Level 3: 24x24, 256 channels
        self.enc3_block1 = ResBlock(ch2, ch3, emb_dim)
        self.enc3_block2 = ResBlock(ch3, ch3, emb_dim)
        self.down3 = Downsample(ch3)  # 24 → 12

        # =====================
        # Bottleneck: 12x12, 256 channels
        # =====================
        self.mid_block1 = ResBlock(ch3, ch3, emb_dim)
        self.mid_attn = Attention(ch3) if use_attention else nn.Identity()
        self.mid_block2 = ResBlock(ch3, ch3, emb_dim)

        # =====================
        # Decoder
        # =====================
        # Level 3: 12 → 24, receives skip from enc3
        self.up3 = Upsample(ch3)  # 12 → 24
        self.dec3_block1 = ResBlock(ch3 + ch3, ch3, emb_dim)  # +skip from enc3_block2
        self.dec3_block2 = ResBlock(ch3 + ch3, ch3, emb_dim)  # +skip from enc3_block1

        # Level 2: 24 → 48, receives skip from enc2
        self.up2 = Upsample(ch3)  # 24 → 48
        self.dec2_block1 = ResBlock(ch3 + ch2, ch2, emb_dim)  # +skip from enc2_block2
        self.dec2_block2 = ResBlock(ch2 + ch2, ch2, emb_dim)  # +skip from enc2_block1

        # Level 1: 48 → 96, receives skip from enc1
        self.up1 = Upsample(ch2)  # 48 → 96
        self.dec1_block1 = ResBlock(ch2 + ch1, ch1, emb_dim)  # +skip from enc1_block2
        self.dec1_block2 = ResBlock(ch1 + ch1, ch1, emb_dim)  # +skip from enc1_block1

        # =====================
        # Output
        # =====================
        self.conv_out = nn.Sequential(
            nn.GroupNorm(get_groups(ch1), ch1),
            nn.SiLU(),
            nn.Conv2d(ch1, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

        # Print info
        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"FlowUNet: {params:.1f}M parameters")

    def forward(self, x, t):
        emb = self.time_emb(t)

        # =====================
        # Encoder - save outputs for skip connections
        # =====================
        h = self.conv_in(x)  # 96x96, ch1

        # Level 1
        s1_1 = self.enc1_block1(h, emb)  # 96x96, ch1
        s1_2 = self.enc1_block2(s1_1, emb)  # 96x96, ch1
        h = self.down1(s1_2)  # 48x48, ch1

        # Level 2
        s2_1 = self.enc2_block1(h, emb)  # 48x48, ch2
        s2_2 = self.enc2_block2(s2_1, emb)  # 48x48, ch2
        h = self.down2(s2_2)  # 24x24, ch2

        # Level 3
        s3_1 = self.enc3_block1(h, emb)  # 24x24, ch3
        s3_2 = self.enc3_block2(s3_1, emb)  # 24x24, ch3
        h = self.down3(s3_2)  # 12x12, ch3

        # =====================
        # Bottleneck
        # =====================
        h = self.mid_block1(h, emb)  # 12x12, ch3
        h = self.mid_attn(h)  # 12x12, ch3
        h = self.mid_block2(h, emb)  # 12x12, ch3

        # =====================
        # Decoder - use skip connections in reverse order
        # =====================
        # Level 3: 12 → 24
        h = self.up3(h)  # 24x24, ch3
        h = self.dec3_block1(torch.cat([h, s3_2], dim=1), emb)  # 24x24, ch3
        h = self.dec3_block2(torch.cat([h, s3_1], dim=1), emb)  # 24x24, ch3

        # Level 2: 24 → 48
        h = self.up2(h)  # 48x48, ch3
        h = self.dec2_block1(torch.cat([h, s2_2], dim=1), emb)  # 48x48, ch2
        h = self.dec2_block2(torch.cat([h, s2_1], dim=1), emb)  # 48x48, ch2

        # Level 1: 48 → 96
        h = self.up1(h)  # 96x96, ch2
        h = self.dec1_block1(torch.cat([h, s1_2], dim=1), emb)  # 96x96, ch1
        h = self.dec1_block2(torch.cat([h, s1_1], dim=1), emb)  # 96x96, ch1

        return self.conv_out(h)


if __name__ == "__main__":
    print("Testing FlowUNet...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FlowUNet(base_ch=64).to(device)

    x = torch.randn(2, 3, 96, 96, device=device)
    t = torch.rand(2, device=device)

    with torch.no_grad():
        y = model(x, t)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Output mean: {y.mean().item():.6f} (should be ~0)")
    print(f"Output std:  {y.std().item():.6f} (should be ~0)")
    print("✓ Test passed!")
