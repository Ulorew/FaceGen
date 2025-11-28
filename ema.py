import torch


class EMA:
    """
    Exponential Moving Average for model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)

        for batch in dataloader:
            # Training step
            loss.backward()
            optimizer.step()

            # Update EMA after each step
            ema.update()

        # For evaluation/sampling, use EMA weights
        ema.apply_shadow()
        samples = sample(model, ...)
        ema.restore()  # Restore training weights
    """

    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        """
        Args:
            model: the model to track
            decay: EMA decay rate (higher = slower updates)
            warmup_steps: linearly increase decay from 0 to target during warmup
        """
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Create shadow parameters (deep copy of initial weights)
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def get_current_decay(self):
        """Warmup: start with low decay, increase to target."""
        if self.step_count < self.warmup_steps:
            # Linear warmup from 0 to self.decay
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        return self.decay

    @torch.no_grad()
    def update(self):
        """Update shadow parameters. Call after each optimizer step."""
        decay = self.get_current_decay()

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # θ_ema = decay * θ_ema + (1 - decay) * θ
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

        self.step_count += 1

    @torch.no_grad()
    def apply_shadow(self):
        """Replace model params with EMA params. Call before evaluation."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        """Restore original params after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """For saving checkpoints."""
        return {
            'shadow': self.shadow,
            'step_count': self.step_count,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict):
        """For loading checkpoints."""
        self.shadow = state_dict['shadow']
        self.step_count = state_dict['step_count']
        self.decay = state_dict['decay']

