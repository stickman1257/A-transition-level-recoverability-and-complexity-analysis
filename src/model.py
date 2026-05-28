from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


AVAILABLE_RECONSTRUCTION_MODELS = (
    "vanilla_unet",
    "conditional_unet",
    "cnn_1d",
    "lstm",
    "gru",
)
MODEL_TYPE_ALIASES = {
    "unet": "vanilla_unet",
    "u_net": "vanilla_unet",
    "u-net": "vanilla_unet",
    "vanillaunet": "vanilla_unet",
    "vanilla_u_net": "vanilla_unet",
    "vanilla-u-net": "vanilla_unet",
    "conditional_u_net": "conditional_unet",
    "conditional-u-net": "conditional_unet",
    "conditionalunet": "conditional_unet",
    "1dcnn": "cnn_1d",
    "cnn1d": "cnn_1d",
    "1d-cnn": "cnn_1d",
    "1d_cnn": "cnn_1d",
}


@dataclass
class ReconstructionModelConfig:
    num_channels: int
    model_type: str = "conditional_unet"
    latent_dim: int = 64
    window_size: int = 1024
    base_channels: int = 32
    recurrent_layers: int = 1
    bidirectional: bool = True


def normalize_reconstruction_model_type(model_type: str) -> str:
    canonical = str(model_type).strip().lower().replace(" ", "_")
    canonical = MODEL_TYPE_ALIASES.get(canonical, canonical)
    if canonical not in AVAILABLE_RECONSTRUCTION_MODELS:
        available = ", ".join(AVAILABLE_RECONSTRUCTION_MODELS)
        raise ValueError(f"Unsupported model_type={model_type!r}. Available: {available}")
    return canonical


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDilatedBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock1D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ConditionalUNet(nn.Module):
    def __init__(self, config: ReconstructionModelConfig) -> None:
        super().__init__()
        self.config = config
        c = config.num_channels
        b = config.base_channels

        self.input_proj = nn.Conv1d(c, b, kernel_size=1)
        self.down1 = DownBlock1D(b, b)
        self.down2 = DownBlock1D(b, b * 2)
        self.down3 = DownBlock1D(b * 2, b * 4)
        self.bottleneck = ConvBlock1D(b * 4, b * 8)
        self.up3 = UpBlock1D(b * 8, b * 4, b * 4)
        self.up2 = UpBlock1D(b * 4, b * 2, b * 2)
        self.up1 = UpBlock1D(b * 2, b, b)
        self.output_head = nn.Conv1d(b, c, kernel_size=1)
        self.condition_skip = nn.Conv1d(c, c, kernel_size=1)
        self._init_condition_skip()

    def _init_condition_skip(self) -> None:
        with torch.no_grad():
            self.condition_skip.weight.zero_()
            self.condition_skip.bias.zero_()
            eye = torch.eye(self.config.num_channels, dtype=self.condition_skip.weight.dtype)
            self.condition_skip.weight[:, :, 0].copy_(eye)

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, None, None]:
        del target, deterministic
        x = self.input_proj(condition)
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        residual = self.output_head(x)
        residual = F.interpolate(residual, size=self.config.window_size, mode="linear", align_corners=False)
        skip = self.condition_skip(condition)
        recon = skip + residual
        return recon, None, None


class VanillaUNet(nn.Module):
    def __init__(self, config: ReconstructionModelConfig) -> None:
        super().__init__()
        self.config = config
        c = config.num_channels
        b = config.base_channels

        self.input_proj = nn.Conv1d(c, b, kernel_size=1)
        self.down1 = DownBlock1D(b, b)
        self.down2 = DownBlock1D(b, b * 2)
        self.down3 = DownBlock1D(b * 2, b * 4)
        self.bottleneck = ConvBlock1D(b * 4, b * 8)
        self.up3 = UpBlock1D(b * 8, b * 4, b * 4)
        self.up2 = UpBlock1D(b * 4, b * 2, b * 2)
        self.up1 = UpBlock1D(b * 2, b, b)
        self.output_head = nn.Conv1d(b, c, kernel_size=1)

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, None, None]:
        del target, deterministic
        x = self.input_proj(condition)
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        recon = self.output_head(x)
        recon = F.interpolate(recon, size=self.config.window_size, mode="linear", align_corners=False)
        return recon, None, None


class ConditionalCNN1D(nn.Module):
    def __init__(self, config: ReconstructionModelConfig) -> None:
        super().__init__()
        self.config = config
        c = config.num_channels
        b = config.base_channels

        self.input_proj = nn.Conv1d(c, b, kernel_size=1)
        self.trunk = nn.Sequential(
            ResidualDilatedBlock1D(b, dilation=1),
            ResidualDilatedBlock1D(b, dilation=2),
            ResidualDilatedBlock1D(b, dilation=4),
            ResidualDilatedBlock1D(b, dilation=8),
        )
        self.output_head = nn.Sequential(
            nn.Conv1d(b, b, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(b, c, kernel_size=1),
        )
        self.condition_skip = nn.Conv1d(c, c, kernel_size=1)
        self._init_condition_skip()

    def _init_condition_skip(self) -> None:
        with torch.no_grad():
            self.condition_skip.weight.zero_()
            self.condition_skip.bias.zero_()
            eye = torch.eye(self.config.num_channels, dtype=self.condition_skip.weight.dtype)
            self.condition_skip.weight[:, :, 0].copy_(eye)

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, None, None]:
        del target, deterministic
        x = self.input_proj(condition)
        x = self.trunk(x)
        residual = self.output_head(x)
        skip = self.condition_skip(condition)
        return skip + residual, None, None


class ConditionalRecurrentModel(nn.Module):
    def __init__(self, config: ReconstructionModelConfig, cell_type: str) -> None:
        super().__init__()
        self.config = config
        self.cell_type = cell_type

        c = config.num_channels
        hidden_dim = config.latent_dim
        input_dim = max(config.base_channels, c)
        directions = 2 if config.bidirectional else 1
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU

        self.input_proj = nn.Linear(c, input_dim)
        self.recurrent = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=config.recurrent_layers,
            batch_first=True,
            dropout=0.1 if config.recurrent_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * directions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c),
        )
        self.condition_skip = nn.Conv1d(c, c, kernel_size=1)
        self._init_condition_skip()

    def _init_condition_skip(self) -> None:
        with torch.no_grad():
            self.condition_skip.weight.zero_()
            self.condition_skip.bias.zero_()
            eye = torch.eye(self.config.num_channels, dtype=self.condition_skip.weight.dtype)
            self.condition_skip.weight[:, :, 0].copy_(eye)

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, None, None]:
        del target, deterministic
        x = condition.transpose(1, 2)
        x = self.input_proj(x)
        x, _ = self.recurrent(x)
        residual = self.output_proj(x).transpose(1, 2)
        skip = self.condition_skip(condition)
        return skip + residual, None, None


class ConditionalLSTM(ConditionalRecurrentModel):
    def __init__(self, config: ReconstructionModelConfig) -> None:
        super().__init__(config=config, cell_type="lstm")


class ConditionalGRU(ConditionalRecurrentModel):
    def __init__(self, config: ReconstructionModelConfig) -> None:
        super().__init__(config=config, cell_type="gru")


def build_reconstruction_model(config: ReconstructionModelConfig) -> nn.Module:
    model_type = normalize_reconstruction_model_type(config.model_type)
    config.model_type = model_type

    if model_type == "vanilla_unet":
        return VanillaUNet(config)
    if model_type == "conditional_unet":
        return ConditionalUNet(config)
    if model_type == "cnn_1d":
        return ConditionalCNN1D(config)
    if model_type == "lstm":
        return ConditionalLSTM(config)
    if model_type == "gru":
        return ConditionalGRU(config)

    raise ValueError(f"Unsupported model_type={model_type!r}")


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.0,
    corr_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse_loss = F.mse_loss(recon, target, reduction="mean")
    l1_loss = F.l1_loss(recon, target, reduction="mean")
    corr_loss = _correlation_loss(recon, target)
    recon_loss = mse_loss + l1_weight * l1_loss + corr_weight * corr_loss
    return recon_loss, {
        "loss": float(recon_loss.item()),
        "recon_loss": float(recon_loss.item()),
        "mse_loss": float(mse_loss.item()),
        "l1_loss": float(l1_loss.item()),
        "corr_loss": float(corr_loss.item()),
    }


def _correlation_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    recon_centered = recon - recon.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    numerator = (recon_centered * target_centered).mean(dim=-1)
    denominator = recon_centered.pow(2).mean(dim=-1).sqrt() * target_centered.pow(2).mean(dim=-1).sqrt()
    corr = numerator / (denominator + eps)
    return 1.0 - corr.mean()


# Compatibility aliases kept so existing imports/checkpoints don't need a wider refactor.
CVAEConfig = ReconstructionModelConfig
ConditionalVAE = ConditionalUNet
cvae_loss = reconstruction_loss
