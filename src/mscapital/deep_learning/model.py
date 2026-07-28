from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SequenceModelConfig:
    market_features: int
    transaction_features: int
    transaction_grid_features: int
    market_steps: int = 212
    event_steps: int = 256
    grid_steps: int = 60
    hidden_size: int = 128
    attention_layers: int = 4
    attention_heads: int = 4
    dropout: float = 0.10
    gradient_checkpointing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualTCNBlock(nn.Module):
    def __init__(self, hidden_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.norm1 = ChannelLayerNorm(hidden_size)
        self.conv1 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, dilation=dilation, padding=padding
        )
        self.norm2 = ChannelLayerNorm(hidden_size)
        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, dilation=dilation, padding=padding
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = values
        values = self.conv1(self.dropout(self.activation(self.norm1(values))))
        values = self.conv2(self.dropout(self.activation(self.norm2(values))))
        values = residual + values
        if mask is not None:
            values = values * mask.unsqueeze(1).to(values.dtype)
        return values


class ChannelLayerNorm(nn.Module):
    """Layer-normalize channels without mixing real steps with left padding."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class TCNStack(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_size: int,
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_features, hidden_size)
        self.blocks = nn.ModuleList(
            ResidualTCNBlock(hidden_size, dilation, dropout) for dilation in dilations
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = self.input_projection(values)
        if mask is not None:
            values = values * mask.unsqueeze(-1).to(values.dtype)
        values = values.transpose(1, 2)
        for block in self.blocks:
            values = block(values, mask)
        return self.output_norm(values.transpose(1, 2))


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        original_valid = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~original_valid, 0] = True
        logits = self.score(values).squeeze(-1).masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        return pooled * original_valid.unsqueeze(-1).to(pooled.dtype)


def _last_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask, -1).max(dim=1).values.clamp_min(0)
    result = values[torch.arange(values.shape[0], device=values.device), indices]
    return result * mask.any(dim=1).unsqueeze(-1).to(result.dtype)


class MarketEncoder(nn.Module):
    def __init__(self, input_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.tcn = TCNStack(input_features, hidden_size, (1, 2, 4, 8, 16), dropout)
        self.pool = MaskedAttentionPool(hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(values, mask)
        return self.output(
            torch.cat((self.pool(encoded, mask), _last_valid(encoded, mask)), -1)
        )


class TransactionEventEncoder(nn.Module):
    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.value_projection = nn.Linear(config.transaction_features, hidden)
        self.side_embedding = nn.Embedding(3, hidden, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.event_steps, hidden)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.input_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.attention_heads,
                dim_feedforward=hidden * 3,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.attention_layers)
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.pool = MaskedAttentionPool(hidden)
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
        )
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self, values: torch.Tensor, side: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.value_projection(values) + self.side_embedding(side)
        encoded = self.input_norm(
            encoded + self.position_embedding[:, : values.shape[1]]
        )
        encoded = self.dropout(encoded) * mask.unsqueeze(-1).to(encoded.dtype)
        padding_mask = ~mask
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                encoded = checkpoint(
                    lambda current, module=layer: module(
                        current, src_key_padding_mask=padding_mask
                    ),
                    encoded,
                    use_reentrant=False,
                )
            else:
                encoded = layer(encoded, src_key_padding_mask=padding_mask)
            encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        encoded = self.final_norm(encoded)
        return self.output(
            torch.cat((self.pool(encoded, mask), _last_valid(encoded, mask)), -1)
        )


class TransactionGridEncoder(nn.Module):
    def __init__(self, input_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.tcn = TCNStack(input_features, hidden_size, (1, 2, 4, 8), dropout)
        self.pool = MaskedAttentionPool(hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        encoded = self.tcn(values)
        return self.output(torch.cat((self.pool(encoded, mask), encoded[:, -1]), -1))


class MSCapitalSequenceModel(nn.Module):
    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.market = MarketEncoder(config.market_features, hidden, config.dropout)
        self.transaction_event = TransactionEventEncoder(config)
        self.transaction_grid = TransactionGridEncoder(
            config.transaction_grid_features, hidden, config.dropout
        )
        self.transaction_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
        )
        self.gate = nn.Linear(hidden * 2, hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        market_values: torch.Tensor,
        market_mask: torch.Tensor,
        transaction_values: torch.Tensor,
        transaction_side: torch.Tensor,
        transaction_mask: torch.Tensor,
        transaction_grid: torch.Tensor,
    ) -> torch.Tensor:
        market = self.market(market_values, market_mask)
        event = self.transaction_event(
            transaction_values, transaction_side, transaction_mask
        )
        grid = self.transaction_grid(transaction_grid)
        transaction = self.transaction_fusion(torch.cat((event, grid), dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat((market, transaction), dim=-1)))
        fused = market + gate * transaction
        return self.head(fused).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def competition_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse = torch.mean(torch.square(prediction - target))
    cosine_penalty = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    if cosine_weight > 0 and prediction.numel() > 1:
        denominator = torch.linalg.vector_norm(prediction) * torch.linalg.vector_norm(
            target
        )
        cosine = torch.sum(prediction * target) / denominator.clamp_min(1e-12)
        target_variance = torch.var(target, unbiased=False).detach()
        cosine_penalty = target_variance * (1.0 - cosine)
    loss = mse + cosine_weight * cosine_penalty
    return loss, {"mse": mse.detach(), "cosine_penalty": cosine_penalty.detach()}
