from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tepa_eval.conditions import DIRECTION_BUCKETS, MAGNITUDE_BUCKETS
from tepa_eval.trajectory import trajectory_to_heatmap_logits

PREDICTION_MODEL_NAMES = (
    "tepa",
    "tepa_latent",
    "tepa_latent_vit",
    "context_memory_tepa",
    "monolithic",
    "fused_transformer",
    "fused_latent_transformer",
    "stuffed_image",
)
TARGET_MODEL_NAME = "target_autoencoder"
CONDITION_MODEL_NAME = "condition_semantics"
SUPPORTED_MODEL_NAMES = PREDICTION_MODEL_NAMES + (TARGET_MODEL_NAME, CONDITION_MODEL_NAME)


class ContextEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ViTContextEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        image_size: int = 64,
        patch_size: int = 8,
        token_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}.")
        resolved_token_dim = token_dim or max(96, latent_dim)
        num_heads = 4 if resolved_token_dim % 4 == 0 else 3
        patch_count = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3,
            resolved_token_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, resolved_token_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, resolved_token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resolved_token_dim,
            nhead=num_heads,
            dim_feedforward=resolved_token_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        pooled_dim = resolved_token_dim * 3
        self.pooler = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embedding(image).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        encoded_patches = encoded[:, 1:]
        mean_pool = encoded_patches.mean(dim=1)
        max_pool = encoded_patches.max(dim=1).values
        return self.pooler(torch.cat([encoded[:, 0], mean_pool, max_pool], dim=-1))


class ContextMemoryEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        image_size: int = 64,
        patch_size: int = 8,
        token_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}.")
        resolved_token_dim = token_dim or max(96, latent_dim)
        num_heads = 4 if resolved_token_dim % 4 == 0 else 3
        patch_count = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3,
            resolved_token_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, resolved_token_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, resolved_token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resolved_token_dim,
            nhead=num_heads,
            dim_feedforward=resolved_token_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = (
            nn.Identity()
            if resolved_token_dim == latent_dim
            else nn.Sequential(
                nn.LayerNorm(resolved_token_dim),
                nn.Linear(resolved_token_dim, latent_dim),
            )
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embedding(image).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        return self.output_projection(self.encoder(tokens))


class TextConditionEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        max_text_len: int = 192,
        token_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        resolved_token_dim = token_dim or max(64, latent_dim // 2)
        num_heads = 4 if resolved_token_dim % 4 == 0 else 2
        self.max_text_len = max_text_len
        self.embedding = nn.Embedding(257, resolved_token_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, resolved_token_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_text_len + 1, resolved_token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resolved_token_dim,
            nhead=num_heads,
            dim_feedforward=resolved_token_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        pooled_dim = resolved_token_dim * 3
        self.pooler = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_text_len:
            raise ValueError(f"TextConditionEncoder received {tokens.shape[1]} tokens, max is {self.max_text_len}.")
        batch_size = tokens.shape[0]
        embedded = self.embedding(tokens)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, embedded], dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.shape[1]]
        padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
                tokens == 0,
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        encoded_tokens = encoded[:, 1:]
        token_mask = (tokens != 0).unsqueeze(-1)
        masked_tokens = encoded_tokens * token_mask
        mean_pool = masked_tokens.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        max_candidates = encoded_tokens.masked_fill(~token_mask, -1.0e4)
        max_pool = max_candidates.max(dim=1).values
        has_tokens = token_mask.any(dim=1).expand_as(max_pool)
        max_pool = torch.where(has_tokens, max_pool, torch.zeros_like(max_pool))
        return self.pooler(torch.cat([encoded[:, 0], mean_pool, max_pool], dim=-1))


class PredictionHeads(nn.Module):
    def __init__(self, latent_dim: int, final_dim: int, heatmap_dim: int) -> None:
        super().__init__()
        self.final_pos = nn.Linear(latent_dim, final_dim)
        self.heatmap = nn.Linear(latent_dim, heatmap_dim)
        self.wall_contact = nn.Linear(latent_dim, 1)
        self.time_to_contact = nn.Linear(latent_dim, 1)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "final_pos": self.final_pos(latent),
            "heatmap": self.heatmap(latent),
            "wall_contact": self.wall_contact(latent),
            "time_to_contact": self.time_to_contact(latent),
        }


class ConditionSemanticsModel(nn.Module):
    def __init__(self, latent_dim: int, max_pucks: int, max_text_len: int = 192) -> None:
        super().__init__()
        self.condition_encoder = TextConditionEncoder(latent_dim, max_text_len=max_text_len)
        hidden_dim = max(latent_dim * 2, 128)
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.object_logits = nn.Linear(hidden_dim, max_pucks)
        self.impulse = nn.Linear(hidden_dim, 2)
        self.horizon = nn.Linear(hidden_dim, 1)
        self.direction_logits = nn.Linear(hidden_dim, len(DIRECTION_BUCKETS))
        self.magnitude_logits = nn.Linear(hidden_dim, len(MAGNITUDE_BUCKETS))

    def forward(self, text_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        z_condition = self.condition_encoder(text_tokens)
        decoded = self.decoder(z_condition)
        return {
            "z_condition": z_condition,
            "object_logits": self.object_logits(decoded),
            "impulse": self.impulse(decoded),
            "horizon": self.horizon(decoded),
            "direction_logits": self.direction_logits(decoded),
            "magnitude_logits": self.magnitude_logits(decoded),
        }

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward(batch["text_tokens"])


class TargetEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        final_dim: int,
        trajectory_dim: int,
        heatmap_size: int,
        horizon: int,
        normalize_latents: bool = False,
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.trajectory_dim = trajectory_dim
        self.horizon_scale = float(horizon + 1)
        self.normalize_latents = normalize_latents
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(final_dim + 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.fuser = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        final_pos: torch.Tensor,
        trajectory: torch.Tensor,
        heatmap: torch.Tensor,
        wall_contact: torch.Tensor,
        time_to_contact: torch.Tensor,
    ) -> torch.Tensor:
        heatmap_image = heatmap.reshape(-1, 1, self.heatmap_size, self.heatmap_size)
        heatmap_latent = self.heatmap_encoder(heatmap_image)
        trajectory_latent = self.trajectory_encoder(trajectory.reshape(-1, self.trajectory_dim))
        scalar_features = torch.cat(
            [
                final_pos,
                wall_contact,
                time_to_contact / self.horizon_scale,
            ],
            dim=-1,
        )
        scalar_latent = self.scalar_encoder(scalar_features)
        latent = self.fuser(torch.cat([heatmap_latent, trajectory_latent, scalar_latent], dim=-1))
        return F.normalize(latent, dim=-1) if self.normalize_latents else latent


class TargetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        final_dim: int,
        trajectory_dim: int,
        heatmap_size: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        if final_dim % 2 != 0:
            raise ValueError("Final position dimension must contain x/y pairs.")
        if trajectory_dim != horizon * final_dim:
            raise ValueError("Trajectory dimension must equal horizon * final_dim.")
        self.max_pucks = final_dim // 2
        self.horizon = horizon
        hidden_dim = max(latent_dim * 2, 128)
        self.features = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.start_position = nn.Linear(hidden_dim, final_dim)
        self.step_delta = nn.Linear(hidden_dim, max(horizon - 1, 0) * final_dim)
        self.final_pos = nn.Linear(hidden_dim, final_dim)
        self.wall_contact = nn.Linear(hidden_dim, 1)
        self.time_to_contact = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(latent)
        trajectory = self.decode_trajectory(features)
        return {
            "final_pos": self.final_pos(features),
            "trajectory": trajectory,
            "heatmap": trajectory_to_heatmap_logits(
                trajectory,
                heatmap_size=self.heatmap_size,
                max_pucks=self.max_pucks,
            ),
            "wall_contact": self.wall_contact(features),
            "time_to_contact": self.time_to_contact(features),
        }

    def decode_trajectory(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        start = torch.sigmoid(self.start_position(features)).reshape(batch_size, 1, self.max_pucks, 2)
        if self.horizon == 1:
            return start.flatten(start_dim=1)

        raw_delta = self.step_delta(features).reshape(batch_size, self.horizon - 1, self.max_pucks, 2)
        step_scale = 1.25 / float(max(self.horizon - 1, 1))
        deltas = torch.tanh(raw_delta) * step_scale
        trajectory = torch.cat([start, start + torch.cumsum(deltas, dim=1)], dim=1)
        return trajectory.flatten(start_dim=1)


class TargetAutoencoderModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        final_dim: int,
        trajectory_dim: int,
        heatmap_size: int,
        horizon: int,
        normalize_latents: bool = False,
    ) -> None:
        super().__init__()
        self.target_encoder = TargetEncoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            normalize_latents=normalize_latents,
        )
        self.target_decoder = TargetDecoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
        )

    def forward(
        self,
        final_pos: torch.Tensor,
        trajectory: torch.Tensor,
        heatmap: torch.Tensor,
        wall_contact: torch.Tensor,
        time_to_contact: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        z_target = self.target_encoder(final_pos, trajectory, heatmap, wall_contact, time_to_contact)
        return {
            "z_target": z_target,
            "target_reconstruction": self.target_decoder(z_target),
        }

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return self.forward(
            batch["target_final_pos"],
            batch["target_traj"],
            batch["target_heatmap"],
            batch["target_wall_contact"],
            batch["target_ttc"],
        )


class TEPAModel(nn.Module):
    def __init__(self, latent_dim: int, final_dim: int, heatmap_dim: int, max_text_len: int = 192) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(latent_dim)
        self.condition_encoder = TextConditionEncoder(latent_dim, max_text_len=max_text_len)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
        )
        self.heads = PredictionHeads(latent_dim, final_dim, heatmap_dim)

    def forward(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        condition_image: torch.Tensor | None = None,
        z_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_context = self.context_encoder(context_image)
        if z_condition is None:
            if text_tokens is None:
                raise ValueError("TEPAModel requires text_tokens unless z_condition is provided.")
            z_condition = self.condition_encoder(text_tokens)
        z_hat_target = self.predictor(torch.cat([z_context, z_condition], dim=-1))
        return self.heads(z_hat_target)

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward(
            batch["context_image"],
            batch.get("text_tokens"),
            batch.get("condition_image"),
            batch.get("z_condition"),
        )


class LatentTEPAModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        final_dim: int,
        heatmap_dim: int,
        heatmap_size: int,
        horizon: int,
        trajectory_dim: int,
        max_text_len: int = 192,
        normalize_latents: bool = False,
        context_encoder: nn.Module | None = None,
        context_state_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.normalize_latents = normalize_latents
        self.include_target_reconstruction = True
        self.context_encoder = context_encoder or ContextEncoder(latent_dim)
        self.condition_encoder = TextConditionEncoder(latent_dim, max_text_len=max_text_len)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.context_state_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, context_state_dim),
            )
            if context_state_dim is not None
            else None
        )
        self.target_encoder = TargetEncoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            normalize_latents=normalize_latents,
        )
        self.target_decoder = TargetDecoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
        )

    def predict_latent(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        z_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_context, z_condition = self.encode_context_condition(context_image, text_tokens, z_condition)
        latent = self.predictor(torch.cat([z_context, z_condition], dim=-1))
        return F.normalize(latent, dim=-1) if self.normalize_latents else latent

    def encode_context_condition(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        z_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_context = self.context_encoder(context_image)
        if z_condition is None:
            if text_tokens is None:
                raise ValueError("LatentTEPAModel requires text_tokens unless z_condition is provided.")
            z_condition = self.condition_encoder(text_tokens)
        return z_context, z_condition

    def forward(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        condition_image: torch.Tensor | None = None,
        z_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_context, z_condition = self.encode_context_condition(context_image, text_tokens, z_condition)
        z_hat_target = self.predictor(torch.cat([z_context, z_condition], dim=-1))
        if self.normalize_latents:
            z_hat_target = F.normalize(z_hat_target, dim=-1)
        predictions = self.target_decoder(z_hat_target)
        predictions["z_hat_target"] = z_hat_target
        if self.context_state_head is not None:
            predictions["context_state"] = self.context_state_head(z_context)
        return predictions

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        predictions = self.forward(
            batch["context_image"],
            batch.get("text_tokens"),
            batch.get("condition_image"),
            batch.get("z_condition"),
        )
        z_target = batch.get("z_target")
        if not isinstance(z_target, torch.Tensor):
            z_target = self.target_encoder(
                batch["target_final_pos"],
                batch["target_traj"],
                batch["target_heatmap"],
                batch["target_wall_contact"],
                batch["target_ttc"],
            )
        predictions["z_target"] = z_target
        if self.include_target_reconstruction:
            predictions["target_reconstruction"] = self.target_decoder(z_target)
        return predictions


class ContextMemoryTEPAModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        final_dim: int,
        heatmap_size: int,
        horizon: int,
        trajectory_dim: int,
        image_size: int,
        max_text_len: int = 192,
        normalize_latents: bool = False,
        context_state_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.normalize_latents = normalize_latents
        self.include_target_reconstruction = True
        self.context_memory_encoder = ContextMemoryEncoder(latent_dim=latent_dim, image_size=image_size)
        self.condition_encoder = TextConditionEncoder(latent_dim, max_text_len=max_text_len)
        num_heads = 4 if latent_dim % 4 == 0 else 2
        self.context_memory_norm = nn.LayerNorm(latent_dim)
        self.condition_query_norm = nn.LayerNorm(latent_dim)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=0.05,
            batch_first=True,
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(latent_dim * 3),
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.context_state_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, context_state_dim),
            )
            if context_state_dim is not None
            else None
        )
        self.target_encoder = TargetEncoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            normalize_latents=normalize_latents,
        )
        self.target_decoder = TargetDecoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
        )

    def encode_context_memory(self, context_image: torch.Tensor) -> torch.Tensor:
        return self.context_memory_encoder(context_image)

    def predict_from_memory(
        self,
        context_memory: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        z_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_condition is None:
            if text_tokens is None:
                raise ValueError("ContextMemoryTEPAModel requires text_tokens unless z_condition is provided.")
            z_condition = self.condition_encoder(text_tokens)
        context_memory = self._expand_context_memory(context_memory, z_condition.shape[0])
        memory = self.context_memory_norm(context_memory)
        query = self.condition_query_norm(z_condition).unsqueeze(1)
        attended, _ = self.memory_attention(query, memory, memory, need_weights=False)
        context_summary = memory.mean(dim=1)
        z_hat_target = self.predictor(torch.cat([attended.squeeze(1), z_condition, context_summary], dim=-1))
        return F.normalize(z_hat_target, dim=-1) if self.normalize_latents else z_hat_target

    def forward(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        condition_image: torch.Tensor | None = None,
        z_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_memory = self.encode_context_memory(context_image)
        z_hat_target = self.predict_from_memory(context_memory, text_tokens=text_tokens, z_condition=z_condition)
        predictions = self.target_decoder(z_hat_target)
        predictions["z_hat_target"] = z_hat_target
        if self.context_state_head is not None:
            memory = self._expand_context_memory(context_memory, z_hat_target.shape[0])
            predictions["context_state"] = self.context_state_head(self.context_memory_norm(memory).mean(dim=1))
        return predictions

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        predictions = self.forward(
            batch["context_image"],
            batch.get("text_tokens"),
            batch.get("condition_image"),
            batch.get("z_condition"),
        )
        z_target = batch.get("z_target")
        if not isinstance(z_target, torch.Tensor):
            z_target = self.target_encoder(
                batch["target_final_pos"],
                batch["target_traj"],
                batch["target_heatmap"],
                batch["target_wall_contact"],
                batch["target_ttc"],
            )
        predictions["z_target"] = z_target
        if self.include_target_reconstruction:
            predictions["target_reconstruction"] = self.target_decoder(z_target)
        return predictions

    def _expand_context_memory(self, context_memory: torch.Tensor, batch_size: int) -> torch.Tensor:
        if context_memory.shape[0] == batch_size:
            return context_memory
        if context_memory.shape[0] == 1:
            return context_memory.expand(batch_size, -1, -1)
        raise ValueError(
            f"Context memory batch {context_memory.shape[0]} cannot be broadcast to condition batch {batch_size}."
        )


class StuffedImageModel(nn.Module):
    def __init__(self, latent_dim: int, final_dim: int, heatmap_dim: int) -> None:
        super().__init__()
        self.stuffed_encoder = ContextEncoder(latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.heads = PredictionHeads(latent_dim, final_dim, heatmap_dim)

    def forward(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor,
        condition_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if condition_image is None:
            raise ValueError("StuffedImageModel requires a rendered condition_image.")
        stuffed_context = torch.cat([context_image, condition_image], dim=-1)
        z_context_condition = self.stuffed_encoder(stuffed_context)
        z_hat_target = self.predictor(z_context_condition)
        return self.heads(z_hat_target)


class FusedTransformerModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        final_dim: int,
        heatmap_dim: int,
        image_size: int,
        max_text_len: int,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        token_dim = max(32, latent_dim // 2)
        num_heads = 4 if token_dim % 4 == 0 else 2
        image_tokens_per_side = image_size // patch_size
        image_token_count = image_tokens_per_side * image_tokens_per_side

        self.image_patch = nn.Conv2d(3, token_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Embedding(257, token_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1 + image_token_count + max_text_len, token_dim)
        )
        self.type_embedding = nn.Embedding(3, token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.fused_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.pooler = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.heads = PredictionHeads(latent_dim, final_dim, heatmap_dim)

    def forward(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor,
        condition_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = context_image.shape[0]
        image_tokens = self.image_patch(context_image).flatten(2).transpose(1, 2)
        text_tokens_embedded = self.text_embedding(text_tokens)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        fused_tokens = torch.cat([cls_tokens, image_tokens, text_tokens_embedded], dim=1)

        type_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=context_image.device),
                torch.ones((batch_size, image_tokens.shape[1]), dtype=torch.long, device=context_image.device),
                torch.full(
                    (batch_size, text_tokens.shape[1]),
                    2,
                    dtype=torch.long,
                    device=context_image.device,
                ),
            ],
            dim=1,
        )
        fused_tokens = fused_tokens + self.position_embedding[:, : fused_tokens.shape[1]] + self.type_embedding(type_ids)

        padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1 + image_tokens.shape[1]), dtype=torch.bool, device=context_image.device),
                text_tokens == 0,
            ],
            dim=1,
        )
        fused_latent = self.fused_encoder(fused_tokens, src_key_padding_mask=padding_mask)[:, 0]
        z_context_condition = self.pooler(fused_latent)
        return self.heads(z_context_condition)


class FusedLatentTransformerModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        final_dim: int,
        heatmap_size: int,
        horizon: int,
        trajectory_dim: int,
        image_size: int,
        max_text_len: int,
        patch_size: int = 8,
        normalize_latents: bool = False,
        context_state_dim: int | None = None,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}.")
        self.normalize_latents = normalize_latents
        self.include_target_reconstruction = True
        token_dim = max(96, latent_dim)
        num_heads = 4 if token_dim % 4 == 0 else 3
        image_token_count = (image_size // patch_size) ** 2

        self.image_patch = nn.Conv2d(3, token_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Embedding(257, token_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + image_token_count + max_text_len, token_dim))
        self.type_embedding = nn.Embedding(3, token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fused_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
            enable_nested_tensor=False,
        )
        self.pooler = nn.Sequential(
            nn.LayerNorm(token_dim * 4),
            nn.Linear(token_dim * 4, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.context_state_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, context_state_dim),
            )
            if context_state_dim is not None
            else None
        )
        self.target_encoder = TargetEncoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            normalize_latents=normalize_latents,
        )
        self.target_decoder = TargetDecoder(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def encode_fused(self, context_image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = context_image.shape[0]
        image_tokens = self.image_patch(context_image).flatten(2).transpose(1, 2)
        text_tokens_embedded = self.text_embedding(text_tokens)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        fused_tokens = torch.cat([cls_tokens, image_tokens, text_tokens_embedded], dim=1)

        type_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=context_image.device),
                torch.ones((batch_size, image_tokens.shape[1]), dtype=torch.long, device=context_image.device),
                torch.full(
                    (batch_size, text_tokens.shape[1]),
                    2,
                    dtype=torch.long,
                    device=context_image.device,
                ),
            ],
            dim=1,
        )
        fused_tokens = fused_tokens + self.position_embedding[:, : fused_tokens.shape[1]] + self.type_embedding(type_ids)

        padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1 + image_tokens.shape[1]), dtype=torch.bool, device=context_image.device),
                text_tokens == 0,
            ],
            dim=1,
        )
        encoded = self.fused_encoder(fused_tokens, src_key_padding_mask=padding_mask)
        image_encoded = encoded[:, 1 : 1 + image_tokens.shape[1]]
        text_encoded = encoded[:, 1 + image_tokens.shape[1] :]
        text_mask = (text_tokens != 0).unsqueeze(-1)
        masked_text = text_encoded * text_mask
        text_mean = masked_text.sum(dim=1) / text_mask.sum(dim=1).clamp_min(1)
        text_max_candidates = text_encoded.masked_fill(~text_mask, -1.0e4)
        text_max = text_max_candidates.max(dim=1).values
        has_text = text_mask.any(dim=1).expand_as(text_max)
        text_max = torch.where(has_text, text_max, torch.zeros_like(text_max))
        pooled = torch.cat([encoded[:, 0], image_encoded.mean(dim=1), text_mean, text_max], dim=-1)
        return self.pooler(pooled)

    def forward(
        self,
        context_image: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        condition_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if text_tokens is None:
            raise ValueError("FusedLatentTransformerModel requires text_tokens.")
        z_context_condition = self.encode_fused(context_image, text_tokens)
        z_hat_target = self.predictor(z_context_condition)
        if self.normalize_latents:
            z_hat_target = F.normalize(z_hat_target, dim=-1)
        predictions = self.target_decoder(z_hat_target)
        predictions["z_hat_target"] = z_hat_target
        if self.context_state_head is not None:
            predictions["context_state"] = self.context_state_head(z_context_condition)
        return predictions

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        predictions = self.forward(
            batch["context_image"],
            batch.get("text_tokens"),
            batch.get("condition_image"),
        )
        z_target = batch.get("z_target")
        if not isinstance(z_target, torch.Tensor):
            z_target = self.target_encoder(
                batch["target_final_pos"],
                batch["target_traj"],
                batch["target_heatmap"],
                batch["target_wall_contact"],
                batch["target_ttc"],
            )
        predictions["z_target"] = z_target
        if self.include_target_reconstruction:
            predictions["target_reconstruction"] = self.target_decoder(z_target)
        return predictions


def build_model(
    model_name: str,
    latent_dim: int,
    max_pucks: int,
    heatmap_size: int,
    image_size: int = 64,
    max_text_len: int = 192,
    horizon: int = 40,
    normalize_latents: bool = False,
    context_state_dim: int | None = None,
) -> nn.Module:
    final_dim = max_pucks * 2
    heatmap_dim = heatmap_size * heatmap_size
    trajectory_dim = horizon * final_dim
    if model_name == "tepa":
        return TEPAModel(
            latent_dim=latent_dim,
            final_dim=final_dim,
            heatmap_dim=heatmap_dim,
            max_text_len=max_text_len,
        )
    if model_name == "tepa_latent":
        return LatentTEPAModel(
            latent_dim=latent_dim,
            final_dim=final_dim,
            heatmap_dim=heatmap_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            trajectory_dim=trajectory_dim,
            max_text_len=max_text_len,
            normalize_latents=normalize_latents,
            context_state_dim=context_state_dim,
        )
    if model_name == "tepa_latent_vit":
        return LatentTEPAModel(
            latent_dim=latent_dim,
            final_dim=final_dim,
            heatmap_dim=heatmap_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            trajectory_dim=trajectory_dim,
            max_text_len=max_text_len,
            normalize_latents=normalize_latents,
            context_encoder=ViTContextEncoder(latent_dim=latent_dim, image_size=image_size),
            context_state_dim=context_state_dim,
        )
    if model_name == "context_memory_tepa":
        return ContextMemoryTEPAModel(
            latent_dim=latent_dim,
            final_dim=final_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            trajectory_dim=trajectory_dim,
            image_size=image_size,
            max_text_len=max_text_len,
            normalize_latents=normalize_latents,
            context_state_dim=context_state_dim,
        )
    if model_name == TARGET_MODEL_NAME:
        return TargetAutoencoderModel(
            latent_dim=latent_dim,
            final_dim=final_dim,
            trajectory_dim=trajectory_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            normalize_latents=normalize_latents,
        )
    if model_name == CONDITION_MODEL_NAME:
        return ConditionSemanticsModel(latent_dim=latent_dim, max_pucks=max_pucks, max_text_len=max_text_len)
    if model_name in {"monolithic", "fused_transformer"}:
        return FusedTransformerModel(
            latent_dim=latent_dim,
            final_dim=final_dim,
            heatmap_dim=heatmap_dim,
            image_size=image_size,
            max_text_len=max_text_len,
        )
    if model_name == "fused_latent_transformer":
        return FusedLatentTransformerModel(
            latent_dim=latent_dim,
            final_dim=final_dim,
            heatmap_size=heatmap_size,
            horizon=horizon,
            trajectory_dim=trajectory_dim,
            image_size=image_size,
            max_text_len=max_text_len,
            normalize_latents=normalize_latents,
            context_state_dim=context_state_dim,
        )
    if model_name == "stuffed_image":
        return StuffedImageModel(latent_dim=latent_dim, final_dim=final_dim, heatmap_dim=heatmap_dim)
    raise ValueError(f"Unknown model: {model_name}")
