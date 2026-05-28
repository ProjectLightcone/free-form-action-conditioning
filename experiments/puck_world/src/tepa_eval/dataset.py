from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tepa_eval.conditions import DIRECTION_TO_INDEX, MAGNITUDE_TO_INDEX
from tepa_eval.io import read_jsonl
from tepa_eval.trajectory import trajectory_to_soft_heatmap

PAD_TOKEN = 0
_SOFT_HEATMAP_CACHE: dict[tuple[str, int, int, int], np.ndarray] = {}


class PuckDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        max_text_len: int = 192,
        include_condition_image: bool = True,
        target_latents_by_event: dict[int, np.ndarray] | None = None,
        condition_latents_by_condition: dict[int, np.ndarray] | None = None,
        condition_family_filter: set[str] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.max_text_len = max_text_len
        self.include_condition_image = include_condition_image
        self.target_latents_by_event = target_latents_by_event
        self.condition_latents_by_condition = condition_latents_by_condition
        self.conditions = read_jsonl(self.dataset_dir / "manifests" / f"conditions_{split}.jsonl")
        if condition_family_filter is not None:
            self.conditions = [row for row in self.conditions if str(row["family"]) in condition_family_filter]

        scenes = np.load(self.dataset_dir / "arrays" / "scenes" / "shard_00000.npz")
        outcomes_path = self.dataset_dir / "arrays" / "outcomes" / "shard_00000.npz"
        outcomes = np.load(outcomes_path)
        conditions = np.load(self.dataset_dir / "arrays" / "conditions" / "shard_00000.npz")

        self.scene_index = {int(scene_id): idx for idx, scene_id in enumerate(scenes["scene_id"])}
        self.events_by_id = _event_records(self.dataset_dir)
        self.event_to_scene = {event_id: int(row["scene_id"]) for event_id, row in self.events_by_id.items()}
        self.outcome_index = {int(event_id): idx for idx, event_id in enumerate(outcomes["event_id"])}
        self.condition_index = {
            int(condition_id): idx for idx, condition_id in enumerate(conditions["condition_id"])
        }

        self.context_image = scenes["context_image"]
        self.context_state = scenes["context_state"]
        self.condition_params = conditions["condition_params"]
        self.target_final_pos = outcomes["target_final_pos"]
        self.target_traj = outcomes["target_traj"]
        self.target_heatmap = outcomes["target_heatmap"]
        self.target_soft_heatmap = _cached_soft_heatmaps(
            outcomes_path=outcomes_path,
            target_traj=self.target_traj,
            target_heatmap=self.target_heatmap,
            max_pucks=self.target_final_pos.shape[-1] // 2,
        )
        self.target_wall_contact = outcomes["target_wall_contact"]
        self.target_ttc = outcomes["target_ttc"]

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.conditions[index]
        event_id = int(row["event_id"])
        event = self.events_by_id[event_id]
        metadata = row.get("metadata", {})
        scene_id = self.event_to_scene[event_id]
        scene_idx = self.scene_index[scene_id]
        outcome_idx = self.outcome_index[event_id]
        condition_idx = self.condition_index[int(row["condition_id"])]

        image = torch.from_numpy(self.context_image[scene_idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        condition_text = str(row["payload_inline"])
        impulse = event["impulse"]
        sample: dict[str, torch.Tensor | str | int] = {
            "context_image": image,
            "context_state": torch.from_numpy(self.context_state[scene_idx].astype(np.float32)).flatten(),
            "condition_params": torch.from_numpy(self.condition_params[condition_idx].astype(np.float32)),
            "condition_object_id": torch.tensor(int(event["target_object_id"]), dtype=torch.long),
            "condition_impulse": torch.tensor([float(impulse[0]), float(impulse[1])], dtype=torch.float32),
            "condition_horizon": torch.tensor([float(event["horizon"])], dtype=torch.float32),
            "condition_direction": torch.tensor(
                DIRECTION_TO_INDEX[str(event["direction_bucket"])],
                dtype=torch.long,
            ),
            "condition_magnitude": torch.tensor(
                MAGNITUDE_TO_INDEX[str(event["magnitude_bucket"])],
                dtype=torch.long,
            ),
            "condition_has_exact_impulse": torch.tensor(
                1.0 if bool(metadata.get("exact_impulse", row["family"] != "natural_language")) else 0.0,
                dtype=torch.float32,
            ),
            "target_final_pos": torch.from_numpy(self.target_final_pos[outcome_idx].astype(np.float32)).flatten(),
            "target_traj": torch.from_numpy(self.target_traj[outcome_idx].astype(np.float32)).flatten(),
            "target_heatmap": torch.from_numpy(self.target_heatmap[outcome_idx].astype(np.float32)).flatten(),
            "target_soft_heatmap": torch.from_numpy(
                self.target_soft_heatmap[outcome_idx].astype(np.float32)
            ).flatten(),
            "target_wall_contact": torch.from_numpy(self.target_wall_contact[outcome_idx].astype(np.float32)),
            "target_ttc": torch.from_numpy(self.target_ttc[outcome_idx].astype(np.float32)),
            "event_id": event_id,
            "condition_id": int(row["condition_id"]),
            "scene_hash": str(row["scene_hash"]),
            "event_hash": str(row["event_hash"]),
            "family": str(row["family"]),
            "condition_text": condition_text,
        }
        condition_id = int(row["condition_id"])
        if self.condition_latents_by_condition is not None:
            try:
                sample["z_condition"] = torch.from_numpy(
                    self.condition_latents_by_condition[condition_id].astype(np.float32)
                )
            except KeyError as exc:
                raise KeyError(f"Missing cached condition latent for condition_id={condition_id}") from exc
        else:
            sample["text_tokens"] = torch.tensor(byte_tokenize(condition_text, self.max_text_len), dtype=torch.long)
        if self.include_condition_image:
            sample["condition_image"] = text_to_image(condition_text, image.shape[-1])
        if self.target_latents_by_event is not None:
            try:
                sample["z_target"] = torch.from_numpy(self.target_latents_by_event[event_id].astype(np.float32))
            except KeyError as exc:
                raise KeyError(f"Missing cached target latent for event_id={event_id}") from exc
        return sample


class TargetOutcomeDataset(PuckDataset):
    """Outcome-only view that keeps one row per event within a split."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        max_text_len: int = 192,
        include_condition_image: bool = True,
        target_latents_by_event: dict[int, np.ndarray] | None = None,
        condition_latents_by_condition: dict[int, np.ndarray] | None = None,
        condition_family_filter: set[str] | None = None,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            split=split,
            max_text_len=max_text_len,
            include_condition_image=include_condition_image,
            target_latents_by_event=target_latents_by_event,
            condition_latents_by_condition=condition_latents_by_condition,
            condition_family_filter=condition_family_filter,
        )
        seen_event_ids: set[int] = set()
        unique_conditions: list[dict[str, Any]] = []
        for row in self.conditions:
            event_id = int(row["event_id"])
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            unique_conditions.append(row)
        self.conditions = unique_conditions


def byte_tokenize(text: str, max_len: int) -> list[int]:
    raw = text.encode("utf-8")[:max_len]
    tokens = [byte + 1 for byte in raw]
    if len(tokens) < max_len:
        tokens.extend([PAD_TOKEN] * (max_len - len(tokens)))
    return tokens


def text_to_image(text: str, size: int) -> torch.Tensor:
    """Encode serialized condition text as a fixed byte grid for stuffing.

    This keeps the monolithic baseline to one fused visual input without doing
    slow per-sample font rendering during training.
    """
    raw = np.frombuffer(text.encode("utf-8")[: size * size], dtype=np.uint8)
    grid = np.zeros((size, size), dtype=np.float32)
    if raw.size:
        grid.reshape(-1)[: raw.size] = raw.astype(np.float32) / 255.0
    occupied = (grid > 0).astype(np.float32)
    row_ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None].repeat(size, axis=1)
    arr = np.stack([grid, occupied, row_ramp], axis=0)
    return torch.from_numpy(arr)


def _cached_soft_heatmaps(
    outcomes_path: Path,
    target_traj: np.ndarray,
    target_heatmap: np.ndarray,
    max_pucks: int,
    chunk_size: int = 512,
) -> np.ndarray:
    heatmap_dim = int(np.prod(target_heatmap.shape[1:]))
    heatmap_size = int(np.sqrt(heatmap_dim))
    if heatmap_size * heatmap_size != heatmap_dim:
        return target_heatmap.reshape(target_heatmap.shape[0], -1).astype(np.float32)

    cache_key = (
        str(outcomes_path.resolve()),
        int(outcomes_path.stat().st_mtime_ns),
        heatmap_size,
        max_pucks,
    )
    cached = _SOFT_HEATMAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, target_traj.shape[0], chunk_size):
            trajectory = torch.from_numpy(target_traj[start : start + chunk_size].astype(np.float32))
            soft_heatmap = trajectory_to_soft_heatmap(
                trajectory,
                heatmap_size=heatmap_size,
                max_pucks=max_pucks,
            )
            chunks.append(soft_heatmap.cpu().numpy().astype(np.float32))
    cached = np.concatenate(chunks, axis=0)
    _SOFT_HEATMAP_CACHE[cache_key] = cached
    return cached


def _event_records(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    events = read_jsonl(dataset_dir / "manifests" / "events.jsonl")
    return {int(row["event_id"]): row for row in events}


def collate_metadata(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0].keys()
    output: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output
