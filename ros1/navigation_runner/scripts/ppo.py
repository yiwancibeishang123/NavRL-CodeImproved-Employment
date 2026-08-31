#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, make_sequence_batch, IndependentBeta, BetaActor, vec_to_new_frame, vec_to_world


class DynamicObstacleHistoryMLPEncoder(nn.Module):
    def __init__(
        self,
        history_steps: int,
        history_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.history_feat_dim = history_feat_dim
        self.input_scale = input_scale
        self.slot_encoder = nn.Sequential(
            nn.Linear(history_steps * history_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )

    def forward(self, obstacle_hist: torch.Tensor, obstacle_mask: torch.Tensor):
        # obstacle_hist: (..., 1, num_slots, history_steps, feat_dim)
        # obstacle_mask: (..., 1, num_slots)
        obstacle_hist = obstacle_hist.squeeze(-4)
        obstacle_mask = obstacle_mask.squeeze(-2).to(obstacle_hist.dtype)
        obstacle_hist = torch.nan_to_num(obstacle_hist, nan=0.0, posinf=self.input_scale, neginf=-self.input_scale)
        obstacle_hist = obstacle_hist / self.input_scale

        batch_shape = obstacle_hist.shape[:-3]
        num_slots, history_steps, feat_dim = obstacle_hist.shape[-3:]
        obstacle_hist = obstacle_hist.reshape(-1, history_steps * feat_dim)
        slot_feature = self.slot_encoder(obstacle_hist).reshape(*batch_shape, num_slots, -1)

        slot_feature = slot_feature * obstacle_mask.unsqueeze(-1)
        valid_slots = obstacle_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_feature = slot_feature.sum(dim=-2) / valid_slots
        pooled_feature = torch.nan_to_num(pooled_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        slot_feature = torch.nan_to_num(slot_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        return pooled_feature, slot_feature


class DynamicObstacleHistoryLSTMEncoder(nn.Module):
    def __init__(
        self,
        history_steps: int,
        history_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.history_feat_dim = history_feat_dim
        self.input_scale = input_scale
        self.input_norm = nn.LayerNorm(history_feat_dim)
        self.temporal_encoder = nn.LSTM(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.temporal_encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden_size = self.temporal_encoder.hidden_size
                # Encourage retaining short-term motion cues early in training.
                param.data[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(self, obstacle_hist: torch.Tensor, obstacle_mask: torch.Tensor):
        # obstacle_hist: (..., 1, num_slots, history_steps, feat_dim)
        # obstacle_mask: (..., 1, num_slots)
        obstacle_hist = obstacle_hist.squeeze(-4)
        obstacle_mask = obstacle_mask.squeeze(-2).to(obstacle_hist.dtype)
        obstacle_hist = torch.nan_to_num(obstacle_hist, nan=0.0, posinf=self.input_scale, neginf=-self.input_scale)
        obstacle_hist = obstacle_hist / self.input_scale
        obstacle_hist = self.input_norm(obstacle_hist)

        batch_shape = obstacle_hist.shape[:-3]
        num_slots, history_steps, feat_dim = obstacle_hist.shape[-3:]
        obstacle_hist = obstacle_hist.reshape(-1, history_steps, feat_dim)

        _, (hidden, _) = self.temporal_encoder(obstacle_hist)
        slot_feature = self.project(hidden[-1]).reshape(*batch_shape, num_slots, -1)
        slot_feature = torch.tanh(slot_feature)

        slot_feature = slot_feature * obstacle_mask.unsqueeze(-1)
        valid_slots = obstacle_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_feature = slot_feature.sum(dim=-2) / valid_slots
        pooled_feature = torch.nan_to_num(pooled_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        slot_feature = torch.nan_to_num(slot_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        return pooled_feature, slot_feature


class DynamicObstacleBaselineEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, out_dim]),
        )

    def forward(self, obstacle_state: torch.Tensor):
        feature = self.encoder(obstacle_state)
        slot_feature = torch.zeros(
            *feature.shape[:-1],
            1,
            feature.shape[-1],
            dtype=feature.dtype,
            device=feature.device,
        )
        return feature, slot_feature


class DynamicObstacleCurrentEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.input_scale = input_scale
        self.slot_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )

    def forward(self, obstacle_state: torch.Tensor):
        # obstacle_state: (..., 1, num_slots, feat_dim)
        obstacle_state = obstacle_state.squeeze(-3)
        obstacle_state = torch.nan_to_num(
            obstacle_state, nan=0.0, posinf=self.input_scale, neginf=-self.input_scale
        )
        obstacle_state = obstacle_state / self.input_scale

        batch_shape = obstacle_state.shape[:-2]
        num_slots, feat_dim = obstacle_state.shape[-2:]
        flat_state = obstacle_state.reshape(-1, feat_dim)
        slot_feature = self.slot_encoder(flat_state).reshape(*batch_shape, num_slots, -1)

        slot_presence = (obstacle_state.abs().sum(dim=-1) > 0).to(slot_feature.dtype)
        slot_feature = slot_feature * slot_presence.unsqueeze(-1)
        valid_slots = slot_presence.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_feature = slot_feature.sum(dim=-2) / valid_slots
        pooled_feature = torch.nan_to_num(pooled_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        slot_feature = torch.nan_to_num(slot_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        return pooled_feature, slot_feature


class LidarSequenceEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, lidar: torch.Tensor):
        batch_shape = lidar.shape[:-3]
        lidar = lidar.reshape(-1, *lidar.shape[-3:])
        feature = self.backbone(lidar)
        return feature.reshape(*batch_shape, -1)


class DynamicObstacleCurrentGRUEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.current_encoder = DynamicObstacleCurrentEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_scale=input_scale,
        )
        self.temporal_encoder = nn.GRU(
            input_size=out_dim,
            hidden_size=out_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(out_dim)
        self._hidden_state = None

    def reset_state(self) -> None:
        self._hidden_state = None

    def forward(self, obstacle_state: torch.Tensor):
        if obstacle_state.dim() == 5:
            batch_size, time_steps = obstacle_state.shape[:2]
            flat_state = obstacle_state.reshape(batch_size * time_steps, *obstacle_state.shape[2:])
            current_feature, slot_feature = self.current_encoder(flat_state)
            current_feature = current_feature.reshape(batch_size, time_steps, -1)
            temporal_feature, _ = self.temporal_encoder(current_feature)
            temporal_feature = self.output_norm(temporal_feature)
            slot_feature = slot_feature.reshape(batch_size, time_steps, *slot_feature.shape[1:])
            return temporal_feature, slot_feature

        if obstacle_state.dim() != 4:
            raise ValueError(f"Unsupported obstacle_state shape for GRU encoder: {obstacle_state.shape}")

        batch_size = obstacle_state.shape[0]
        current_feature, slot_feature = self.current_encoder(obstacle_state)
        if self._hidden_state is None or self._hidden_state.shape[1] != batch_size:
            self._hidden_state = torch.zeros(
                1,
                batch_size,
                self.temporal_encoder.hidden_size,
                device=current_feature.device,
                dtype=current_feature.dtype,
            )
        temporal_feature, next_hidden = self.temporal_encoder(current_feature.unsqueeze(1), self._hidden_state)
        self._hidden_state = next_hidden.detach()
        temporal_feature = self.output_norm(temporal_feature.squeeze(1))
        return temporal_feature, slot_feature


class DynamicObstacleCurrentTemporalEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        temporal_steps: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.temporal_steps = temporal_steps
        self.current_encoder = DynamicObstacleCurrentEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_scale=input_scale,
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(temporal_steps * out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self._feature_buffer = None

    def reset_state(self) -> None:
        self._feature_buffer = None

    def _reshape_init_mask(self, is_init: torch.Tensor, batch_size: int) -> torch.Tensor:
        is_init = is_init.reshape(batch_size, -1)
        return is_init.any(dim=-1)

    def _temporal_from_sequence(self, current_feature: torch.Tensor, is_init: torch.Tensor | None) -> torch.Tensor:
        batch_size, time_steps, feat_dim = current_feature.shape
        history = current_feature.new_zeros(batch_size, self.temporal_steps, feat_dim)
        temporal_outputs = []
        init_mask = None
        if is_init is not None:
            init_mask = is_init.reshape(batch_size, time_steps, -1).any(dim=-1)

        for step_idx in range(time_steps):
            if init_mask is not None:
                reset_mask = init_mask[:, step_idx]
                if reset_mask.any():
                    history = history * (~reset_mask).to(history.dtype).view(batch_size, 1, 1)
            history = torch.cat(
                [history[:, 1:], current_feature[:, step_idx].unsqueeze(1)],
                dim=1,
            )
            temporal_outputs.append(self.temporal_mlp(history.reshape(batch_size, -1)))

        return torch.stack(temporal_outputs, dim=1)

    def _temporal_from_online_buffer(self, current_feature: torch.Tensor, is_init: torch.Tensor | None) -> torch.Tensor:
        batch_size, feat_dim = current_feature.shape
        if self._feature_buffer is None or self._feature_buffer.shape[0] != batch_size:
            self._feature_buffer = current_feature.new_zeros(batch_size, self.temporal_steps, feat_dim)

        if is_init is not None:
            reset_mask = self._reshape_init_mask(is_init, batch_size)
            if reset_mask.any():
                self._feature_buffer = self._feature_buffer * (~reset_mask).to(self._feature_buffer.dtype).view(batch_size, 1, 1)

        self._feature_buffer = torch.cat(
            [self._feature_buffer[:, 1:], current_feature.unsqueeze(1)],
            dim=1,
        )
        return self.temporal_mlp(self._feature_buffer.reshape(batch_size, -1))

    def forward(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None = None):
        if obstacle_state.dim() == 5:
            batch_size, time_steps = obstacle_state.shape[:2]
            flat_state = obstacle_state.reshape(batch_size * time_steps, *obstacle_state.shape[2:])
            current_feature, slot_feature = self.current_encoder(flat_state)
            current_feature = current_feature.reshape(batch_size, time_steps, -1)
            temporal_feature = self._temporal_from_sequence(current_feature, is_init)
            slot_feature = slot_feature.reshape(batch_size, time_steps, *slot_feature.shape[1:])
            return temporal_feature, slot_feature

        if obstacle_state.dim() != 4:
            raise ValueError(f"Unsupported obstacle_state shape for temporal encoder: {obstacle_state.shape}")

        current_feature, slot_feature = self.current_encoder(obstacle_state)
        temporal_feature = self._temporal_from_online_buffer(current_feature, is_init)
        return temporal_feature, slot_feature


class DynamicObstacleCurrentTemporalConvEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        temporal_steps: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.temporal_steps = temporal_steps
        self.current_encoder = DynamicObstacleCurrentEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_scale=input_scale,
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(out_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.output_norm = nn.LayerNorm(out_dim)
        self._feature_buffer = None

    def reset_state(self) -> None:
        self._feature_buffer = None

    def _reshape_init_mask(self, is_init: torch.Tensor, batch_size: int) -> torch.Tensor:
        is_init = is_init.reshape(batch_size, -1)
        return is_init.any(dim=-1)

    def _encode_window(self, history: torch.Tensor, current_feature: torch.Tensor) -> torch.Tensor:
        temporal_map = self.temporal_conv(history.transpose(1, 2)).transpose(1, 2)
        temporal_feature = self.output_norm(current_feature + temporal_map[:, -1])
        return temporal_feature

    def _temporal_from_sequence(self, current_feature: torch.Tensor, is_init: torch.Tensor | None) -> torch.Tensor:
        batch_size, time_steps, feat_dim = current_feature.shape
        history = current_feature.new_zeros(batch_size, self.temporal_steps, feat_dim)
        temporal_outputs = []
        init_mask = None
        if is_init is not None:
            init_mask = is_init.reshape(batch_size, time_steps, -1).any(dim=-1)

        for step_idx in range(time_steps):
            if init_mask is not None:
                reset_mask = init_mask[:, step_idx]
                if reset_mask.any():
                    history = history * (~reset_mask).to(history.dtype).view(batch_size, 1, 1)
            history = torch.cat(
                [history[:, 1:], current_feature[:, step_idx].unsqueeze(1)],
                dim=1,
            )
            temporal_outputs.append(self._encode_window(history, current_feature[:, step_idx]))

        return torch.stack(temporal_outputs, dim=1)

    def _temporal_from_online_buffer(self, current_feature: torch.Tensor, is_init: torch.Tensor | None) -> torch.Tensor:
        batch_size, feat_dim = current_feature.shape
        if self._feature_buffer is None or self._feature_buffer.shape[0] != batch_size:
            self._feature_buffer = current_feature.new_zeros(batch_size, self.temporal_steps, feat_dim)

        if is_init is not None:
            reset_mask = self._reshape_init_mask(is_init, batch_size)
            if reset_mask.any():
                self._feature_buffer = self._feature_buffer * (~reset_mask).to(self._feature_buffer.dtype).view(batch_size, 1, 1)

        self._feature_buffer = torch.cat(
            [self._feature_buffer[:, 1:], current_feature.unsqueeze(1)],
            dim=1,
        )
        return self._encode_window(self._feature_buffer, current_feature)

    def forward(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None = None):
        if obstacle_state.dim() == 5:
            batch_size, time_steps = obstacle_state.shape[:2]
            flat_state = obstacle_state.reshape(batch_size * time_steps, *obstacle_state.shape[2:])
            current_feature, slot_feature = self.current_encoder(flat_state)
            current_feature = current_feature.reshape(batch_size, time_steps, -1)
            temporal_feature = self._temporal_from_sequence(current_feature, is_init)
            slot_feature = slot_feature.reshape(batch_size, time_steps, *slot_feature.shape[1:])
            return temporal_feature, slot_feature

        if obstacle_state.dim() != 4:
            raise ValueError(f"Unsupported obstacle_state shape for temporal conv encoder: {obstacle_state.shape}")

        current_feature, slot_feature = self.current_encoder(obstacle_state)
        temporal_feature = self._temporal_from_online_buffer(current_feature, is_init)
        return temporal_feature, slot_feature


class DynamicObstacleCurrentSlotResidualEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.current_encoder = DynamicObstacleCurrentEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_scale=input_scale,
        )
        self.slot_fuser = nn.Sequential(
            nn.Linear(out_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self._prev_slot_feature = None

    def reset_state(self) -> None:
        self._prev_slot_feature = None

    def _presence_mask(self, slot_feature: torch.Tensor) -> torch.Tensor:
        return (slot_feature.abs().sum(dim=-1) > 0).to(slot_feature.dtype)

    def _fuse_slot_feature(self, current_slot_feature: torch.Tensor, prev_slot_feature: torch.Tensor):
        delta_feature = current_slot_feature - prev_slot_feature
        slot_input = torch.cat(
            [current_slot_feature, delta_feature, delta_feature.abs()],
            dim=-1,
        )
        fused_slot_feature = self.slot_fuser(slot_input)
        slot_presence = self._presence_mask(current_slot_feature)
        fused_slot_feature = fused_slot_feature * slot_presence.unsqueeze(-1)
        valid_slots = slot_presence.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_feature = fused_slot_feature.sum(dim=-2) / valid_slots
        pooled_feature = torch.nan_to_num(pooled_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        fused_slot_feature = torch.nan_to_num(fused_slot_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        return pooled_feature, fused_slot_feature

    def _forward_sequence(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None):
        batch_size, time_steps = obstacle_state.shape[:2]
        flat_state = obstacle_state.reshape(batch_size * time_steps, *obstacle_state.shape[2:])
        _, current_slot_feature = self.current_encoder(flat_state)
        current_slot_feature = current_slot_feature.reshape(batch_size, time_steps, *current_slot_feature.shape[1:])

        prev_slot_feature = current_slot_feature.new_zeros(batch_size, *current_slot_feature.shape[-2:])
        init_mask = None
        if is_init is not None:
            init_mask = is_init.reshape(batch_size, time_steps, -1).any(dim=-1)

        pooled_outputs = []
        slot_outputs = []
        for step_idx in range(time_steps):
            if init_mask is not None:
                reset_mask = init_mask[:, step_idx]
                if reset_mask.any():
                    prev_slot_feature = prev_slot_feature * (~reset_mask).to(prev_slot_feature.dtype).view(batch_size, 1, 1)
            pooled_feature, fused_slot_feature = self._fuse_slot_feature(
                current_slot_feature[:, step_idx],
                prev_slot_feature,
            )
            pooled_outputs.append(pooled_feature)
            slot_outputs.append(fused_slot_feature)
            prev_slot_feature = current_slot_feature[:, step_idx]

        pooled_outputs = torch.stack(pooled_outputs, dim=1)
        slot_outputs = torch.stack(slot_outputs, dim=1)
        return pooled_outputs, slot_outputs

    def _forward_online(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None):
        batch_size = obstacle_state.shape[0]
        _, current_slot_feature = self.current_encoder(obstacle_state)
        if self._prev_slot_feature is None or self._prev_slot_feature.shape[0] != batch_size:
            self._prev_slot_feature = current_slot_feature.new_zeros(batch_size, *current_slot_feature.shape[-2:])

        if is_init is not None:
            reset_mask = is_init.reshape(batch_size, -1).any(dim=-1)
            if reset_mask.any():
                self._prev_slot_feature = self._prev_slot_feature * (~reset_mask).to(self._prev_slot_feature.dtype).view(batch_size, 1, 1)

        pooled_feature, fused_slot_feature = self._fuse_slot_feature(current_slot_feature, self._prev_slot_feature)
        self._prev_slot_feature = current_slot_feature.detach()
        return pooled_feature, fused_slot_feature

    def forward(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None = None):
        if obstacle_state.dim() == 5:
            return self._forward_sequence(obstacle_state, is_init)

        if obstacle_state.dim() != 4:
            raise ValueError(f"Unsupported obstacle_state shape for slot residual encoder: {obstacle_state.shape}")

        return self._forward_online(obstacle_state, is_init)


class DynamicObstacleCurrentMotionDescriptorEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.input_scale = input_scale
        self.current_encoder = DynamicObstacleCurrentEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_scale=input_scale,
        )
        descriptor_dim = feat_dim * 3 + 4
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self.slot_fuser = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self.slot_score = nn.Sequential(
            nn.Linear(out_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._prev_obstacle_state = None

    def reset_state(self) -> None:
        self._prev_obstacle_state = None

    def _presence_mask(self, obstacle_state: torch.Tensor) -> torch.Tensor:
        return (obstacle_state.abs().sum(dim=-1) > 0)

    def _build_motion_descriptor(self, current_state: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        delta_state = current_state - prev_state
        abs_delta_state = delta_state.abs()

        current_dist_2d = current_state[..., 3:4]
        prev_dist_2d = prev_state[..., 3:4]
        current_dist_z = current_state[..., 4:5]
        prev_dist_z = prev_state[..., 4:5]
        current_vel = current_state[..., 5:8]
        prev_vel = prev_state[..., 5:8]

        closing_rate_2d = prev_dist_2d - current_dist_2d
        closing_rate_z = prev_dist_z.abs() - current_dist_z.abs()
        speed = current_vel.norm(dim=-1, keepdim=True)
        delta_speed = (current_vel - prev_vel).norm(dim=-1, keepdim=True)

        descriptor = torch.cat(
            [
                current_state,
                delta_state,
                abs_delta_state,
                closing_rate_2d,
                closing_rate_z,
                speed,
                delta_speed,
            ],
            dim=-1,
        )
        descriptor = torch.nan_to_num(
            descriptor,
            nan=0.0,
            posinf=self.input_scale,
            neginf=-self.input_scale,
        )
        descriptor = descriptor / self.input_scale
        return descriptor

    def _fuse_slot_feature(self, current_state: torch.Tensor, prev_state: torch.Tensor):
        _, current_slot_feature = self.current_encoder(current_state.unsqueeze(-3))
        descriptor = self._build_motion_descriptor(current_state, prev_state)
        batch_shape = descriptor.shape[:-2]
        num_slots, descriptor_dim = descriptor.shape[-2:]
        flat_descriptor = descriptor.reshape(-1, descriptor_dim)
        descriptor_feature = self.descriptor_encoder(flat_descriptor).reshape(*batch_shape, num_slots, -1)

        fused_input = torch.cat([current_slot_feature, descriptor_feature], dim=-1)
        fused_slot_feature = self.slot_fuser(fused_input)

        slot_presence = self._presence_mask(current_state)
        masked_scores = self.slot_score(fused_slot_feature).squeeze(-1)
        masked_scores = masked_scores.masked_fill(~slot_presence, -1e9)
        slot_attention = torch.softmax(masked_scores, dim=-1)
        slot_attention = torch.where(
            slot_presence,
            slot_attention,
            torch.zeros_like(slot_attention),
        )
        norm = slot_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_attention = slot_attention / norm

        fused_slot_feature = fused_slot_feature * slot_presence.unsqueeze(-1).to(fused_slot_feature.dtype)
        pooled_feature = (fused_slot_feature * slot_attention.unsqueeze(-1)).sum(dim=-2)
        pooled_feature = torch.nan_to_num(pooled_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        fused_slot_feature = torch.nan_to_num(fused_slot_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        return pooled_feature, fused_slot_feature

    def _forward_sequence(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None):
        obstacle_state = obstacle_state.squeeze(-3)
        batch_size, time_steps = obstacle_state.shape[:2]
        prev_state = obstacle_state.new_zeros(batch_size, *obstacle_state.shape[-2:])
        init_mask = None
        if is_init is not None:
            init_mask = is_init.reshape(batch_size, time_steps, -1).any(dim=-1)

        pooled_outputs = []
        slot_outputs = []
        for step_idx in range(time_steps):
            if init_mask is not None:
                reset_mask = init_mask[:, step_idx]
                if reset_mask.any():
                    prev_state = prev_state * (~reset_mask).to(prev_state.dtype).view(batch_size, 1, 1)
            pooled_feature, fused_slot_feature = self._fuse_slot_feature(
                obstacle_state[:, step_idx],
                prev_state,
            )
            pooled_outputs.append(pooled_feature)
            slot_outputs.append(fused_slot_feature)
            prev_state = obstacle_state[:, step_idx]

        return torch.stack(pooled_outputs, dim=1), torch.stack(slot_outputs, dim=1)

    def _forward_online(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None):
        batch_size = obstacle_state.shape[0]
        current_state = obstacle_state.squeeze(-3)
        if self._prev_obstacle_state is None or self._prev_obstacle_state.shape[0] != batch_size:
            self._prev_obstacle_state = current_state.new_zeros(batch_size, *current_state.shape[-2:])

        if is_init is not None:
            reset_mask = is_init.reshape(batch_size, -1).any(dim=-1)
            if reset_mask.any():
                self._prev_obstacle_state = self._prev_obstacle_state * (~reset_mask).to(self._prev_obstacle_state.dtype).view(batch_size, 1, 1)

        pooled_feature, fused_slot_feature = self._fuse_slot_feature(
            current_state,
            self._prev_obstacle_state,
        )
        self._prev_obstacle_state = current_state.detach()
        return pooled_feature, fused_slot_feature

    def forward(self, obstacle_state: torch.Tensor, is_init: torch.Tensor | None = None):
        if obstacle_state.dim() == 5:
            return self._forward_sequence(obstacle_state, is_init)

        if obstacle_state.dim() != 4:
            raise ValueError(f"Unsupported obstacle_state shape for motion descriptor encoder: {obstacle_state.shape}")

        return self._forward_online(obstacle_state, is_init)


class DynamicObstacleCurrentGeometryEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        input_scale: float = 10.0,
        cpa_horizon: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_scale = input_scale
        self.cpa_horizon = cpa_horizon
        self.current_encoder = DynamicObstacleCurrentEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_scale=input_scale,
        )
        geometry_dim = 12
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self.slot_fuser = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self.slot_score = nn.Sequential(
            nn.Linear(out_dim, max(1, hidden_dim // 2)),
            nn.ELU(),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def _presence_mask(self, obstacle_state: torch.Tensor) -> torch.Tensor:
        return obstacle_state.abs().sum(dim=-1) > 0

    def _build_geometry_descriptor(self, obstacle_state: torch.Tensor) -> torch.Tensor:
        pos = obstacle_state[..., :3]
        dist_2d = obstacle_state[..., 3:4].clamp_min(1e-6)
        dist_z = obstacle_state[..., 4:5].abs()
        vel = obstacle_state[..., 5:8]
        width = obstacle_state[..., 8:9].clamp_min(0.0)
        height = obstacle_state[..., 9:10].clamp_min(0.0)

        pos_xy = pos[..., :2]
        vel_xy = vel[..., :2]
        radial_dir_xy = pos_xy / dist_2d
        closing_rate = -(vel_xy * radial_dir_xy).sum(dim=-1, keepdim=True)
        closing_rate = closing_rate.clamp_min(0.0)

        radial_velocity = (vel_xy * radial_dir_xy).sum(dim=-1, keepdim=True) * radial_dir_xy
        lateral_velocity = vel_xy - radial_velocity
        lateral_speed = lateral_velocity.norm(dim=-1, keepdim=True)
        speed_2d = vel_xy.norm(dim=-1, keepdim=True)
        speed_3d = vel.norm(dim=-1, keepdim=True)

        clearance_2d = (dist_2d - 0.5 * width).clamp_min(0.0)
        clearance_z = (dist_z - 0.5 * height).clamp_min(0.0)
        inv_ttc = torch.where(
            closing_rate > 1e-4,
            closing_rate / clearance_2d.clamp_min(1e-3),
            torch.zeros_like(closing_rate),
        )

        vel_xy_sq = (vel_xy.square().sum(dim=-1, keepdim=True)).clamp_min(1e-6)
        t_cpa = (-(pos_xy * vel_xy).sum(dim=-1, keepdim=True) / vel_xy_sq).clamp(0.0, self.cpa_horizon)
        d_cpa = (pos_xy + vel_xy * t_cpa).norm(dim=-1, keepdim=True)

        frontness = (pos[..., 0:1] > 0).to(obstacle_state.dtype)
        lateral_offset = pos[..., 1:2].abs()

        descriptor = torch.cat(
            [
                dist_2d / self.input_scale,
                clearance_2d / self.input_scale,
                dist_z / self.input_scale,
                clearance_z / self.input_scale,
                speed_2d / self.input_scale,
                speed_3d / self.input_scale,
                closing_rate / self.input_scale,
                lateral_speed / self.input_scale,
                inv_ttc.clamp(0.0, 1.0),
                d_cpa / self.input_scale,
                frontness,
                lateral_offset / self.input_scale,
            ],
            dim=-1,
        )
        descriptor = torch.nan_to_num(descriptor, nan=0.0, posinf=1.0, neginf=0.0)
        return descriptor

    def forward(self, obstacle_state: torch.Tensor):
        if obstacle_state.dim() != 4:
            raise ValueError(f"Unsupported obstacle_state shape for geometry encoder: {obstacle_state.shape}")

        current_feature, current_slot_feature = self.current_encoder(obstacle_state)
        current_state = obstacle_state.squeeze(-3)
        geometry_descriptor = self._build_geometry_descriptor(current_state)

        batch_shape = geometry_descriptor.shape[:-2]
        num_slots, descriptor_dim = geometry_descriptor.shape[-2:]
        flat_descriptor = geometry_descriptor.reshape(-1, descriptor_dim)
        geometry_feature = self.geometry_encoder(flat_descriptor).reshape(*batch_shape, num_slots, -1)

        fused_slot_feature = self.slot_fuser(torch.cat([current_slot_feature, geometry_feature], dim=-1))
        slot_presence = self._presence_mask(current_state)
        fused_slot_feature = fused_slot_feature * slot_presence.unsqueeze(-1).to(fused_slot_feature.dtype)

        slot_scores = self.slot_score(fused_slot_feature).squeeze(-1)
        slot_scores = slot_scores.masked_fill(~slot_presence, -1e9)
        slot_attention = torch.softmax(slot_scores, dim=-1)
        slot_attention = torch.where(slot_presence, slot_attention, torch.zeros_like(slot_attention))
        slot_attention = slot_attention / slot_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        pooled_feature = (fused_slot_feature * slot_attention.unsqueeze(-1)).sum(dim=-2)
        pooled_feature = torch.nan_to_num(pooled_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        fused_slot_feature = torch.nan_to_num(fused_slot_feature, nan=0.0, posinf=1.0, neginf=-1.0)
        return pooled_feature, fused_slot_feature


class RiskFieldPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        field_size: int,
    ) -> None:
        super().__init__()
        self.field_size = field_size
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.head = nn.Linear(hidden_dim, field_size * field_size)

    def forward(self, obstacle_feature: torch.Tensor):
        hidden = self.backbone(obstacle_feature)
        field_logit = self.head(hidden).unflatten(-1, (1, self.field_size, self.field_size))
        field = torch.sigmoid(field_logit)
        return field, field_logit


class DecoderRiskFieldPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        field_size: int,
        latent_channels: int = 16,
        latent_size: int = 4,
    ) -> None:
        super().__init__()
        self.field_size = field_size
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, latent_channels * latent_size * latent_size),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(latent_channels, latent_channels // 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(latent_channels // 2, 1, kernel_size=1),
        )

    def forward(self, obstacle_feature: torch.Tensor):
        hidden = self.backbone(obstacle_feature)
        latent = self.project(hidden).unflatten(-1, (self.latent_channels, self.latent_size, self.latent_size))
        latent = latent.reshape(-1, self.latent_channels, self.latent_size, self.latent_size)
        latent = F.interpolate(
            latent,
            size=(self.field_size, self.field_size),
            mode="bilinear",
            align_corners=False,
        )
        field_logit = self.decoder(latent)
        field_logit = field_logit.reshape(*obstacle_feature.shape[:-1], 1, self.field_size, self.field_size)
        field = torch.sigmoid(field_logit)
        return field, field_logit


class RiskFieldEncoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(hidden_dim // 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.LazyConv2d(hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.ELU(),
            Rearrange("n c h w -> n (c h w)"),
            nn.LazyLinear(out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )

    def forward(self, risk_field: torch.Tensor):
        batch_shape = risk_field.shape[:-3]
        field = risk_field.reshape(-1, *risk_field.shape[-3:])
        feature = self.encoder(field)
        return feature.reshape(*batch_shape, -1)


class ResidualCostCritic(nn.Module):
    """Cost critic with an explicit residual branch for Step3 risk conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        center_risk_value: bool = False,
        gate_with_risk_score: bool = False,
    ) -> None:
        super().__init__()
        self.center_risk_value = center_risk_value
        self.gate_with_risk_score = gate_with_risk_score
        self.base_branch = nn.Sequential(
            make_mlp([hidden_dim, hidden_dim]),
            nn.LazyLinear(1),
        )
        self.risk_branch = nn.Sequential(
            make_mlp([hidden_dim, hidden_dim]),
            nn.LazyLinear(1),
        )

    def forward(
        self,
        base_feature: torch.Tensor,
        risk_embedding: torch.Tensor,
        risk_summary: torch.Tensor,
    ):
        risk_condition = torch.cat([risk_embedding, risk_summary], dim=-1)
        base_value = self.base_branch(base_feature)
        risk_value = self.risk_branch(risk_condition)
        if self.center_risk_value and risk_value.numel() > 1:
            risk_value = risk_value - risk_value.mean()
        if self.gate_with_risk_score:
            # Use the predicted field's max risk as a transparent gate so the
            # residual branch is strongest where the risk-field is active
            # without starving the branch early in training.
            risk_gate = 0.25 + 0.75 * risk_summary[..., 1:2].clamp(0.0, 1.0)
            risk_value = risk_value * risk_gate
        cost_value = base_value + risk_value
        return cost_value, base_value, risk_value


class RiskGuidedBetaActor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        field_size: int,
        field_range: float,
        hidden_dim: int = 128,
        probe_distance: float = 2.5,
        near_scale: float = 0.7,
        far_scale: float = 1.35,
        lateral_offset: float = 0.6,
        refinement_scale: float = 0.35,
        profile_hidden_dim: int = 64,
        concentration_modulation_scale: float = 0.4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.field_size = field_size
        self.field_range = field_range
        self.probe_distance = probe_distance
        self.near_scale = near_scale
        self.far_scale = far_scale
        self.lateral_offset = lateral_offset
        self.refinement_scale = refinement_scale
        self.concentration_modulation_scale = concentration_modulation_scale

        self.nominal_mean_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.LazyLinear(action_dim),
            nn.Tanh(),
        )
        self.nominal_concentration_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.LazyLinear(action_dim),
        )
        self.profile_encoder = nn.Sequential(
            nn.LazyLinear(profile_hidden_dim),
            nn.LayerNorm(profile_hidden_dim),
            nn.ELU(),
            nn.LazyLinear(profile_hidden_dim),
            nn.LayerNorm(profile_hidden_dim),
            nn.ELU(),
        )
        self.modulation_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.LazyLinear(2 * action_dim),
        )

    def _sample_risk(self, field: torch.Tensor, probe_xy: torch.Tensor) -> torch.Tensor:
        norm_xy = (probe_xy / self.field_range).clamp(-1.0, 1.0)
        grid = norm_xy.reshape(-1, 1, 1, 2)
        sampled = F.grid_sample(
            field,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(*norm_xy.shape[:-1], 1)

    def _query_profile(self, action_xy: torch.Tensor, predicted_risk_field: torch.Tensor) -> torch.Tensor:
        field = predicted_risk_field.reshape(-1, 1, self.field_size, self.field_size)
        action_norm = action_xy.norm(dim=-1, keepdim=True)
        default_dir = torch.zeros_like(action_xy)
        default_dir[..., 0] = 1.0
        unit_dir = action_xy / action_norm.clamp_min(1e-6)
        unit_dir = torch.where(action_norm > 1e-3, unit_dir, default_dir)
        unit_dir = F.normalize(unit_dir, dim=-1, eps=1e-6)
        lateral_dir = torch.stack([-unit_dir[..., 1], unit_dir[..., 0]], dim=-1)

        center = unit_dir * self.probe_distance
        near = unit_dir * (self.probe_distance * self.near_scale)
        far = unit_dir * (self.probe_distance * self.far_scale)
        left = center + lateral_dir * self.lateral_offset
        right = center - lateral_dir * self.lateral_offset

        profile = torch.cat(
            [
                self._sample_risk(field, near),
                self._sample_risk(field, center),
                self._sample_risk(field, far),
                self._sample_risk(field, left),
                self._sample_risk(field, right),
            ],
            dim=-1,
        )
        return torch.nan_to_num(profile, nan=0.0, posinf=1.0, neginf=0.0)

    def forward(
        self,
        base_feature: torch.Tensor,
        risk_embedding: torch.Tensor,
        risk_summary: torch.Tensor,
        predicted_risk_field: torch.Tensor,
    ):
        nominal_action = self.nominal_mean_head(base_feature)
        nominal_concentration = 2.0 + F.softplus(self.nominal_concentration_head(base_feature))
        nominal_pref = ((nominal_action + 1.0) * 0.5).clamp(1e-3, 1.0 - 1e-3)
        risk_profile = self._query_profile(nominal_action[..., :2], predicted_risk_field)

        center_risk = risk_profile[..., 1:2]
        lateral_mean = 0.5 * (risk_profile[..., 3:4] + risk_profile[..., 4:5])
        left_right_asym = risk_profile[..., 3:4] - risk_profile[..., 4:5]
        near_far_diff = risk_profile[..., 0:1] - risk_profile[..., 2:3]
        profile_energy = risk_profile.mean(dim=-1, keepdim=True)
        profile_features = torch.cat(
            [
                risk_profile,
                center_risk,
                lateral_mean,
                left_right_asym,
                near_far_diff,
                profile_energy,
            ],
            dim=-1,
        )
        profile_embedding = self.profile_encoder(profile_features)
        modulation_input = torch.cat(
            [base_feature, risk_embedding, risk_summary, nominal_action, profile_embedding, risk_profile],
            dim=-1,
        )
        logit_shift_raw, concentration_shift_raw = self.modulation_head(modulation_input).chunk(2, dim=-1)
        nominal_logit = torch.logit(nominal_pref)
        logit_shift = torch.tanh(logit_shift_raw) * self.refinement_scale
        concentration_shift = torch.tanh(concentration_shift_raw) * self.concentration_modulation_scale

        final_logit = nominal_logit + logit_shift
        final_pref = torch.sigmoid(final_logit).clamp(1e-3, 1.0 - 1e-3)
        concentration = (nominal_concentration * torch.exp(concentration_shift)).clamp_min(2.0)
        refined_action = final_pref * 2.0 - 1.0
        risk_aware_action = refined_action
        action_update = (refined_action - nominal_action).abs()
        alpha = 1.0 + final_pref * concentration
        beta = 1.0 + (1.0 - final_pref) * concentration
        alpha = alpha.clamp(1.001, 20.0)
        beta = beta.clamp(1.001, 20.0)
        return alpha, beta, nominal_action, risk_aware_action, refined_action, risk_profile, action_update


class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device


        # Feature extractor for LiDAR
        lidar_backbone = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)

        history_cfg = cfg.get("observation_aux", {})
        method_cfg = cfg.get("method", {})
        step1_enabled = bool(method_cfg.get("step1_enabled", True))
        step2_enabled = bool(method_cfg.get("step2_enabled", True))
        step3_enabled = bool(method_cfg.get("step3_enabled", True))
        history_steps = int(history_cfg.get("history_steps", 4))
        history_feat_dim = int(history_cfg.get("history_feat_dim", 7))
        history_hidden_dim = int(history_cfg.get("history_hidden_dim", 64))
        history_out_dim = int(history_cfg.get("history_out_dim", 64))
        history_input_scale = float(history_cfg.get("history_input_scale", 10.0))
        obstacle_input_mode = str(history_cfg.get("obstacle_input_mode", "history")).lower()
        history_encoder_type = str(history_cfg.get("history_encoder_type", "mlp")).lower()
        risk_prediction_enabled = bool(
            history_cfg.get("risk_prediction_enabled", history_cfg.get("prediction_enabled", False))
        )
        risk_prediction_hidden_dim = int(
            history_cfg.get("risk_prediction_hidden_dim", history_cfg.get("prediction_hidden_dim", 64))
        )
        risk_prediction_loss_coef = float(
            history_cfg.get("risk_prediction_loss_coef", history_cfg.get("prediction_loss_coef", 0.1))
        )
        risk_prediction_future_steps = int(
            history_cfg.get("risk_prediction_future_steps", history_cfg.get("prediction_future_steps", 3))
        )
        risk_prediction_future_stride = int(
            history_cfg.get("risk_prediction_future_stride", history_cfg.get("prediction_future_stride", 1))
        )
        risk_predictor_decoder_enabled = bool(history_cfg.get("risk_predictor_decoder_enabled", False))
        risk_prediction_learning_rate = float(
            history_cfg.get("risk_prediction_learning_rate", history_cfg.get("prediction_learning_rate", cfg.feature_extractor.learning_rate))
        )
        risk_conflict_distance = float(history_cfg.get("risk_conflict_distance", 1.0))
        risk_field_size = int(history_cfg.get("risk_field_size", 8))
        risk_field_range = float(history_cfg.get("risk_field_range", 4.0))
        risk_field_hidden_dim = int(history_cfg.get("risk_field_hidden_dim", 32))
        risk_field_time_discount = float(history_cfg.get("risk_field_time_discount", 0.8))
        risk_target_base_sigma = float(history_cfg.get("risk_target_base_sigma", 0.45))
        risk_target_step_sigma_scale = float(history_cfg.get("risk_target_step_sigma_scale", 0.06))
        risk_target_corridor_width = float(history_cfg.get("risk_target_corridor_width", 1.25))
        risk_target_forward_weight = float(history_cfg.get("risk_target_forward_weight", 0.35))
        risk_target_approach_scale = float(history_cfg.get("risk_target_approach_scale", 0.8))
        risk_target_crossing_scale = float(history_cfg.get("risk_target_crossing_scale", 0.45))
        risk_embedding_dim = int(history_cfg.get("risk_embedding_dim", 32))
        risk_policy_fusion_enabled = bool(history_cfg.get("risk_policy_fusion_enabled", False))
        risk_policy_use_summary = bool(history_cfg.get("risk_policy_use_summary", False))
        risk_actor_penalty_enabled = bool(history_cfg.get("risk_actor_penalty_enabled", False))
        risk_actor_penalty_coef = float(history_cfg.get("risk_actor_penalty_coef", 0.0))
        risk_actor_probe_distance = float(history_cfg.get("risk_actor_probe_distance", 2.0))
        risk_actor_patch_size = int(history_cfg.get("risk_actor_patch_size", 1))
        risk_actor_patch_weighted = bool(history_cfg.get("risk_actor_patch_weighted", False))
        risk_actor_patch_weight_sigma = float(history_cfg.get("risk_actor_patch_weight_sigma", 1.0))
        risk_actor_adaptive_enabled = bool(history_cfg.get("risk_actor_adaptive_enabled", False))
        risk_actor_adaptive_scale = float(history_cfg.get("risk_actor_adaptive_scale", 0.0))
        risk_actor_goal_progress_enabled = bool(history_cfg.get("risk_actor_goal_progress_enabled", False))
        risk_actor_goal_progress_scale = float(history_cfg.get("risk_actor_goal_progress_scale", 0.0))
        risk_actor_profile_enabled = bool(history_cfg.get("risk_actor_profile_enabled", False))
        risk_actor_profile_near_scale = float(history_cfg.get("risk_actor_profile_near_scale", 0.7))
        risk_actor_profile_far_scale = float(history_cfg.get("risk_actor_profile_far_scale", 1.3))
        risk_actor_profile_lateral_offset = float(history_cfg.get("risk_actor_profile_lateral_offset", 0.5))
        risk_actor_threshold = float(history_cfg.get("risk_actor_threshold", 0.08))
        risk_action_refinement_enabled = bool(history_cfg.get("risk_action_refinement_enabled", False))
        risk_action_refinement_hidden_dim = int(history_cfg.get("risk_action_refinement_hidden_dim", 128))
        risk_action_refinement_scale = float(history_cfg.get("risk_action_refinement_scale", 0.35))
        risk_action_refinement_profile_hidden_dim = int(
            history_cfg.get("risk_action_refinement_profile_hidden_dim", 64)
        )
        risk_action_candidate_risk_weight = float(
            history_cfg.get("risk_action_candidate_risk_weight", 2.5)
        )
        risk_action_candidate_align_weight = float(
            history_cfg.get("risk_action_candidate_align_weight", 1.25)
        )
        risk_action_candidate_speed_weight = float(
            history_cfg.get("risk_action_candidate_speed_weight", 0.35)
        )
        risk_action_candidate_deviation_weight = float(
            history_cfg.get("risk_action_candidate_deviation_weight", 0.4)
        )
        risk_action_candidate_temperature = float(
            history_cfg.get("risk_action_candidate_temperature", 0.35)
        )
        risk_action_concentration_modulation_scale = float(
            history_cfg.get("risk_action_concentration_modulation_scale", 0.4)
        )
        stage3_curriculum_enabled = bool(history_cfg.get("stage3_curriculum_enabled", False))
        stage3_fusion_start_frames = int(float(history_cfg.get("stage3_fusion_start_frames", 0)))
        stage3_penalty_start_frames = int(float(history_cfg.get("stage3_penalty_start_frames", 0)))
        stage3_profile_start_frames = int(float(history_cfg.get("stage3_profile_start_frames", 0)))
        stage3_penalty_warmup_frames = int(float(history_cfg.get("stage3_penalty_warmup_frames", 1)))
        stage3_profile_warmup_frames = int(float(history_cfg.get("stage3_profile_warmup_frames", 1)))
        lstm_lr_scale = float(history_cfg.get("lstm_lr_scale", 0.5))
        lstm_epoch_num = int(history_cfg.get("lstm_epoch_num", 2))
        critic_value_clip = float(history_cfg.get("critic_value_clip", 10.0))
        constrained_cfg = cfg.get("constrained_rl", {})
        p3o_cfg = cfg.get("p3o", {})
        p3o_enabled = bool(p3o_cfg.get("enabled", False))
        constrained_enabled = bool(constrained_cfg.get("enabled", False))
        allow_constrained_without_step3 = bool(constrained_cfg.get("allow_without_step3", False)) or p3o_enabled
        if not step1_enabled:
            obstacle_input_mode = "baseline"
        if not step2_enabled:
            risk_prediction_enabled = False
        if not step3_enabled and not allow_constrained_without_step3:
            constrained_enabled = False
        if p3o_enabled:
            constrained_enabled = True
        if constrained_enabled:
            # In constrained mode, keep actor-side risk mechanisms disabled so only the
            # constrained objective shapes behavior.
            risk_policy_fusion_enabled = False
            risk_policy_use_summary = False
            risk_actor_penalty_enabled = False
            risk_actor_adaptive_enabled = False
            risk_actor_goal_progress_enabled = False
            risk_actor_profile_enabled = False
            risk_action_refinement_enabled = False
            stage3_curriculum_enabled = False
        cost_critic_hidden_dim = int(constrained_cfg.get("cost_critic_hidden_dim", 128))
        cost_critic_coef = float(constrained_cfg.get("cost_critic_coef", 1.0))
        lagrange_init = float(constrained_cfg.get("lagrange_init", 0.0))
        lagrange_lr = float(constrained_cfg.get("lagrange_lr", 1e-2))
        lagrange_max = float(constrained_cfg.get("lagrange_max", 10.0))
        cost_budget = float(constrained_cfg.get("cost_budget", 0.12))
        cost_gamma = float(constrained_cfg.get("cost_gamma", 0.99))
        cost_lambda = float(constrained_cfg.get("cost_lambda", 0.95))
        cost_critic_conditioning = str(constrained_cfg.get("cost_critic_conditioning", "full")).lower()
        valid_cost_critic_conditioning = {"no_cond_same_dim", "summary", "embedding", "full"}
        if cost_critic_conditioning not in valid_cost_critic_conditioning:
            cost_critic_conditioning = "full"
        cost_critic_conditioning_style = str(constrained_cfg.get("cost_critic_conditioning_style", "concat")).lower()
        if cost_critic_conditioning_style not in {
            "concat",
            "gated",
            "residual",
            "residual_shuffled",
            "calibrated_residual",
            "calibrated_residual_shuffled",
            "risk_guided_residual",
            "risk_guided_residual_shuffled",
        }:
            cost_critic_conditioning_style = "concat"
        cost_condition_hidden_dim = int(constrained_cfg.get("cost_condition_hidden_dim", cost_critic_hidden_dim))
        cost_condition_gate_scale = float(constrained_cfg.get("cost_condition_gate_scale", 0.5))
        cost_condition_shift_scale = float(constrained_cfg.get("cost_condition_shift_scale", 0.5))
        cost_critic_high_risk_ratio = float(constrained_cfg.get("cost_critic_high_risk_ratio", 0.2))
        cost_critic_risk_score_source = str(constrained_cfg.get("cost_critic_risk_score_source", "summary_max")).lower()
        if cost_critic_risk_score_source not in {"summary_max", "topk", "goal_corridor"}:
            cost_critic_risk_score_source = "summary_max"
        cost_critic_risk_priority_coef = float(constrained_cfg.get("cost_critic_risk_priority_coef", 0.0))
        cost_critic_risk_priority_max = float(constrained_cfg.get("cost_critic_risk_priority_max", 3.0))
        cost_critic_risk_residual_coef = float(constrained_cfg.get("cost_critic_risk_residual_coef", 0.0))
        cost_critic_risk_align_coef = float(constrained_cfg.get("cost_critic_risk_align_coef", 0.02))
        cost_critic_risk_rank_coef = float(constrained_cfg.get("cost_critic_risk_rank_coef", 0.05))
        cost_critic_risk_center_coef = float(constrained_cfg.get("cost_critic_risk_center_coef", 0.01))
        cost_critic_risk_weighting_enabled = bool(constrained_cfg.get("cost_critic_risk_weighting_enabled", False))
        cost_critic_risk_weight_alpha = float(constrained_cfg.get("cost_critic_risk_weight_alpha", 2.0))
        cost_critic_risk_weight_source = str(constrained_cfg.get("cost_critic_risk_weight_source", "max")).lower()
        if cost_critic_risk_weight_source not in {"mean", "max", "active", "energy", "topk"}:
            cost_critic_risk_weight_source = "max"
        cost_critic_risk_weight_topk_ratio = float(constrained_cfg.get("cost_critic_risk_weight_topk_ratio", 0.1))
        cost_critic_risk_weight_max = float(constrained_cfg.get("cost_critic_risk_weight_max", 5.0))
        cost_critic_risk_weight_normalize = bool(constrained_cfg.get("cost_critic_risk_weight_normalize", True))
        cost_source = str(constrained_cfg.get("cost_source", "env")).lower()
        if cost_source not in {"env", "risk_field", "hybrid"}:
            cost_source = "env"
        risk_cost_mode = str(constrained_cfg.get("risk_cost_mode", "topk")).lower()
        if risk_cost_mode not in {"topk", "goal_corridor"}:
            risk_cost_mode = "topk"
        risk_cost_topk_ratio = float(constrained_cfg.get("risk_cost_topk_ratio", 0.1))
        risk_cost_weight = float(constrained_cfg.get("risk_cost_weight", 0.0))
        risk_cost_threshold = float(constrained_cfg.get("risk_cost_threshold", 0.1))
        risk_cost_start_frames = int(float(constrained_cfg.get("risk_cost_start_frames", 0)))
        risk_cost_warmup_frames = int(float(constrained_cfg.get("risk_cost_warmup_frames", 1)))
        risk_cost_shuffle_enabled = bool(constrained_cfg.get("risk_cost_shuffle_enabled", False))
        risk_cost_corridor_distance = float(constrained_cfg.get("risk_cost_corridor_distance", 2.5))
        risk_cost_corridor_near_scale = float(constrained_cfg.get("risk_cost_corridor_near_scale", 0.55))
        risk_cost_corridor_far_scale = float(constrained_cfg.get("risk_cost_corridor_far_scale", 1.25))
        risk_cost_corridor_lateral_offset = float(constrained_cfg.get("risk_cost_corridor_lateral_offset", 0.6))
        risk_cost_detach = bool(constrained_cfg.get("risk_cost_detach", True))
        self.history_encoder_type = history_encoder_type
        self.method_step1_enabled = step1_enabled
        self.method_step2_enabled = step2_enabled
        self.method_step3_enabled = step3_enabled
        if step1_enabled and step2_enabled and step3_enabled:
            self.method_variant_code = 3.0
        elif step1_enabled and step2_enabled and not step3_enabled:
            self.method_variant_code = 2.0
        elif step1_enabled and not step2_enabled and not step3_enabled:
            self.method_variant_code = 1.0
        elif not step1_enabled and not step2_enabled and not step3_enabled:
            self.method_variant_code = 0.0
        else:
            self.method_variant_code = -1.0
        self.risk_prediction_enabled = risk_prediction_enabled
        self.risk_predictor_decoder_enabled = risk_predictor_decoder_enabled
        self.risk_prediction_loss_coef = risk_prediction_loss_coef
        self.risk_prediction_future_steps = risk_prediction_future_steps
        self.risk_prediction_future_stride = risk_prediction_future_stride
        self.risk_prediction_learning_rate = risk_prediction_learning_rate
        self.risk_conflict_distance = risk_conflict_distance
        self.risk_field_size = risk_field_size
        self.risk_field_range = risk_field_range
        self.risk_field_time_discount = risk_field_time_discount
        self.risk_target_base_sigma = max(risk_target_base_sigma, 0.05)
        self.risk_target_step_sigma_scale = max(risk_target_step_sigma_scale, 0.0)
        self.risk_target_corridor_width = max(risk_target_corridor_width, 0.1)
        self.risk_target_forward_weight = max(risk_target_forward_weight, 0.0)
        self.risk_target_approach_scale = max(risk_target_approach_scale, 0.0)
        self.risk_target_crossing_scale = max(risk_target_crossing_scale, 0.0)
        self.risk_prediction_dim = 1
        self.risk_embedding_dim = risk_embedding_dim
        cell_size = (2.0 * self.risk_field_range) / float(self.risk_field_size)
        grid_lin = torch.linspace(
            -self.risk_field_range + 0.5 * cell_size,
            self.risk_field_range - 0.5 * cell_size,
            self.risk_field_size,
            device=self.device,
        )
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        self.risk_grid_x = grid_x.view(1, 1, 1, self.risk_field_size, self.risk_field_size)
        self.risk_grid_y = grid_y.view(1, 1, 1, self.risk_field_size, self.risk_field_size)
        refinement_mode = risk_action_refinement_enabled and risk_prediction_enabled
        self.risk_policy_fusion_target = (risk_policy_fusion_enabled and risk_prediction_enabled) and not refinement_mode
        self.risk_policy_use_summary_target = risk_policy_use_summary and self.risk_policy_fusion_target
        self.risk_policy_fusion_enabled = self.risk_policy_fusion_target
        self.risk_policy_use_summary = self.risk_policy_use_summary_target
        self.risk_field_hidden_dim = risk_field_hidden_dim
        self.risk_actor_penalty_target = (risk_actor_penalty_enabled and risk_prediction_enabled) and not refinement_mode
        self.risk_actor_penalty_enabled = self.risk_actor_penalty_target
        self.risk_actor_penalty_coef_max = risk_actor_penalty_coef
        self.risk_actor_penalty_coef = self.risk_actor_penalty_coef_max
        self.risk_actor_probe_distance = risk_actor_probe_distance
        self.risk_actor_patch_size = max(1, risk_actor_patch_size)
        self.risk_actor_patch_weighted_target = risk_actor_patch_weighted and self.risk_actor_patch_size > 1
        self.risk_actor_patch_weighted = self.risk_actor_patch_weighted_target
        self.risk_actor_patch_weight_sigma = max(risk_actor_patch_weight_sigma, 1e-3)
        self.risk_actor_adaptive_target = risk_actor_adaptive_enabled and self.risk_actor_penalty_target
        self.risk_actor_adaptive_enabled = self.risk_actor_adaptive_target
        self.risk_actor_adaptive_scale_max = risk_actor_adaptive_scale
        self.risk_actor_adaptive_scale = self.risk_actor_adaptive_scale_max
        self.risk_actor_goal_progress_target = risk_actor_goal_progress_enabled and self.risk_actor_penalty_target
        self.risk_actor_goal_progress_enabled = self.risk_actor_goal_progress_target
        self.risk_actor_goal_progress_scale_max = risk_actor_goal_progress_scale
        self.risk_actor_goal_progress_scale = self.risk_actor_goal_progress_scale_max
        self.risk_actor_profile_target = risk_actor_profile_enabled and self.risk_actor_penalty_target
        self.risk_actor_profile_enabled = self.risk_actor_profile_target
        self.risk_actor_profile_near_scale = risk_actor_profile_near_scale
        self.risk_actor_profile_far_scale = risk_actor_profile_far_scale
        self.risk_actor_profile_lateral_offset = risk_actor_profile_lateral_offset
        self.risk_actor_threshold = max(risk_actor_threshold, 0.0)
        self.risk_action_refinement_enabled = refinement_mode
        self.risk_action_refinement_hidden_dim = risk_action_refinement_hidden_dim
        self.risk_action_refinement_scale = risk_action_refinement_scale
        self.risk_action_refinement_profile_hidden_dim = risk_action_refinement_profile_hidden_dim
        self.risk_action_candidate_risk_weight = risk_action_candidate_risk_weight
        self.risk_action_candidate_align_weight = risk_action_candidate_align_weight
        self.risk_action_candidate_speed_weight = risk_action_candidate_speed_weight
        self.risk_action_candidate_deviation_weight = risk_action_candidate_deviation_weight
        self.risk_action_candidate_temperature = risk_action_candidate_temperature
        self.risk_action_concentration_modulation_scale = risk_action_concentration_modulation_scale
        self.stage3_curriculum_enabled = (stage3_curriculum_enabled and risk_prediction_enabled) and not refinement_mode
        self.stage3_fusion_start_frames = stage3_fusion_start_frames
        self.stage3_penalty_start_frames = stage3_penalty_start_frames
        self.stage3_profile_start_frames = stage3_profile_start_frames
        self.stage3_penalty_warmup_frames = max(1, stage3_penalty_warmup_frames)
        self.stage3_profile_warmup_frames = max(1, stage3_profile_warmup_frames)
        self.value_clip = critic_value_clip
        self.training_epoch_num = cfg.training_epoch_num
        self.obstacle_input_mode = obstacle_input_mode
        self.constrained_rl_enabled = constrained_enabled
        rpo_cfg = cfg.get("rpo", {})
        self.rpo_enabled = bool(rpo_cfg.get("enabled", False))
        self.rpo_alpha = max(0.0, float(rpo_cfg.get("alpha", 0.0)))
        self.p3o_enabled = p3o_enabled
        self.p3o_penalty_coef = max(0.0, float(p3o_cfg.get("penalty_coef", 1.0)))
        self.p3o_violation_scale = max(0.0, float(p3o_cfg.get("violation_scale", 1.0)))
        ppg_cfg = cfg.get("ppg", {})
        self.ppg_enabled = bool(ppg_cfg.get("enabled", False))
        self.ppg_aux_epoch_num = max(0, int(ppg_cfg.get("aux_epoch_num", 0)))
        self.ppg_aux_num_minibatches = max(1, int(ppg_cfg.get("aux_num_minibatches", cfg.num_minibatches)))
        self.ppg_aux_interval = max(1, int(ppg_cfg.get("aux_interval", 1)))
        self.ppg_aux_value_coef = max(0.0, float(ppg_cfg.get("aux_value_coef", 1.0)))
        self.ppg_aux_kl_coef = max(0.0, float(ppg_cfg.get("aux_kl_coef", 1.0)))
        self.ppg_aux_learning_rate = float(ppg_cfg.get("aux_learning_rate", cfg.critic.learning_rate))
        self.ppg_detach_actor_feature = bool(ppg_cfg.get("detach_actor_feature", True))
        self.ppg_update_counter = 0
        self.cost_critic_coef = max(cost_critic_coef, 0.0)
        self.lagrange_lr = max(lagrange_lr, 0.0)
        self.lagrange_max = max(lagrange_max, 0.0)
        self.cost_budget = cost_budget
        self.cost_gamma = cost_gamma
        self.cost_lambda = cost_lambda
        self.cost_critic_conditioning = cost_critic_conditioning
        self.cost_critic_conditioning_style = cost_critic_conditioning_style
        self.cost_condition_gate_scale = max(cost_condition_gate_scale, 0.0)
        self.cost_condition_shift_scale = max(cost_condition_shift_scale, 0.0)
        self.cost_critic_high_risk_ratio = min(1.0, max(cost_critic_high_risk_ratio, 1e-6))
        self.cost_critic_risk_score_source = cost_critic_risk_score_source
        self.cost_critic_risk_priority_coef = max(cost_critic_risk_priority_coef, 0.0)
        self.cost_critic_risk_priority_max = max(cost_critic_risk_priority_max, 1.0)
        self.cost_critic_risk_residual_coef = max(cost_critic_risk_residual_coef, 0.0)
        self.cost_critic_risk_align_coef = max(cost_critic_risk_align_coef, 0.0)
        self.cost_critic_risk_rank_coef = max(cost_critic_risk_rank_coef, 0.0)
        self.cost_critic_risk_center_coef = max(cost_critic_risk_center_coef, 0.0)
        self.cost_critic_risk_weighting_enabled = cost_critic_risk_weighting_enabled
        self.cost_critic_risk_weight_alpha = max(cost_critic_risk_weight_alpha, 0.0)
        self.cost_critic_risk_weight_source = cost_critic_risk_weight_source
        self.cost_critic_risk_weight_topk_ratio = min(1.0, max(cost_critic_risk_weight_topk_ratio, 1e-6))
        self.cost_critic_risk_weight_max = max(cost_critic_risk_weight_max, 1.0)
        self.cost_critic_risk_weight_normalize = cost_critic_risk_weight_normalize
        self.cost_source = cost_source
        self.risk_cost_mode = risk_cost_mode
        self.risk_cost_topk_ratio = min(1.0, max(risk_cost_topk_ratio, 1e-6))
        self.risk_cost_weight = max(risk_cost_weight, 0.0)
        self.risk_cost_threshold = max(risk_cost_threshold, 0.0)
        self.risk_cost_start_frames = max(0, risk_cost_start_frames)
        self.risk_cost_warmup_frames = max(1, risk_cost_warmup_frames)
        self.risk_cost_shuffle_enabled = risk_cost_shuffle_enabled
        self.risk_cost_corridor_distance = min(self.risk_field_range, max(risk_cost_corridor_distance, 0.1))
        self.risk_cost_corridor_near_scale = max(risk_cost_corridor_near_scale, 0.1)
        self.risk_cost_corridor_far_scale = max(risk_cost_corridor_far_scale, self.risk_cost_corridor_near_scale)
        self.risk_cost_corridor_lateral_offset = max(risk_cost_corridor_lateral_offset, 0.0)
        self.risk_cost_scale = 0.0
        self.risk_cost_detach = risk_cost_detach

        if obstacle_input_mode == "baseline":
            dynamic_obstacle_network = DynamicObstacleBaselineEncoder(
                out_dim=history_out_dim,
            ).to(self.device)
            dynamic_obstacle_in_keys = [("agents", "observation", "dynamic_obstacle")]
        elif obstacle_input_mode == "current":
            dynamic_obstacle_network = DynamicObstacleCurrentEncoder(
                feat_dim=10,
                hidden_dim=history_hidden_dim,
                out_dim=history_out_dim,
                input_scale=history_input_scale,
            ).to(self.device)
            dynamic_obstacle_in_keys = [("agents", "observation", "dynamic_obstacle")]
        elif obstacle_input_mode == "current_gru":
            dynamic_obstacle_network = DynamicObstacleCurrentGRUEncoder(
                feat_dim=10,
                hidden_dim=history_hidden_dim,
                out_dim=history_out_dim,
                input_scale=history_input_scale,
            ).to(self.device)
            dynamic_obstacle_in_keys = [("agents", "observation", "dynamic_obstacle")]
        elif obstacle_input_mode == "current_temporal":
            dynamic_obstacle_network = DynamicObstacleCurrentTemporalEncoder(
                feat_dim=10,
                hidden_dim=history_hidden_dim,
                out_dim=history_out_dim,
                temporal_steps=history_steps,
                input_scale=history_input_scale,
            ).to(self.device)
            dynamic_obstacle_in_keys = [
                ("agents", "observation", "dynamic_obstacle"),
                "is_init",
            ]
        elif obstacle_input_mode == "current_temporal_conv":
            dynamic_obstacle_network = DynamicObstacleCurrentTemporalConvEncoder(
                feat_dim=10,
                hidden_dim=history_hidden_dim,
                out_dim=history_out_dim,
                temporal_steps=history_steps,
                input_scale=history_input_scale,
            ).to(self.device)
            dynamic_obstacle_in_keys = [
                ("agents", "observation", "dynamic_obstacle"),
                "is_init",
            ]
        elif obstacle_input_mode == "current_slot_residual":
            dynamic_obstacle_network = DynamicObstacleCurrentSlotResidualEncoder(
                feat_dim=10,
                hidden_dim=history_hidden_dim,
                out_dim=history_out_dim,
                input_scale=history_input_scale,
            ).to(self.device)
            dynamic_obstacle_in_keys = [
                ("agents", "observation", "dynamic_obstacle"),
                "is_init",
            ]
        elif obstacle_input_mode == "current_motion_descriptor":
            dynamic_obstacle_network = DynamicObstacleCurrentMotionDescriptorEncoder(
                feat_dim=10,
                hidden_dim=history_hidden_dim,
                out_dim=history_out_dim,
                input_scale=history_input_scale,
            ).to(self.device)
            dynamic_obstacle_in_keys = [
                ("agents", "observation", "dynamic_obstacle"),
                "is_init",
            ]
        elif obstacle_input_mode == "current_geometry":
            dynamic_obstacle_network = DynamicObstacleCurrentGeometryEncoder(
                feat_dim=10,
                hidden_dim=history_hidden_dim,
                out_dim=history_out_dim,
                input_scale=history_input_scale,
            ).to(self.device)
            dynamic_obstacle_in_keys = [("agents", "observation", "dynamic_obstacle")]
        elif obstacle_input_mode == "history":
            # Step 3: use obstacle history directly, without future prediction or auxiliary losses.
            if history_encoder_type == "mlp":
                dynamic_obstacle_network = DynamicObstacleHistoryMLPEncoder(
                    history_steps=history_steps,
                    history_feat_dim=history_feat_dim,
                    hidden_dim=history_hidden_dim,
                    out_dim=history_out_dim,
                    input_scale=history_input_scale,
                ).to(self.device)
            elif history_encoder_type == "lstm":
                dynamic_obstacle_network = DynamicObstacleHistoryLSTMEncoder(
                    history_steps=history_steps,
                    history_feat_dim=history_feat_dim,
                    hidden_dim=history_hidden_dim,
                    out_dim=history_out_dim,
                    input_scale=history_input_scale,
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported history encoder type: {history_encoder_type}")
            dynamic_obstacle_in_keys = [
                ("agents", "observation", "dynamic_obstacle_hist"),
                ("agents", "observation", "dynamic_obstacle_mask"),
            ]
        else:
            raise ValueError(f"Unsupported obstacle input mode: {obstacle_input_mode}")
        self.risk_predictor = None
        if self.risk_prediction_enabled:
            predictor_cls = DecoderRiskFieldPredictor if self.risk_predictor_decoder_enabled else RiskFieldPredictor
            self.risk_predictor = predictor_cls(
                input_dim=history_out_dim,
                hidden_dim=risk_prediction_hidden_dim,
                field_size=risk_field_size,
            ).to(self.device)
        self.risk_field_encoder = None
        if self.risk_prediction_enabled:
            self.risk_field_encoder = RiskFieldEncoder(
                hidden_dim=risk_field_hidden_dim,
                out_dim=risk_embedding_dim,
            ).to(self.device)

        feature_lr = cfg.feature_extractor.learning_rate
        actor_lr = cfg.actor.learning_rate
        critic_lr = cfg.critic.learning_rate
        if obstacle_input_mode == "history" and history_encoder_type == "lstm":
            feature_lr *= lstm_lr_scale
            actor_lr *= lstm_lr_scale
            critic_lr *= lstm_lr_scale
            self.training_epoch_num = lstm_epoch_num

        # Feature extractor
        feature_dim = 256
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(LidarSequenceEncoder(lidar_backbone), [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(
                dynamic_obstacle_network,
                dynamic_obstacle_in_keys,
                ["_dynamic_obstacle_feature", "_dynamic_obstacle_slot_feature"],
            ),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False),
            TensorDictModule(make_mlp([feature_dim, feature_dim]), ["_feature"], ["_feature"]),
        ).to(self.device)

        self.cost_condition_norm = None
        self.cost_condition_gate = None
        self.cost_condition_shift = None
        if self.constrained_rl_enabled and self.cost_critic_conditioning_style == "gated":
            risk_condition_dim = self.risk_embedding_dim + 4
            cost_condition_hidden_dim = max(16, cost_condition_hidden_dim)
            self.cost_condition_norm = nn.LayerNorm(risk_condition_dim).to(self.device)
            self.cost_condition_gate = nn.Sequential(
                nn.Linear(risk_condition_dim, cost_condition_hidden_dim),
                nn.ELU(),
                nn.Linear(cost_condition_hidden_dim, feature_dim),
            ).to(self.device)
            self.cost_condition_shift = nn.Sequential(
                nn.Linear(risk_condition_dim, cost_condition_hidden_dim),
                nn.ELU(),
                nn.Linear(cost_condition_hidden_dim, feature_dim),
            ).to(self.device)

        # Actor etwork
        self.n_agents, self.action_dim = action_spec.shape
        if self.risk_action_refinement_enabled:
            actor_module = TensorDictModule(
                RiskGuidedBetaActor(
                    action_dim=self.action_dim,
                    field_size=self.risk_field_size,
                    field_range=self.risk_field_range,
                    hidden_dim=self.risk_action_refinement_hidden_dim,
                    probe_distance=self.risk_actor_probe_distance,
                    near_scale=self.risk_actor_profile_near_scale,
                    far_scale=self.risk_actor_profile_far_scale,
                    lateral_offset=self.risk_actor_profile_lateral_offset,
                    refinement_scale=self.risk_action_refinement_scale,
                    profile_hidden_dim=self.risk_action_refinement_profile_hidden_dim,
                    concentration_modulation_scale=self.risk_action_concentration_modulation_scale,
                ),
                ["_feature", "_risk_embedding", "_risk_summary", "_predicted_risk_field"],
                ["alpha", "beta", "_nominal_action", "_risk_aware_action", "_refined_action", "_risk_query_profile", "_action_update"],
            )
        else:
            actor_module = TensorDictModule(BetaActor(self.action_dim), ["_policy_feature"], ["alpha", "beta"])
        self.actor = ProbabilisticActor(
            actor_module,
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_policy_feature"], ["state_value"]
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)
        self.cost_critic = None
        self.cost_value_norm = None
        if self.constrained_rl_enabled:
            if self.cost_critic_conditioning_style in {
                "residual",
                "residual_shuffled",
                "calibrated_residual",
                "calibrated_residual_shuffled",
                "risk_guided_residual",
                "risk_guided_residual_shuffled",
            }:
                center_risk_value = self.cost_critic_conditioning_style in {
                    "calibrated_residual",
                    "calibrated_residual_shuffled",
                    "risk_guided_residual",
                    "risk_guided_residual_shuffled",
                }
                gate_with_risk_score = self.cost_critic_conditioning_style in {
                    "risk_guided_residual",
                    "risk_guided_residual_shuffled",
                }
                self.cost_critic = TensorDictModule(
                    ResidualCostCritic(
                        cost_critic_hidden_dim,
                        center_risk_value=center_risk_value,
                        gate_with_risk_score=gate_with_risk_score,
                    ),
                    ["_feature", "_cost_risk_embedding", "_cost_risk_summary"],
                    ["cost_value", "cost_base_value", "cost_risk_value"],
                ).to(self.device)
            else:
                self.cost_critic = TensorDictModule(
                    nn.Sequential(
                        make_mlp([cost_critic_hidden_dim, cost_critic_hidden_dim]),
                        nn.LazyLinear(1),
                    ),
                    ["_cost_value_feature"],
                    ["cost_value"],
                ).to(self.device)
            self.cost_value_norm = ValueNorm(1).to(self.device)
            self.register_buffer("lagrange_multiplier", torch.tensor(lagrange_init, device=self.device))

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.cost_gae = GAE(cost_gamma, cost_lambda) if self.constrained_rl_enabled else None
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        feature_params = list(self.feature_extractor.parameters())
        self.feature_extractor_optim = torch.optim.Adam(feature_params, lr=feature_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.ppg_aux_optim = None
        if self.ppg_enabled and self.ppg_aux_epoch_num > 0:
            self.ppg_aux_optim = torch.optim.Adam(
                list(self.feature_extractor.parameters())
                + list(self.actor.parameters())
                + list(self.critic.parameters()),
                lr=self.ppg_aux_learning_rate,
            )
        self.cost_critic_optim = None
        if self.cost_critic is not None:
            cost_critic_params = list(self.cost_critic.parameters())
            if self.cost_condition_gate is not None:
                cost_critic_params += list(self.cost_condition_gate.parameters())
            if self.cost_condition_shift is not None:
                cost_critic_params += list(self.cost_condition_shift.parameters())
            if self.cost_condition_norm is not None:
                cost_critic_params += list(self.cost_condition_norm.parameters())
            self.cost_critic_optim = torch.optim.Adam(cost_critic_params, lr=critic_lr)
        self.risk_prediction_optim = None
        if self.risk_predictor is not None:
            prediction_params = list(self.risk_predictor.parameters())
            if self.risk_field_encoder is not None:
                prediction_params += list(self.risk_field_encoder.parameters())
            self.risk_prediction_optim = torch.optim.Adam(prediction_params, lr=risk_prediction_learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        if obstacle_input_mode in {"current_temporal", "current_temporal_conv", "current_slot_residual", "current_motion_descriptor"}:
            init_shape = tuple(dummy_input.batch_size) + (1,) if len(dummy_input.batch_size) > 0 else (1,)
            dummy_input.set("is_init", torch.zeros(init_shape, dtype=torch.bool, device=self.device))
        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)
        if self.cost_critic is not None:
            self.cost_critic.apply(init_)
        if self.cost_condition_gate is not None:
            self.cost_condition_gate.apply(init_)
        if self.cost_condition_shift is not None:
            self.cost_condition_shift.apply(init_)
        self.set_training_progress(0)

    @staticmethod
    def _ramp(env_frames: int, start: int, warmup: int) -> float:
        if env_frames < start:
            return 0.0
        if warmup <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (env_frames - start) / float(warmup))))

    def set_training_progress(self, env_frames: int):
        self.risk_cost_scale = self._ramp(
            env_frames,
            self.risk_cost_start_frames,
            self.risk_cost_warmup_frames,
        )
        if not self.stage3_curriculum_enabled:
            return

        fusion_on = env_frames >= self.stage3_fusion_start_frames and self.risk_policy_fusion_target
        penalty_ramp = self._ramp(env_frames, self.stage3_penalty_start_frames, self.stage3_penalty_warmup_frames)
        profile_ramp = self._ramp(env_frames, self.stage3_profile_start_frames, self.stage3_profile_warmup_frames)

        self.risk_policy_fusion_enabled = fusion_on
        self.risk_policy_use_summary = self.risk_policy_use_summary_target and (profile_ramp > 0.0)

        self.risk_actor_penalty_enabled = self.risk_actor_penalty_target and (penalty_ramp > 0.0)
        self.risk_actor_penalty_coef = self.risk_actor_penalty_coef_max * penalty_ramp
        self.risk_actor_goal_progress_enabled = self.risk_actor_goal_progress_target and self.risk_actor_penalty_enabled
        self.risk_actor_goal_progress_scale = self.risk_actor_goal_progress_scale_max * penalty_ramp

        self.risk_actor_profile_enabled = self.risk_actor_profile_target and (profile_ramp > 0.0)
        self.risk_actor_patch_weighted = self.risk_actor_patch_weighted_target and (profile_ramp > 0.0)
        self.risk_actor_adaptive_enabled = self.risk_actor_adaptive_target and (profile_ramp > 0.0)
        self.risk_actor_adaptive_scale = self.risk_actor_adaptive_scale_max * profile_ramp
    def reset_temporal_state(self):
        for module in self.feature_extractor:
            wrapped = getattr(module, "module", None)
            if wrapped is not None and hasattr(wrapped, "reset_state"):
                wrapped.reset_state()

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        if self.risk_prediction_enabled:
            self._predict_future_risk(tensordict)
        self._build_policy_feature(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        if self.cost_critic is not None:
            self.cost_critic(tensordict)
        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    @torch.no_grad()
    def act(self, tensordict):
        """Run inference without executing training-only value/cost heads.

        The actor still receives the exact feature path used during training,
        including the geometry encoder and optional risk embedding. Value,
        cost-critic, loss and optimizer work are intentionally skipped.
        """
        self.feature_extractor(tensordict)
        if self.risk_prediction_enabled:
            self._predict_future_risk(tensordict)
        self._build_policy_feature(tensordict)
        self.actor(tensordict)
        actions = (
            2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit
            - self.cfg.actor.action_limit
        )
        actions_world = vec_to_world(
            actions, tensordict["agents", "observation", "direction"]
        )
        tensordict["agents", "action"] = actions_world
        return tensordict

    def _predict_future_risk(self, tensordict: TensorDict):
        if not self.risk_prediction_enabled or self.risk_predictor is None:
            return
        obstacle_feature = tensordict["_dynamic_obstacle_feature"].detach()
        predicted_risk_field, risk_field_logit = self.risk_predictor(obstacle_feature)
        tensordict.set("_predicted_risk_field", predicted_risk_field)
        tensordict.set("_predicted_risk_field_logit", risk_field_logit)

    def _shuffle_cost_risk_condition(self, risk_embedding: torch.Tensor, risk_summary: torch.Tensor):
        flat_embedding = risk_embedding.reshape(-1, risk_embedding.shape[-1])
        flat_summary = risk_summary.reshape(-1, risk_summary.shape[-1])
        if flat_embedding.shape[0] <= 1:
            return risk_embedding, risk_summary
        perm = torch.randperm(flat_embedding.shape[0], device=flat_embedding.device)
        shuffled_embedding = flat_embedding[perm].reshape_as(risk_embedding)
        shuffled_summary = flat_summary[perm].reshape_as(risk_summary)
        return shuffled_embedding, shuffled_summary

    def _build_policy_feature(self, tensordict: TensorDict):
        base_feature = tensordict["_feature"]
        batch_shape = base_feature.shape[:-1]
        if not self.risk_prediction_enabled or self.risk_predictor is None:
            risk_summary = torch.zeros(*batch_shape, 4, device=base_feature.device, dtype=base_feature.dtype)
            risk_embedding = torch.zeros(*batch_shape, self.risk_embedding_dim, device=base_feature.device, dtype=base_feature.dtype)
            risk_attention = torch.zeros(*batch_shape, self.risk_field_size * self.risk_field_size, device=base_feature.device, dtype=base_feature.dtype)
            policy_feature = base_feature
        else:
            predicted_risk_field = tensordict["_predicted_risk_field"]
            field_flat = predicted_risk_field.flatten(start_dim=-3)
            mean_risk = field_flat.mean(dim=-1, keepdim=True)
            max_risk = field_flat.max(dim=-1).values.unsqueeze(-1)
            active_ratio = (field_flat > 0.1).float().mean(dim=-1, keepdim=True)
            field_energy = field_flat.square().sum(dim=-1, keepdim=True) / float(self.risk_field_size * self.risk_field_size)
            risk_summary = torch.cat(
                [
                    mean_risk,
                    max_risk,
                    active_ratio,
                    field_energy,
                ],
                dim=-1,
            ).detach()
            risk_embedding = self.risk_field_encoder(predicted_risk_field.detach())
            risk_attention = field_flat / field_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            if self.risk_action_refinement_enabled:
                policy_feature = torch.cat([base_feature, risk_embedding, risk_summary], dim=-1)
            elif self.risk_policy_fusion_enabled:
                policy_feature_parts = [base_feature, risk_embedding]
                if self.risk_policy_use_summary:
                    policy_feature_parts.append(risk_summary)
                policy_feature = torch.cat(policy_feature_parts, dim=-1)
            else:
                policy_feature = base_feature
        if self.constrained_rl_enabled:
            zero_embedding = torch.zeros_like(risk_embedding)
            zero_summary = torch.zeros_like(risk_summary)
            if self.cost_critic_conditioning == "no_cond_same_dim":
                cost_risk_embedding = zero_embedding
                cost_risk_summary = zero_summary
            elif self.cost_critic_conditioning == "summary":
                cost_risk_embedding = zero_embedding
                cost_risk_summary = risk_summary
            elif self.cost_critic_conditioning == "embedding":
                cost_risk_embedding = risk_embedding
                cost_risk_summary = zero_summary
            else:
                cost_risk_embedding = risk_embedding
                cost_risk_summary = risk_summary

            if self.cost_critic_conditioning_style in {
                "residual_shuffled",
                "calibrated_residual_shuffled",
                "risk_guided_residual_shuffled",
            }:
                cost_risk_embedding, cost_risk_summary = self._shuffle_cost_risk_condition(
                    cost_risk_embedding,
                    cost_risk_summary,
                )

            if self.cost_critic_conditioning_style == "gated":
                risk_condition = torch.cat([cost_risk_embedding, cost_risk_summary], dim=-1)
                risk_condition = self.cost_condition_norm(risk_condition)
                gate = torch.tanh(self.cost_condition_gate(risk_condition))
                shift = torch.tanh(self.cost_condition_shift(risk_condition))
                conditioned_feature = (
                    base_feature * (1.0 + self.cost_condition_gate_scale * gate)
                    + self.cost_condition_shift_scale * shift
                )
                cost_value_feature = torch.cat(
                    [conditioned_feature, cost_risk_embedding, cost_risk_summary],
                    dim=-1,
                )
            else:
                cost_value_feature = torch.cat([base_feature, cost_risk_embedding, cost_risk_summary], dim=-1)
        else:
            cost_risk_embedding = torch.zeros_like(risk_embedding)
            cost_risk_summary = torch.zeros_like(risk_summary)
            cost_value_feature = base_feature
        tensordict.set("_risk_summary", risk_summary)
        tensordict.set("_risk_embedding", risk_embedding)
        tensordict.set("_risk_attention", risk_attention)
        tensordict.set("_cost_risk_embedding", cost_risk_embedding)
        tensordict.set("_cost_risk_summary", cost_risk_summary)
        tensordict.set("_policy_feature", policy_feature)
        tensordict.set("_cost_value_feature", cost_value_feature)

    def _update_lagrange_multiplier(self, mean_cost: torch.Tensor):
        if not self.constrained_rl_enabled:
            return
        with torch.no_grad():
            updated = self.lagrange_multiplier + self.lagrange_lr * (mean_cost - self.cost_budget)
            self.lagrange_multiplier.copy_(updated.clamp_(0.0, self.lagrange_max))

    def _sample_predicted_risk(self, predicted_risk_field: torch.Tensor, probe_xy: torch.Tensor) -> torch.Tensor:
        field = predicted_risk_field.reshape(-1, 1, self.risk_field_size, self.risk_field_size)
        norm_xy = (probe_xy / self.risk_field_range).clamp(-1.0, 1.0)
        grid = norm_xy.reshape(-1, 1, 1, 2)
        sampled = F.grid_sample(
            field,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(*norm_xy.shape[:-1], 1)

    def _compute_goal_corridor_risk_score(self, predicted_risk_field: torch.Tensor) -> torch.Tensor:
        batch_shape = predicted_risk_field.shape[:-3]
        device = predicted_risk_field.device
        dtype = predicted_risk_field.dtype
        flat_count = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else 1

        def probe_xy(x_distance: float, y_offset: float = 0.0) -> torch.Tensor:
            xy = torch.zeros(flat_count, 2, device=device, dtype=dtype)
            xy[:, 0] = min(self.risk_field_range, max(0.0, x_distance))
            xy[:, 1] = max(-self.risk_field_range, min(self.risk_field_range, y_offset))
            return xy.reshape(*batch_shape, 2)

        distance = self.risk_cost_corridor_distance
        near_risk = self._sample_predicted_risk(
            predicted_risk_field,
            probe_xy(distance * self.risk_cost_corridor_near_scale),
        )
        center_risk = self._sample_predicted_risk(
            predicted_risk_field,
            probe_xy(distance),
        )
        far_risk = self._sample_predicted_risk(
            predicted_risk_field,
            probe_xy(distance * self.risk_cost_corridor_far_scale),
        )
        left_risk = self._sample_predicted_risk(
            predicted_risk_field,
            probe_xy(distance, self.risk_cost_corridor_lateral_offset),
        )
        right_risk = self._sample_predicted_risk(
            predicted_risk_field,
            probe_xy(distance, -self.risk_cost_corridor_lateral_offset),
        )

        frontal_risk = torch.maximum(center_risk, far_risk)
        flank_risk = 0.5 * (left_risk + right_risk)
        corridor_score = 0.25 * near_risk + 0.50 * frontal_risk + 0.25 * flank_risk
        return torch.nan_to_num(corridor_score)

    def _compute_risk_field_cost(self, tensordict: TensorDict, batch_size=None):
        if (
            not self.constrained_rl_enabled
            or not self.risk_prediction_enabled
            or "_predicted_risk_field" not in tensordict.keys(True, True)
        ):
            return None
        predicted_risk_field = tensordict["_predicted_risk_field"]
        if self.risk_cost_detach:
            predicted_risk_field = predicted_risk_field.detach()
        if self.risk_cost_mode == "goal_corridor":
            risk_score = self._compute_goal_corridor_risk_score(predicted_risk_field)
        else:
            field_flat = predicted_risk_field.flatten(start_dim=-3)
            topk_num = max(1, int(round(field_flat.shape[-1] * self.risk_cost_topk_ratio)))
            topk_risk = torch.topk(field_flat, topk_num, dim=-1).values
            risk_score = topk_risk.mean(dim=-1, keepdim=True)
        if self.risk_cost_shuffle_enabled and risk_score.numel() > 1:
            flat_score = risk_score.reshape(-1, risk_score.shape[-1])
            perm = torch.randperm(flat_score.shape[0], device=flat_score.device)
            risk_score = flat_score[perm].reshape_as(risk_score)
        risk_excess = F.relu(risk_score - self.risk_cost_threshold)
        risk_cost = risk_excess * self.risk_cost_weight * self.risk_cost_scale
        risk_cost = torch.nan_to_num(risk_cost)
        if batch_size is not None:
            risk_cost = risk_cost.view(*batch_size, 1)
        return risk_cost

    def _combine_constrained_cost(self, env_cost, risk_field_cost):
        if not self.constrained_rl_enabled:
            return None
        if env_cost is None:
            return risk_field_cost
        if risk_field_cost is None:
            return env_cost
        if self.cost_source == "risk_field":
            return risk_field_cost
        if self.cost_source == "hybrid":
            return env_cost + risk_field_cost
        return env_cost

    def _build_risk_prediction_targets(self, tensordict: TensorDict):
        catalogs = tensordict[("info", "dyn_obs_catalog")]
        ids = tensordict[("agents", "observation", "dynamic_obstacle_ids")].squeeze(-2)
        directions = tensordict[("agents", "observation", "direction")].squeeze(-2)
        drone_pos = tensordict[("info", "drone_state")][..., 0, :3]
        env_num, rollout_steps = catalogs.shape[:2]
        future_steps = self.risk_prediction_future_steps
        future_stride = self.risk_prediction_future_stride
        field_target = torch.zeros(
            (env_num, rollout_steps, 1, self.risk_field_size, self.risk_field_size),
            device=catalogs.device,
            dtype=catalogs.dtype,
        )
        safe_ids = ids.long().clamp_min(0)
        valid_id = ids >= 0
        for step_idx in range(future_steps):
            offset = step_idx * future_stride
            if offset >= rollout_steps:
                break
            steps = rollout_steps - offset
            future_catalog = catalogs[:, offset:, ...]
            future_xy = future_catalog[..., :2]
            future_vel = future_catalog[..., 2:4]
            future_visible = future_catalog[..., 4:5]
            curr_ids = safe_ids[:, :steps, :]
            gathered_xy = torch.gather(
                future_xy,
                2,
                curr_ids.unsqueeze(-1).expand(*curr_ids.shape, future_xy.shape[-1]),
            )
            gathered_vel = torch.gather(
                future_vel,
                2,
                curr_ids.unsqueeze(-1).expand(*curr_ids.shape, future_vel.shape[-1]),
            )
            gathered_visible = torch.gather(
                future_visible,
                2,
                curr_ids.unsqueeze(-1).expand(*curr_ids.shape, future_visible.shape[-1]),
            )
            num_slots = curr_ids.shape[-1]

            rel_current_goal = torch.zeros(
                (env_num, steps, num_slots, 3),
                device=catalogs.device,
                dtype=catalogs.dtype,
            )
            rel_current_goal[..., :2] = gathered_xy

            if offset > 0:
                future_drone_pos = drone_pos[:, offset:, :]
                current_drone_pos = drone_pos[:, :steps, :]
                current_dir = directions[:, :steps, :]
                drone_shift_world = future_drone_pos - current_drone_pos
                drone_shift_goal = vec_to_new_frame(
                    drone_shift_world.reshape(env_num * steps, 1, 3),
                    current_dir.reshape(env_num * steps, 3),
                ).reshape(env_num, steps, 1, 3)
                rel_current_goal = rel_current_goal + drone_shift_goal
            rel_current_goal = torch.nan_to_num(rel_current_goal, nan=0.0, posinf=0.0, neginf=0.0)

            x = rel_current_goal[..., 0]
            y = rel_current_goal[..., 1]
            vx = gathered_vel[..., 0]
            vy = gathered_vel[..., 1]
            dist = torch.sqrt(x.square() + y.square()).clamp_min(1e-4)
            approach_rate = (-(x * vx + y * vy) / dist).clamp_min(0.0)
            crossing_rate = vy.abs()
            forward_factor = torch.sigmoid(x / self.risk_target_corridor_width)
            forward_weight = (1.0 - self.risk_target_forward_weight) + self.risk_target_forward_weight * forward_factor
            corridor_weight = torch.exp(
                -0.5 * (y / self.risk_target_corridor_width).square()
            )
            motion_weight = (
                1.0
                + self.risk_target_approach_scale * torch.tanh(approach_rate)
                + self.risk_target_crossing_scale * torch.tanh(crossing_rate)
            )
            step_discount = self.risk_field_time_discount ** step_idx
            amplitude = (
                step_discount
                * forward_weight
                * corridor_weight
                * motion_weight
                * gathered_visible.squeeze(-1).clamp(0.0, 1.0)
            )
            sigma = self.risk_target_base_sigma + self.risk_target_step_sigma_scale * step_idx
            sigma = sigma + 0.15 * torch.tanh(crossing_rate)
            sigma_sq = sigma.square().unsqueeze(-1).unsqueeze(-1).clamp_min(1e-4)

            dx = self.risk_grid_x - x.unsqueeze(-1).unsqueeze(-1)
            dy = self.risk_grid_y - y.unsqueeze(-1).unsqueeze(-1)
            kernel = torch.exp(-0.5 * (dx.square() + dy.square()) / sigma_sq)

            valid = (
                valid_id[:, :steps, :]
                & (gathered_visible.squeeze(-1) > 0.5)
                & (x.abs() < (self.risk_field_range + 2.0 * sigma))
                & (y.abs() < (self.risk_field_range + 2.0 * sigma))
            )
            kernel = kernel * amplitude.unsqueeze(-1).unsqueeze(-1) * valid.unsqueeze(-1).unsqueeze(-1).to(catalogs.dtype)
            slot_risk = 1.0 - torch.prod(1.0 - kernel.clamp(0.0, 0.999), dim=2)
            current_target = field_target[:, :steps, 0]
            field_target[:, :steps, 0] = 1.0 - (1.0 - current_target) * (1.0 - slot_risk)

        field_target = field_target.clamp(0.0, 1.0)
        tensordict.set("_risk_field_target", field_target)

    def _compute_action_risk_penalty(self, tensordict: TensorDict):
        if self.constrained_rl_enabled or not self.risk_actor_penalty_enabled:
            zero = torch.zeros((), device=self.device)
            return zero, zero, zero, zero, zero

        predicted_risk_field = tensordict["_predicted_risk_field"].detach()
        alpha = tensordict["alpha"]
        beta = tensordict["beta"]
        mean_action_normalized = alpha / (alpha + beta).clamp_min(1e-6)
        local_action = (2.0 * mean_action_normalized * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        action_xy = local_action[..., :2]

        field = predicted_risk_field.reshape(-1, 1, self.risk_field_size, self.risk_field_size)

        def sample_risk_at(probe_xy: torch.Tensor) -> torch.Tensor:
            norm_xy = (probe_xy / self.risk_field_range).clamp(-1.0, 1.0)
            patch_size = self.risk_actor_patch_size
            if patch_size <= 1:
                grid = norm_xy.reshape(-1, 1, 1, 2)
            else:
                half = patch_size // 2
                cell_spacing = 2.0 / max(1, self.risk_field_size - 1)
                offset_axis = torch.linspace(
                    -half,
                    half,
                    steps=patch_size,
                    device=norm_xy.device,
                    dtype=norm_xy.dtype,
                ) * cell_spacing
                offset_y, offset_x = torch.meshgrid(offset_axis, offset_axis, indexing="ij")
                patch_offsets = torch.stack([offset_x, offset_y], dim=-1)
                grid = norm_xy.reshape(-1, 1, 1, 2) + patch_offsets.unsqueeze(0)
                grid = grid.clamp(-1.0, 1.0)
            sampled_risk_patch = F.grid_sample(
                field,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            if self.risk_actor_patch_weighted and patch_size > 1:
                half = patch_size // 2
                patch_axis = torch.linspace(
                    -half,
                    half,
                    steps=patch_size,
                    device=sampled_risk_patch.device,
                    dtype=sampled_risk_patch.dtype,
                )
                patch_y, patch_x = torch.meshgrid(patch_axis, patch_axis, indexing="ij")
                patch_dist2 = patch_x.square() + patch_y.square()
                patch_weights = torch.exp(-0.5 * patch_dist2 / (self.risk_actor_patch_weight_sigma ** 2))
                patch_weights = patch_weights / patch_weights.sum().clamp_min(1e-6)
                sampled_risk = (sampled_risk_patch * patch_weights.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))
                return sampled_risk.reshape(*norm_xy.shape[:-1], 1)
            return sampled_risk_patch.mean(dim=(-1, -2)).reshape(*norm_xy.shape[:-1], 1)

        goal_dir = tensordict[("agents", "observation", "direction")].squeeze(-2)[..., :2]
        goal_dir = F.normalize(goal_dir, dim=-1, eps=1e-6)
        action_dir = action_xy / self.cfg.actor.action_limit
        action_dir_norm = action_dir.norm(dim=-1, keepdim=True)
        unit_action_dir = action_dir / action_dir_norm.clamp_min(1e-6)
        unit_action_dir = torch.where(action_dir_norm > 1e-3, unit_action_dir, goal_dir)
        unit_action_dir = F.normalize(unit_action_dir, dim=-1, eps=1e-6)
        lateral_dir = torch.stack([-unit_action_dir[..., 1], unit_action_dir[..., 0]], dim=-1)

        center_probe_xy = unit_action_dir * self.risk_actor_probe_distance
        center_risk = sample_risk_at(center_probe_xy)
        profile_score = center_risk

        if self.risk_actor_profile_enabled:
            near_probe_xy = unit_action_dir * (self.risk_actor_probe_distance * self.risk_actor_profile_near_scale)
            far_probe_xy = unit_action_dir * (self.risk_actor_probe_distance * self.risk_actor_profile_far_scale)
            lateral_offset = lateral_dir * self.risk_actor_profile_lateral_offset
            left_probe_xy = center_probe_xy + lateral_offset
            right_probe_xy = center_probe_xy - lateral_offset

            near_risk = sample_risk_at(near_probe_xy)
            far_risk = sample_risk_at(far_probe_xy)
            left_risk = sample_risk_at(left_probe_xy)
            right_risk = sample_risk_at(right_probe_xy)

            frontal_risk = torch.maximum(center_risk, far_risk)
            flank_risk = 0.5 * (left_risk + right_risk)
            asymmetry_risk = (left_risk - right_risk).abs()
            profile_score = (
                0.15 * near_risk
                + 0.45 * frontal_risk
                + 0.25 * flank_risk
                + 0.15 * asymmetry_risk
            )

        thresholded_risk = F.relu(profile_score - self.risk_actor_threshold)
        if self.risk_actor_adaptive_enabled:
            adaptive_weight = 1.0 + self.risk_actor_adaptive_scale * profile_score.detach()
        else:
            adaptive_weight = torch.ones_like(profile_score)

        goal_progress_bonus = torch.zeros_like(profile_score)
        if self.risk_actor_goal_progress_enabled:
            action_goal_alignment = (action_dir * goal_dir).sum(dim=-1, keepdim=True)
            safe_gate = (1.0 - thresholded_risk.detach()).clamp(0.0, 1.0)
            goal_progress_bonus = action_goal_alignment.clamp_min(0.0) * safe_gate

        penalty = thresholded_risk * adaptive_weight
        if self.risk_actor_goal_progress_enabled:
            penalty = penalty - self.risk_actor_goal_progress_scale * goal_progress_bonus

        penalty = penalty.mean()
        return (
            penalty,
            adaptive_weight.mean(),
            goal_progress_bonus.mean(),
            profile_score.mean(),
            thresholded_risk.mean(),
        )

    def _compute_cost_critic_risk_weight(self, tensordict: TensorDict, target: torch.Tensor):
        """Return normalized per-sample weights for risk-prioritized cost critic fitting."""
        one = torch.ones_like(target)
        zero = torch.zeros_like(target)
        is_risk_guided_style = self.cost_critic_conditioning_style in {
            "risk_guided_residual",
            "risk_guided_residual_shuffled",
        }
        if is_risk_guided_style:
            alpha = self.cost_critic_risk_priority_coef
            max_weight = self.cost_critic_risk_priority_max
        else:
            alpha = self.cost_critic_risk_weight_alpha
            max_weight = self.cost_critic_risk_weight_max
        if not (self.cost_critic_risk_weighting_enabled or is_risk_guided_style) or alpha <= 0.0:
            return one, zero

        score_source = (
            self.cost_critic_risk_score_source
            if is_risk_guided_style
            else self.cost_critic_risk_weight_source
        )
        score = self._cost_risk_score_for_target(tensordict, target, source=score_source)

        if score is None:
            return one, zero

        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        while score.dim() < target.dim():
            score = score.unsqueeze(-2)
        score = score.expand_as(target)
        raw_score = score
        if self.cost_critic_risk_weight_normalize:
            score = score / score.mean().clamp_min(1e-6)
        weight = 1.0 + alpha * score
        weight = weight.clamp(1.0, max_weight)
        if self.cost_critic_risk_weight_normalize:
            weight = weight / weight.mean().clamp_min(1e-6)
        weight = torch.nan_to_num(weight, nan=1.0, posinf=max_weight, neginf=1.0)
        return weight.detach(), raw_score.detach()

    def _cost_risk_score_for_target(self, tensordict: TensorDict, target: torch.Tensor, source: str | None = None):
        source = source or self.cost_critic_risk_score_source
        score = None
        if source in {"topk", "goal_corridor"}:
            try:
                field = tensordict["_predicted_risk_field"].detach()
                if source == "goal_corridor":
                    score = self._compute_goal_corridor_risk_score(field)
                else:
                    flat = field.flatten(start_dim=-3)
                    k = max(1, int(flat.shape[-1] * self.cost_critic_risk_weight_topk_ratio))
                    score = flat.topk(k, dim=-1).values.mean(dim=-1, keepdim=True)
            except KeyError:
                score = None
        if score is None:
            try:
                summary = tensordict["_risk_summary"].detach()
            except KeyError:
                return torch.zeros_like(target)
            source_to_idx = {
                "summary_mean": 0,
                "mean": 0,
                "summary_max": 1,
                "max": 1,
                "active": 2,
                "energy": 3,
            }
            idx = source_to_idx.get(source, 1)
            score = summary[..., idx:idx + 1]
        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        while score.dim() < target.dim():
            score = score.unsqueeze(-2)
        return score.expand_as(target)

    def _compute_high_risk_cost_metrics(
        self,
        tensordict: TensorDict,
        cost_value: torch.Tensor,
        cost_ret: torch.Tensor,
    ):
        score = self._cost_risk_score_for_target(tensordict, cost_value)
        flat_score = score.reshape(-1)
        flat_value = cost_value.reshape(-1)
        flat_ret = cost_ret.reshape(-1)
        valid = torch.isfinite(flat_score) & torch.isfinite(flat_value) & torch.isfinite(flat_ret)
        if valid.sum() <= 1:
            zero = torch.zeros((), device=cost_value.device)
            return zero, zero, zero

        flat_score = flat_score[valid]
        flat_value = flat_value[valid]
        flat_ret = flat_ret[valid]
        k = max(1, int(flat_score.numel() * self.cost_critic_high_risk_ratio))
        idx = flat_score.topk(k).indices
        high_value = flat_value[idx]
        high_ret = flat_ret[idx]
        high_mae = (high_value - high_ret).abs().mean()
        high_mse = F.mse_loss(high_value, high_ret)
        high_var = high_ret.var(unbiased=False).clamp_min(1e-6)
        high_ev = 1.0 - high_mse / high_var
        high_score = flat_score[idx].mean()
        return torch.nan_to_num(high_ev), torch.nan_to_num(high_mae), torch.nan_to_num(high_score)

    def _compute_risk_value_calibration_loss(self, tensordict: TensorDict, risk_value: torch.Tensor):
        if self.cost_critic_conditioning_style not in {
            "calibrated_residual",
            "calibrated_residual_shuffled",
            "risk_guided_residual",
            "risk_guided_residual_shuffled",
        }:
            zero = torch.zeros((), device=risk_value.device)
            return zero, zero, zero, zero, zero

        score = self._cost_risk_score_for_target(tensordict, risk_value)
        flat_score = score.reshape(-1)
        flat_value = risk_value.reshape(-1)
        valid = torch.isfinite(flat_score) & torch.isfinite(flat_value)
        if valid.sum() <= 2:
            zero = torch.zeros((), device=risk_value.device)
            return zero, zero, zero, zero, zero

        flat_score = flat_score[valid]
        flat_value = flat_value[valid]
        value_centered = flat_value - flat_value.mean()
        score_centered = flat_score - flat_score.mean()
        value_std = value_centered.std(unbiased=False).clamp_min(1e-6)
        score_std = score_centered.std(unbiased=False).clamp_min(1e-6)
        value_z = value_centered / value_std
        score_z = score_centered / score_std

        align_loss = F.mse_loss(value_z, score_z)
        corr = (value_z * score_z).mean().clamp(-1.0, 1.0)
        center_loss = flat_value.mean().square()

        k = max(1, int(flat_score.numel() * self.cost_critic_high_risk_ratio))
        high_idx = flat_score.topk(k).indices
        low_idx = (-flat_score).topk(k).indices
        high_value = flat_value[high_idx]
        low_value = flat_value[low_idx]
        pair_count = min(high_value.numel(), low_value.numel())
        rank_loss = F.softplus(-(high_value[:pair_count] - low_value[:pair_count])).mean()

        total = (
            self.cost_critic_risk_align_coef * align_loss
            + self.cost_critic_risk_rank_coef * rank_loss
            + self.cost_critic_risk_center_coef * center_loss
        )
        return (
            torch.nan_to_num(total),
            torch.nan_to_num(align_loss),
            torch.nan_to_num(rank_loss),
            torch.nan_to_num(center_loss),
            torch.nan_to_num(corr),
        )

    def _compute_risk_guided_residual_loss(
        self,
        tensordict: TensorDict,
        risk_value: torch.Tensor,
        base_value: torch.Tensor | None,
        cost_ret: torch.Tensor,
    ):
        if self.cost_critic_conditioning_style not in {
            "risk_guided_residual",
            "risk_guided_residual_shuffled",
        } or self.cost_critic_risk_residual_coef <= 0.0:
            return torch.zeros((), device=risk_value.device), torch.zeros((), device=risk_value.device)
        if base_value is None:
            return torch.zeros((), device=risk_value.device), torch.zeros((), device=risk_value.device)

        score = self._cost_risk_score_for_target(tensordict, risk_value)
        residual_target = (cost_ret - base_value.detach()).detach()
        residual_loss = F.huber_loss(risk_value, residual_target, delta=10.0, reduction="none")
        score_weight = score / score.mean().clamp_min(1e-6)
        score_weight = score_weight.clamp(0.25, self.cost_critic_risk_priority_max).detach()
        residual_loss = (residual_loss * score_weight).mean()
        total = self.cost_critic_risk_residual_coef * residual_loss
        return torch.nan_to_num(total), torch.nan_to_num(residual_loss)

    def _ppg_aux_update(self, tensordict: TensorDict):
        self.feature_extractor(tensordict)
        if self.risk_prediction_enabled:
            self._predict_future_risk(tensordict)
        self._build_policy_feature(tensordict)

        old_alpha = tensordict["_ppg_old_alpha"].detach().clamp_min(1e-4)
        old_beta = tensordict["_ppg_old_beta"].detach().clamp_min(1e-4)
        old_dist = IndependentBeta(old_alpha, old_beta)
        if self.ppg_detach_actor_feature:
            actor_tensordict = tensordict.clone(False)
            actor_tensordict.set("_policy_feature", tensordict["_policy_feature"].detach())
        else:
            actor_tensordict = tensordict
        new_dist = self.actor.get_dist(actor_tensordict)
        policy_kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

        value = self.critic(tensordict)["state_value"]
        aux_value_loss = self.critic_loss_fn(tensordict["ret"], value).mean()
        aux_loss = self.ppg_aux_value_coef * aux_value_loss + self.ppg_aux_kl_coef * policy_kl
        aux_loss = torch.nan_to_num(aux_loss)

        self.ppg_aux_optim.zero_grad()
        aux_loss.backward()
        ppg_feature_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=5.0)
        ppg_actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        ppg_critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.ppg_aux_optim.step()

        return TensorDict(
            {
                "ppg_aux_loss": aux_loss.detach(),
                "ppg_aux_value_loss": torch.nan_to_num(aux_value_loss.detach()),
                "ppg_aux_kl": torch.nan_to_num(policy_kl.detach()),
                "ppg_feature_grad_norm": torch.nan_to_num(ppg_feature_grad_norm.detach()),
                "ppg_actor_grad_norm": torch.nan_to_num(ppg_actor_grad_norm.detach()),
                "ppg_critic_grad_norm": torch.nan_to_num(ppg_critic_grad_norm.detach()),
            },
            [],
        )

    def train(self, tensordict=True):
        # Keep PyTorch's standard ``train(bool)`` contract for ``eval()`` and
        # leave the PPO update path available as ``train(TensorDict)``.
        if isinstance(tensordict, bool):
            nn.Module.train(self, tensordict)
            return self
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        if self.risk_prediction_enabled:
            self._build_risk_prediction_targets(tensordict)
        next_tensordict = tensordict["next"]
        risk_field_cost = None
        with torch.no_grad():
            next_batch_size = next_tensordict.batch_size
            if self.obstacle_input_mode in {"current_gru", "current_temporal", "current_temporal_conv", "current_slot_residual", "current_motion_descriptor"}:
                self.feature_extractor(next_tensordict)
                if self.risk_prediction_enabled:
                    self._predict_future_risk(next_tensordict)
                self._build_policy_feature(next_tensordict)
                next_values = self.critic(next_tensordict)["state_value"]
                next_cost_values = self.cost_critic(next_tensordict)["cost_value"] if self.cost_critic is not None else None
                risk_field_cost = self._compute_risk_field_cost(next_tensordict)
            else:
                next_tensordict_flat = next_tensordict.reshape(-1)
                self.feature_extractor(next_tensordict_flat) # calculate features for next state value calculation
                if self.risk_prediction_enabled:
                    self._predict_future_risk(next_tensordict_flat)
                self._build_policy_feature(next_tensordict_flat)
                next_values = self.critic(next_tensordict_flat)["state_value"].view(*next_batch_size, 1)
                next_cost_values = self.cost_critic(next_tensordict_flat)["cost_value"].view(*next_batch_size, 1) if self.cost_critic is not None else None
                risk_field_cost = self._compute_risk_field_cost(next_tensordict_flat, next_batch_size)
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        env_costs = tensordict["next", "agents", "cost"] if self.constrained_rl_enabled else None
        costs = self._combine_constrained_cost(env_costs, risk_field_cost)
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

        state_value_finite = torch.isfinite(tensordict["state_value"]).float().mean()
        next_value_finite = torch.isfinite(next_values).float().mean()
        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)
        rewards = torch.nan_to_num(rewards)
        values = torch.nan_to_num(values)
        next_values = torch.nan_to_num(next_values)
        cost_mean = torch.zeros((), device=self.device)
        cost_std = torch.zeros((), device=self.device)
        cost_ret_mean = torch.zeros((), device=self.device)
        cost_ret_std = torch.zeros((), device=self.device)
        cost_adv_std = torch.zeros((), device=self.device)
        cost_value_mean = torch.zeros((), device=self.device)
        cost_value_std = torch.zeros((), device=self.device)
        cost_value_finite = torch.ones((), device=self.device)
        next_cost_value_finite = torch.ones((), device=self.device)
        env_cost_mean = env_costs.mean() if env_costs is not None else torch.zeros((), device=self.device)
        risk_field_cost_mean = risk_field_cost.mean() if risk_field_cost is not None else torch.zeros((), device=self.device)
        risk_field_cost_std = risk_field_cost.std() if risk_field_cost is not None else torch.zeros((), device=self.device)

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = torch.nan_to_num(adv)
        ret = torch.nan_to_num(ret)
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        ret_mean = ret.mean()
        ret_std = ret.std()
        value_mean = values.mean()
        value_std = values.std()
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clamp_min(1e-6)
        adv = torch.nan_to_num(adv)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        ret = torch.nan_to_num(ret)
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        if self.ppg_enabled:
            # PPG auxiliary phase keeps the post-policy-phase actor close to the
            # rollout policy while improving the shared value representation.
            tensordict.set("_ppg_old_alpha", tensordict["alpha"].detach().clone())
            tensordict.set("_ppg_old_beta", tensordict["beta"].detach().clone())
        if self.constrained_rl_enabled and costs is not None and self.cost_critic is not None:
            cost_values = self.cost_value_norm.denormalize(tensordict["cost_value"])
            next_cost_values = self.cost_value_norm.denormalize(next_cost_values)
            costs = torch.nan_to_num(costs)
            cost_values = torch.nan_to_num(cost_values)
            next_cost_values = torch.nan_to_num(next_cost_values)
            cost_adv, cost_ret = self.cost_gae(costs, dones, cost_values, next_cost_values)
            cost_adv = torch.nan_to_num(cost_adv)
            cost_ret = torch.nan_to_num(cost_ret)
            cost_mean = costs.mean()
            cost_std = costs.std()
            cost_ret_mean = cost_ret.mean()
            cost_ret_std = cost_ret.std()
            cost_value_mean = cost_values.mean()
            cost_value_std = cost_values.std()
            cost_adv_std = cost_adv.std()
            cost_adv = (cost_adv - cost_adv.mean()) / cost_adv.std().clamp_min(1e-6)
            cost_adv = torch.nan_to_num(cost_adv)
            self.cost_value_norm.update(cost_ret)
            cost_ret = self.cost_value_norm.normalize(cost_ret)
            cost_ret = torch.nan_to_num(cost_ret)
            tensordict.set("cost_adv", cost_adv)
            tensordict.set("cost_ret", cost_ret)
            if self.p3o_enabled:
                p3o_cost_violation = (cost_mean - self.cost_budget) * self.p3o_violation_scale
                tensordict.set("p3o_cost_violation", p3o_cost_violation.expand_as(cost_adv))
            cost_value_finite = torch.isfinite(tensordict["cost_value"]).float().mean()
            next_cost_value_finite = torch.isfinite(next_cost_values).float().mean()
            if not self.p3o_enabled:
                self._update_lagrange_multiplier(cost_mean)

        # Training
        infos = []
        for epoch in range(self.training_epoch_num):
            if self.obstacle_input_mode in {"current_gru", "current_temporal", "current_temporal_conv", "current_slot_residual", "current_motion_descriptor"}:
                batch = make_sequence_batch(tensordict, self.cfg.num_minibatches)
            else:
                batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict()

        infos = infos.apply(torch.mean, batch_size=[])
        info = {k: v.item() for k, v in infos.items()}
        ppg_info = {
            "ppg_enabled": float(self.ppg_enabled),
            "ppg_aux_loss": 0.0,
            "ppg_aux_value_loss": 0.0,
            "ppg_aux_kl": 0.0,
            "ppg_feature_grad_norm": 0.0,
            "ppg_actor_grad_norm": 0.0,
            "ppg_critic_grad_norm": 0.0,
        }
        if (
            self.ppg_enabled
            and self.ppg_aux_optim is not None
            and self.ppg_aux_epoch_num > 0
            and self.ppg_update_counter % self.ppg_aux_interval == 0
        ):
            ppg_infos = []
            for _ in range(self.ppg_aux_epoch_num):
                if self.obstacle_input_mode in {"current_gru", "current_temporal", "current_temporal_conv", "current_slot_residual", "current_motion_descriptor"}:
                    aux_batch = make_sequence_batch(tensordict, self.ppg_aux_num_minibatches)
                else:
                    aux_batch = make_batch(tensordict, self.ppg_aux_num_minibatches)
                for minibatch in aux_batch:
                    ppg_infos.append(self._ppg_aux_update(minibatch))
            if ppg_infos:
                ppg_infos = torch.stack(ppg_infos).to_tensordict().apply(torch.mean, batch_size=[])
                ppg_info.update({k: v.item() for k, v in ppg_infos.items()})
        self.ppg_update_counter += 1
        info.update(ppg_info)
        info.update({
            "reward_mean": reward_mean.item(),
            "reward_std": reward_std.item(),
            "ret_mean": ret_mean.item(),
            "ret_std": ret_std.item(),
            "value_mean": value_mean.item(),
            "value_std": value_std.item(),
            "adv_std": adv_std.item(),
            "method/step1_enabled": float(self.method_step1_enabled),
            "method/step2_enabled": float(self.method_step2_enabled),
            "method/step3_enabled": float(self.method_step3_enabled),
            "method/variant_code": self.method_variant_code,
            "cost_mean": cost_mean.item(),
            "cost_std": cost_std.item(),
            "env_cost_mean": env_cost_mean.item(),
            "risk_field_cost_mean": risk_field_cost_mean.item(),
            "risk_field_cost_std": risk_field_cost_std.item(),
            "risk_cost_weight": self.risk_cost_weight if self.constrained_rl_enabled else 0.0,
            "risk_cost_mode_code": {"topk": 0.0, "goal_corridor": 1.0}.get(self.risk_cost_mode, 0.0),
            "risk_cost_topk_ratio": self.risk_cost_topk_ratio if self.constrained_rl_enabled else 0.0,
            "risk_cost_threshold": self.risk_cost_threshold if self.constrained_rl_enabled else 0.0,
            "risk_cost_scale": self.risk_cost_scale if self.constrained_rl_enabled else 0.0,
            "risk_cost_shuffle_enabled": float(self.risk_cost_shuffle_enabled) if self.constrained_rl_enabled else 0.0,
            "risk_cost_corridor_distance": self.risk_cost_corridor_distance if self.constrained_rl_enabled else 0.0,
            "risk_cost_corridor_lateral_offset": self.risk_cost_corridor_lateral_offset if self.constrained_rl_enabled else 0.0,
            "cost_source_code": {"env": 0.0, "risk_field": 1.0, "hybrid": 2.0}.get(self.cost_source, 0.0),
            "cost_critic_conditioning_code": {
                "no_cond_same_dim": 0.0,
                "summary": 1.0,
                "embedding": 2.0,
                "full": 3.0,
            }.get(self.cost_critic_conditioning, 3.0),
            "cost_critic_conditioning_style_code": {
                "concat": 0.0,
                "gated": 1.0,
                "residual": 2.0,
                "residual_shuffled": 3.0,
                "calibrated_residual": 4.0,
                "calibrated_residual_shuffled": 5.0,
                "risk_guided_residual": 6.0,
                "risk_guided_residual_shuffled": 7.0,
            }.get(self.cost_critic_conditioning_style, 0.0),
            "cost_critic_high_risk_ratio": self.cost_critic_high_risk_ratio if self.constrained_rl_enabled else 0.0,
            "cost_critic_risk_score_source_code": {
                "summary_max": 0.0,
                "topk": 1.0,
                "goal_corridor": 2.0,
            }.get(self.cost_critic_risk_score_source, 0.0),
            "cost_critic_risk_priority_coef": self.cost_critic_risk_priority_coef if self.constrained_rl_enabled else 0.0,
            "cost_critic_risk_priority_max": self.cost_critic_risk_priority_max if self.constrained_rl_enabled else 0.0,
            "cost_critic_risk_residual_coef": self.cost_critic_risk_residual_coef if self.constrained_rl_enabled else 0.0,
            "cost_critic_risk_align_coef": self.cost_critic_risk_align_coef if self.constrained_rl_enabled else 0.0,
            "cost_critic_risk_rank_coef": self.cost_critic_risk_rank_coef if self.constrained_rl_enabled else 0.0,
            "cost_critic_risk_center_coef": self.cost_critic_risk_center_coef if self.constrained_rl_enabled else 0.0,
            "cost_critic_risk_weighting_enabled": float(self.cost_critic_risk_weighting_enabled),
            "cost_critic_risk_weight_alpha": self.cost_critic_risk_weight_alpha,
            "cost_critic_risk_weight_source_code": {
                "mean": 0.0,
                "max": 1.0,
                "active": 2.0,
                "energy": 3.0,
                "topk": 4.0,
            }.get(self.cost_critic_risk_weight_source, 1.0),
            "cost_critic_risk_weight_mean": infos.get("cost_critic_risk_weight_mean", torch.tensor(1.0, device=self.device)).item(),
            "cost_critic_risk_weight_std": infos.get("cost_critic_risk_weight_std", torch.tensor(0.0, device=self.device)).item(),
            "cost_critic_risk_weight_max": infos.get("cost_critic_risk_weight_max", torch.tensor(1.0, device=self.device)).item(),
            "cost_critic_risk_score_mean": infos.get("cost_critic_risk_score_mean", torch.tensor(0.0, device=self.device)).item(),
            "cost_high_risk_explained_var": infos.get("cost_high_risk_explained_var", torch.tensor(0.0, device=self.device)).item(),
            "cost_high_risk_mae": infos.get("cost_high_risk_mae", torch.tensor(0.0, device=self.device)).item(),
            "cost_high_risk_score_mean": infos.get("cost_high_risk_score_mean", torch.tensor(0.0, device=self.device)).item(),
            "cost_base_value_mean": infos.get("cost_base_value_mean", torch.tensor(0.0, device=self.device)).item(),
            "cost_base_value_std": infos.get("cost_base_value_std", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_value_mean": infos.get("cost_risk_value_mean", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_value_std": infos.get("cost_risk_value_std", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_value_abs_mean": infos.get("cost_risk_value_abs_mean", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_calibration_loss": infos.get("cost_risk_calibration_loss", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_residual_loss": infos.get("cost_risk_residual_loss", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_align_loss": infos.get("cost_risk_align_loss", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_rank_loss": infos.get("cost_risk_rank_loss", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_center_loss": infos.get("cost_risk_center_loss", torch.tensor(0.0, device=self.device)).item(),
            "cost_risk_score_value_corr": infos.get("cost_risk_score_value_corr", torch.tensor(0.0, device=self.device)).item(),
            "cost_ret_mean": cost_ret_mean.item(),
            "cost_ret_std": cost_ret_std.item(),
            "cost_value_mean": cost_value_mean.item(),
            "cost_value_std": cost_value_std.item(),
            "cost_adv_std": cost_adv_std.item(),
            "state_value_finite": state_value_finite.item(),
            "next_value_finite": next_value_finite.item(),
            "cost_value_finite": cost_value_finite.item(),
            "next_cost_value_finite": next_cost_value_finite.item(),
            "lagrange_multiplier": self.lagrange_multiplier.item() if self.constrained_rl_enabled else 0.0,
            "p3o_enabled": float(self.p3o_enabled),
            "p3o_penalty_coef": self.p3o_penalty_coef if self.p3o_enabled else 0.0,
            "risk_field_loss": infos.get("risk_field_loss", torch.tensor(0.0, device=self.device)).item(),
            "risk_attention_entropy": infos.get("risk_attention_entropy", torch.tensor(0.0, device=self.device)).item(),
            "risk_attention_max": infos.get("risk_attention_max", torch.tensor(0.0, device=self.device)).item(),
            "cost_critic_loss": infos.get("cost_critic_loss", torch.tensor(0.0, device=self.device)).item(),
            "cost_explained_var": infos.get("cost_explained_var", torch.tensor(0.0, device=self.device)).item(),
            "predicted_risk_field_mean": infos.get("predicted_risk_field_mean", torch.tensor(0.0, device=self.device)).item(),
            "target_risk_field_mean": infos.get("target_risk_field_mean", torch.tensor(0.0, device=self.device)).item(),
            "target_risk_field_nonzero_ratio": infos.get("target_risk_field_nonzero_ratio", torch.tensor(0.0, device=self.device)).item(),
            "risk_actor_penalty": infos.get("risk_actor_penalty", torch.tensor(0.0, device=self.device)).item(),
            "risk_actor_adaptive_weight": infos.get("risk_actor_adaptive_weight", torch.tensor(0.0, device=self.device)).item(),
            "risk_actor_goal_progress_bonus": infos.get("risk_actor_goal_progress_bonus", torch.tensor(0.0, device=self.device)).item(),
            "risk_actor_profile_mean": infos.get("risk_actor_profile_mean", torch.tensor(0.0, device=self.device)).item(),
            "risk_actor_thresholded_mean": infos.get("risk_actor_thresholded_mean", torch.tensor(0.0, device=self.device)).item(),
            "risk_query_profile_mean": infos.get("risk_query_profile_mean", torch.tensor(0.0, device=self.device)).item(),
            "nominal_action_mag": infos.get("nominal_action_mag", torch.tensor(0.0, device=self.device)).item(),
            "risk_aware_action_mag": infos.get("risk_aware_action_mag", torch.tensor(0.0, device=self.device)).item(),
            "refined_action_mag": infos.get("refined_action_mag", torch.tensor(0.0, device=self.device)).item(),
            "action_update_mean": infos.get("action_update_mean", torch.tensor(0.0, device=self.device)).item(),
        })
        if self.constrained_rl_enabled:
            info["rf3_stage"] = 2.7
        elif self.risk_action_refinement_enabled:
            info["rf3_stage"] = 3.6
        elif self.risk_prediction_enabled and self.risk_policy_fusion_enabled and self.risk_actor_penalty_enabled:
            if self.risk_actor_profile_enabled:
                info["rf3_stage"] = 3.8
            elif self.risk_policy_use_summary or self.risk_actor_patch_weighted or self.risk_actor_adaptive_enabled:
                info["rf3_stage"] = 3.5
            else:
                info["rf3_stage"] = 3.0
        elif self.risk_prediction_enabled:
            info["rf3_stage"] = 2.0
        else:
            info["rf3_stage"] = 1.0
        return info

    def _update(self, tensordict): # tensordict shape (batch_size, )
        self.feature_extractor(tensordict)
        if self.risk_prediction_enabled:
            self._predict_future_risk(tensordict)
        self._build_policy_feature(tensordict)
        # Get action from the current policy
        rpo_perturb_abs = torch.zeros((), device=self.device)
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get distribution parameters
        if self.rpo_enabled and self.rpo_alpha > 0.0:
            alpha = tensordict["alpha"]
            beta = tensordict["beta"]
            concentration = (alpha + beta).clamp_min(1e-4)
            mean = (alpha / concentration).clamp(1e-4, 1.0 - 1e-4)
            perturb = torch.empty_like(mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            perturbed_mean = (mean + perturb).clamp(1e-4, 1.0 - 1e-4)
            perturbed_alpha = (perturbed_mean * concentration).clamp_min(1e-4)
            perturbed_beta = ((1.0 - perturbed_mean) * concentration).clamp_min(1e-4)
            action_dist = IndependentBeta(perturbed_alpha, perturbed_beta)
            rpo_perturb_abs = perturb.abs().mean()
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss
        advantage = tensordict["adv"] # the advantage is calculated based on GAE in hte previous step
        if self.constrained_rl_enabled and not self.p3o_enabled and "cost_adv" in tensordict.keys(True, True):
            advantage = advantage - self.lagrange_multiplier.detach() * tensordict["cost_adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        reward_surrogate = torch.mean(torch.min(surr1, surr2))
        p3o_cost_surrogate = torch.zeros((), device=self.device)
        p3o_constraint_violation = torch.zeros((), device=self.device)
        p3o_penalty = torch.zeros((), device=self.device)
        if self.p3o_enabled and "cost_adv" in tensordict.keys(True, True):
            cost_advantage = tensordict["cost_adv"]
            cost_surr1 = cost_advantage * ratio
            cost_surr2 = cost_advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
            p3o_cost_surrogate = torch.mean(torch.max(cost_surr1, cost_surr2))
            if "p3o_cost_violation" in tensordict.keys(True, True):
                p3o_constraint_violation = tensordict["p3o_cost_violation"].mean()
            p3o_penalty = self.p3o_penalty_coef * F.relu(p3o_cost_surrogate + p3o_constraint_violation)
        actor_loss = -reward_surrogate * self.action_dim + p3o_penalty
        (
            risk_actor_penalty,
            risk_actor_adaptive_weight,
            risk_actor_goal_progress_bonus,
            risk_actor_profile_mean,
            risk_actor_thresholded_mean,
        ) = self._compute_action_risk_penalty(tensordict)

        # Critic Loss
        b_value = tensordict["state_value"]
        ret = tensordict["ret"] # Return G
        value = self.critic(tensordict)["state_value"]
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        cost_critic_loss = torch.zeros((), device=self.device)
        cost_explained_var = torch.zeros((), device=self.device)
        cost_critic_risk_score_mean = torch.zeros((), device=self.device)
        cost_critic_risk_weight_mean = torch.ones((), device=self.device)
        cost_critic_risk_weight_std = torch.zeros((), device=self.device)
        cost_critic_risk_weight_max = torch.ones((), device=self.device)
        cost_high_risk_explained_var = torch.zeros((), device=self.device)
        cost_high_risk_mae = torch.zeros((), device=self.device)
        cost_high_risk_score_mean = torch.zeros((), device=self.device)
        cost_base_value_mean = torch.zeros((), device=self.device)
        cost_base_value_std = torch.zeros((), device=self.device)
        cost_risk_value_mean = torch.zeros((), device=self.device)
        cost_risk_value_std = torch.zeros((), device=self.device)
        cost_risk_value_abs_mean = torch.zeros((), device=self.device)
        cost_risk_calibration_loss = torch.zeros((), device=self.device)
        cost_risk_residual_loss = torch.zeros((), device=self.device)
        cost_risk_align_loss = torch.zeros((), device=self.device)
        cost_risk_rank_loss = torch.zeros((), device=self.device)
        cost_risk_center_loss = torch.zeros((), device=self.device)
        cost_risk_score_value_corr = torch.zeros((), device=self.device)
        if self.constrained_rl_enabled and self.cost_critic is not None and "cost_ret" in tensordict.keys(True, True):
            b_cost_value = tensordict["cost_value"]
            cost_ret = tensordict["cost_ret"]
            cost_tensordict = self.cost_critic(tensordict)
            cost_value = cost_tensordict["cost_value"]
            cost_value_clipped = b_cost_value + (cost_value - b_cost_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio)
            cost_critic_loss_clipped = F.huber_loss(cost_value_clipped, cost_ret, delta=10.0, reduction="none")
            cost_critic_loss_original = F.huber_loss(cost_value, cost_ret, delta=10.0, reduction="none")
            cost_critic_loss_per_sample = torch.max(cost_critic_loss_clipped, cost_critic_loss_original)
            risk_weight, risk_score = self._compute_cost_critic_risk_weight(tensordict, cost_critic_loss_per_sample)
            cost_critic_loss = (cost_critic_loss_per_sample * risk_weight).mean()
            cost_explained_var = 1 - F.mse_loss(cost_value, cost_ret) / cost_ret.var().clamp_min(1e-6)
            cost_high_risk_explained_var, cost_high_risk_mae, cost_high_risk_score_mean = (
                self._compute_high_risk_cost_metrics(tensordict, cost_value, cost_ret)
            )
            cost_critic_risk_score_mean = risk_score.mean()
            cost_critic_risk_weight_mean = risk_weight.mean()
            cost_critic_risk_weight_std = risk_weight.std()
            cost_critic_risk_weight_max = risk_weight.max()
            if "cost_base_value" in cost_tensordict.keys(True, True):
                cost_base_value = cost_tensordict["cost_base_value"]
                cost_base_value_mean = cost_base_value.mean()
                cost_base_value_std = cost_base_value.std()
            if "cost_risk_value" in cost_tensordict.keys(True, True):
                cost_risk_value = cost_tensordict["cost_risk_value"]
                cost_risk_value_mean = cost_risk_value.mean()
                cost_risk_value_std = cost_risk_value.std()
                cost_risk_value_abs_mean = cost_risk_value.abs().mean()
                (
                    cost_risk_residual_total,
                    cost_risk_residual_loss,
                ) = self._compute_risk_guided_residual_loss(
                    tensordict,
                    cost_risk_value,
                    cost_tensordict.get("cost_base_value"),
                    cost_ret,
                )
                (
                    cost_risk_calibration_loss,
                    cost_risk_align_loss,
                    cost_risk_rank_loss,
                    cost_risk_center_loss,
                    cost_risk_score_value_corr,
                ) = self._compute_risk_value_calibration_loss(tensordict, cost_risk_value)
                cost_critic_loss = cost_critic_loss + cost_risk_calibration_loss + cost_risk_residual_total

        risk_field_loss = torch.zeros((), device=self.device)
        if self.risk_prediction_enabled:
            predicted_risk_field = tensordict["_predicted_risk_field"]
            risk_field_logit = tensordict["_predicted_risk_field_logit"]
            risk_field_target = tensordict["_risk_field_target"]
            risk_field_loss = F.binary_cross_entropy_with_logits(risk_field_logit, risk_field_target)

        risk_attention = tensordict["_risk_attention"]
        risk_attention_entropy = -(risk_attention * torch.log(risk_attention.clamp_min(1e-6))).sum(dim=-1)
        risk_attention_entropy = risk_attention_entropy.mean()
        risk_attention_max = risk_attention.max(dim=-1).values.mean()
        predicted_risk_field_mean = predicted_risk_field.mean() if self.risk_prediction_enabled else torch.zeros((), device=self.device)
        target_risk_field_mean = risk_field_target.mean() if self.risk_prediction_enabled else torch.zeros((), device=self.device)
        target_risk_field_nonzero_ratio = (risk_field_target > 0).float().mean() if self.risk_prediction_enabled else torch.zeros((), device=self.device)
        if "_risk_query_profile" in tensordict.keys(True, True):
            risk_query_profile_mean = tensordict["_risk_query_profile"].mean()
            nominal_action_mag = tensordict["_nominal_action"].norm(dim=-1).mean()
            risk_aware_action_mag = (
                tensordict["_risk_aware_action"].norm(dim=-1).mean()
                if "_risk_aware_action" in tensordict.keys(True, True)
                else torch.zeros((), device=self.device)
            )
            refined_action_mag = tensordict["_refined_action"].norm(dim=-1).mean()
            action_update_mean = (
                tensordict["_action_update"].mean()
                if "_action_update" in tensordict.keys(True, True)
                else torch.zeros((), device=self.device)
            )
        else:
            risk_query_profile_mean = torch.zeros((), device=self.device)
            nominal_action_mag = torch.zeros((), device=self.device)
            risk_aware_action_mag = torch.zeros((), device=self.device)
            refined_action_mag = torch.zeros((), device=self.device)
            action_update_mean = torch.zeros((), device=self.device)

        # Total Loss
        actor_loss = torch.nan_to_num(actor_loss)
        critic_loss = torch.nan_to_num(critic_loss)
        cost_critic_loss = torch.nan_to_num(cost_critic_loss)
        cost_explained_var = torch.nan_to_num(cost_explained_var)
        cost_critic_risk_score_mean = torch.nan_to_num(cost_critic_risk_score_mean)
        cost_critic_risk_weight_mean = torch.nan_to_num(cost_critic_risk_weight_mean)
        cost_critic_risk_weight_std = torch.nan_to_num(cost_critic_risk_weight_std)
        cost_critic_risk_weight_max = torch.nan_to_num(cost_critic_risk_weight_max)
        cost_high_risk_explained_var = torch.nan_to_num(cost_high_risk_explained_var)
        cost_high_risk_mae = torch.nan_to_num(cost_high_risk_mae)
        cost_high_risk_score_mean = torch.nan_to_num(cost_high_risk_score_mean)
        cost_base_value_mean = torch.nan_to_num(cost_base_value_mean)
        cost_base_value_std = torch.nan_to_num(cost_base_value_std)
        cost_risk_value_mean = torch.nan_to_num(cost_risk_value_mean)
        cost_risk_value_std = torch.nan_to_num(cost_risk_value_std)
        cost_risk_value_abs_mean = torch.nan_to_num(cost_risk_value_abs_mean)
        cost_risk_calibration_loss = torch.nan_to_num(cost_risk_calibration_loss)
        cost_risk_residual_loss = torch.nan_to_num(cost_risk_residual_loss)
        cost_risk_align_loss = torch.nan_to_num(cost_risk_align_loss)
        cost_risk_rank_loss = torch.nan_to_num(cost_risk_rank_loss)
        cost_risk_center_loss = torch.nan_to_num(cost_risk_center_loss)
        cost_risk_score_value_corr = torch.nan_to_num(cost_risk_score_value_corr)
        entropy_loss = torch.nan_to_num(entropy_loss)
        risk_field_loss = torch.nan_to_num(risk_field_loss)
        risk_attention_entropy = torch.nan_to_num(risk_attention_entropy)
        risk_attention_max = torch.nan_to_num(risk_attention_max)
        predicted_risk_field_mean = torch.nan_to_num(predicted_risk_field_mean)
        target_risk_field_mean = torch.nan_to_num(target_risk_field_mean)
        target_risk_field_nonzero_ratio = torch.nan_to_num(target_risk_field_nonzero_ratio)
        risk_actor_penalty = torch.nan_to_num(risk_actor_penalty)
        risk_actor_adaptive_weight = torch.nan_to_num(risk_actor_adaptive_weight)
        risk_actor_goal_progress_bonus = torch.nan_to_num(risk_actor_goal_progress_bonus)
        risk_actor_profile_mean = torch.nan_to_num(risk_actor_profile_mean)
        risk_actor_thresholded_mean = torch.nan_to_num(risk_actor_thresholded_mean)
        risk_query_profile_mean = torch.nan_to_num(risk_query_profile_mean)
        nominal_action_mag = torch.nan_to_num(nominal_action_mag)
        risk_aware_action_mag = torch.nan_to_num(risk_aware_action_mag)
        refined_action_mag = torch.nan_to_num(refined_action_mag)
        action_update_mean = torch.nan_to_num(action_update_mean)
        rpo_perturb_abs = torch.nan_to_num(rpo_perturb_abs)
        reward_surrogate = torch.nan_to_num(reward_surrogate)
        p3o_cost_surrogate = torch.nan_to_num(p3o_cost_surrogate)
        p3o_constraint_violation = torch.nan_to_num(p3o_constraint_violation)
        p3o_penalty = torch.nan_to_num(p3o_penalty)
        loss = (
            entropy_loss
            + actor_loss
            + critic_loss
            + self.cost_critic_coef * cost_critic_loss
            + self.risk_prediction_loss_coef * risk_field_loss
            + self.risk_actor_penalty_coef * risk_actor_penalty
        )

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        if self.cost_critic_optim is not None:
            self.cost_critic_optim.zero_grad()
        if self.risk_prediction_optim is not None:
            self.risk_prediction_optim.zero_grad()
        loss.backward()

        feature_grad_clip = 1.0 if (self.history_encoder_type == "lstm" or self.obstacle_input_mode in {"current_gru", "current_temporal", "current_temporal_conv", "current_slot_residual", "current_motion_descriptor"}) else 5.0
        feature_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=feature_grad_clip)
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        cost_critic_grad_norm = torch.zeros((), device=self.device)
        if self.cost_critic is not None:
            cost_critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.cost_critic.parameters(), max_norm=5.)
        risk_prediction_grad_norm = torch.zeros((), device=self.device)
        if self.risk_prediction_optim is not None:
            risk_prediction_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.risk_predictor.parameters(), max_norm=5.0)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        if self.cost_critic_optim is not None:
            self.cost_critic_optim.step()
        if self.risk_prediction_optim is not None:
            self.risk_prediction_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var().clamp_min(1e-6)
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "cost_critic_loss": cost_critic_loss,
            "cost_critic_risk_score_mean": cost_critic_risk_score_mean,
            "cost_critic_risk_weight_mean": cost_critic_risk_weight_mean,
            "cost_critic_risk_weight_std": cost_critic_risk_weight_std,
            "cost_critic_risk_weight_max": cost_critic_risk_weight_max,
            "cost_high_risk_explained_var": cost_high_risk_explained_var,
            "cost_high_risk_mae": cost_high_risk_mae,
            "cost_high_risk_score_mean": cost_high_risk_score_mean,
            "cost_base_value_mean": cost_base_value_mean,
            "cost_base_value_std": cost_base_value_std,
            "cost_risk_value_mean": cost_risk_value_mean,
            "cost_risk_value_std": cost_risk_value_std,
            "cost_risk_value_abs_mean": cost_risk_value_abs_mean,
            "cost_risk_calibration_loss": cost_risk_calibration_loss,
            "cost_risk_residual_loss": cost_risk_residual_loss,
            "cost_risk_align_loss": cost_risk_align_loss,
            "cost_risk_rank_loss": cost_risk_rank_loss,
            "cost_risk_center_loss": cost_risk_center_loss,
            "cost_risk_score_value_corr": cost_risk_score_value_corr,
            "entropy": entropy_loss,
            "risk_field_loss": risk_field_loss,
            "risk_attention_entropy": risk_attention_entropy,
            "risk_attention_max": risk_attention_max,
            "cost_explained_var": cost_explained_var,
            "predicted_risk_field_mean": predicted_risk_field_mean,
            "target_risk_field_mean": target_risk_field_mean,
            "target_risk_field_nonzero_ratio": target_risk_field_nonzero_ratio,
            "risk_actor_penalty": risk_actor_penalty,
            "risk_actor_adaptive_weight": risk_actor_adaptive_weight,
            "risk_actor_goal_progress_bonus": risk_actor_goal_progress_bonus,
            "risk_actor_profile_mean": risk_actor_profile_mean,
            "risk_actor_thresholded_mean": risk_actor_thresholded_mean,
            "risk_query_profile_mean": risk_query_profile_mean,
            "nominal_action_mag": nominal_action_mag,
            "risk_aware_action_mag": risk_aware_action_mag,
            "refined_action_mag": refined_action_mag,
            "action_update_mean": action_update_mean,
            "rpo_enabled": torch.tensor(float(self.rpo_enabled), device=self.device),
            "rpo_perturb_abs": rpo_perturb_abs,
            "p3o_enabled": torch.tensor(float(self.p3o_enabled), device=self.device),
            "p3o_reward_surrogate": reward_surrogate,
            "p3o_cost_surrogate": p3o_cost_surrogate,
            "p3o_constraint_violation": p3o_constraint_violation,
            "p3o_penalty": p3o_penalty,
            "rf3_stage": torch.tensor(-1.0, device=self.device),
            "feature_grad_norm": feature_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "cost_critic_grad_norm": cost_critic_grad_norm,
            "risk_prediction_grad_norm": risk_prediction_grad_norm,
            "explained_var": explained_var
        }, [])
