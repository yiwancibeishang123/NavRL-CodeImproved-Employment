#!/usr/bin/env python3
"""Reproducible observation perturbations for sim-to-sim robustness tests."""

from collections import deque

import torch


class SensorPerturber:
    def __init__(
        self,
        enabled=False,
        seed=0,
        lidar_noise_std=0.0,
        lidar_dropout_ratio=0.0,
        obstacle_position_noise_std=0.0,
        obstacle_velocity_noise_std=0.0,
        obstacle_dropout_ratio=0.0,
        state_noise_std=0.0,
        delay_steps=0,
    ):
        self.enabled = bool(enabled)
        self.lidar_noise_std = max(0.0, float(lidar_noise_std))
        self.lidar_dropout_ratio = min(1.0, max(0.0, float(lidar_dropout_ratio)))
        self.obstacle_position_noise_std = max(0.0, float(obstacle_position_noise_std))
        self.obstacle_velocity_noise_std = max(0.0, float(obstacle_velocity_noise_std))
        self.obstacle_dropout_ratio = min(1.0, max(0.0, float(obstacle_dropout_ratio)))
        self.state_noise_std = max(0.0, float(state_noise_std))
        self.delay_steps = max(0, int(delay_steps))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.history = deque(maxlen=self.delay_steps + 1)

    @classmethod
    def from_ros(cls, rospy):
        return cls(
            enabled=rospy.get_param("~sim_real_perturbation_enabled", False),
            seed=rospy.get_param("~sim_real_perturbation_seed", 7),
            lidar_noise_std=rospy.get_param("~lidar_noise_std", 0.0),
            lidar_dropout_ratio=rospy.get_param("~lidar_dropout_ratio", 0.0),
            obstacle_position_noise_std=rospy.get_param("~obstacle_position_noise_std", 0.0),
            obstacle_velocity_noise_std=rospy.get_param("~obstacle_velocity_noise_std", 0.0),
            obstacle_dropout_ratio=rospy.get_param("~obstacle_dropout_ratio", 0.0),
            state_noise_std=rospy.get_param("~state_noise_std", 0.0),
            delay_steps=rospy.get_param("~observation_delay_steps", 0),
        )

    def _noise_like(self, tensor, std):
        if std <= 0.0:
            return tensor
        noise = torch.randn(tensor.shape, generator=self.generator, dtype=tensor.dtype, device="cpu")
        return tensor + noise.to(tensor.device) * std

    def apply(self, state, lidar, dynamic_obstacle):
        """Return perturbed tensors and metadata; input tensors are not mutated."""
        if not self.enabled:
            return state, lidar, dynamic_obstacle, {"enabled": False, "delay_steps": 0}

        out_state = self._noise_like(state, self.state_noise_std)
        out_lidar = self._noise_like(lidar, self.lidar_noise_std).clamp_min(0.0)
        out_dynamic = dynamic_obstacle.clone()
        out_dynamic[..., :3] = self._noise_like(out_dynamic[..., :3], self.obstacle_position_noise_std)
        out_dynamic[..., 5:8] = self._noise_like(out_dynamic[..., 5:8], self.obstacle_velocity_noise_std)
        if self.lidar_dropout_ratio > 0.0:
            mask = torch.rand(out_lidar.shape, generator=self.generator, device="cpu").to(out_lidar.device)
            out_lidar = torch.where(mask < self.lidar_dropout_ratio, torch.zeros_like(out_lidar), out_lidar)
        if self.obstacle_dropout_ratio > 0.0:
            slots = out_dynamic.shape[-2]
            mask = torch.rand((slots,), generator=self.generator, device="cpu").to(out_dynamic.device)
            out_dynamic[..., mask < self.obstacle_dropout_ratio, :] = 0.0

        self.history.append((out_state.detach().clone(), out_lidar.detach().clone(), out_dynamic.detach().clone()))
        delayed_state, delayed_lidar, delayed_dynamic = self.history[0]
        return delayed_state, delayed_lidar, delayed_dynamic, {
            "enabled": True,
            "delay_steps": self.delay_steps,
            "effective_delay_steps": max(0, len(self.history) - 1),
            "lidar_noise_std": self.lidar_noise_std,
            "lidar_dropout_ratio": self.lidar_dropout_ratio,
            "obstacle_position_noise_std": self.obstacle_position_noise_std,
            "obstacle_velocity_noise_std": self.obstacle_velocity_noise_std,
            "obstacle_dropout_ratio": self.obstacle_dropout_ratio,
            "state_noise_std": self.state_noise_std,
        }

    def metadata(self):
        return {
            "enabled": self.enabled,
            "delay_steps": self.delay_steps,
            "lidar_noise_std": self.lidar_noise_std,
            "lidar_dropout_ratio": self.lidar_dropout_ratio,
            "obstacle_position_noise_std": self.obstacle_position_noise_std,
            "obstacle_velocity_noise_std": self.obstacle_velocity_noise_std,
            "obstacle_dropout_ratio": self.obstacle_dropout_ratio,
            "state_noise_std": self.state_noise_std,
        }
