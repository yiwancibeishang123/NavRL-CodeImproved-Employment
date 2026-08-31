#!/usr/bin/env python3
"""Deployment-side observation construction shared by ROS and replay tests."""

import torch
from typing import Tuple

from utils import vec_to_new_frame


def safe_direction(direction: torch.Tensor) -> torch.Tensor:
    direction = torch.as_tensor(direction, dtype=torch.float32)
    direction = torch.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
    direction = direction.reshape(-1)[:3]
    if direction.numel() != 3 or direction.norm() < 1e-6:
        direction = torch.tensor([1.0, 0.0, 0.0], device=direction.device)
    direction = direction.clone()
    direction[2] = 0.0
    if direction[:2].norm() < 1e-6:
        direction[:2] = torch.tensor([1.0, 0.0], device=direction.device)
    return direction / direction[:2].norm().clamp_min(1e-6)


def build_lidar_observation(
    raypoints,
    position: torch.Tensor,
    lidar_range: float,
    horizontal_beams: int,
    vertical_beams: int,
) -> torch.Tensor:
    """Convert map raycast points to NavRL's fixed (1,1,H,V) tensor."""
    expected = int(horizontal_beams) * int(vertical_beams)
    if raypoints is None or len(raypoints) != expected:
        raise ValueError("raycast returned {} points, expected {}".format(len(raypoints or []), expected))
    points = torch.as_tensor(raypoints, device=position.device, dtype=torch.float32)
    points = torch.nan_to_num(points, nan=lidar_range, posinf=lidar_range, neginf=0.0)
    distances = (points - position.reshape(1, 3)).norm(dim=-1)
    distances = distances.clamp(0.0, float(lidar_range))
    return (float(lidar_range) - distances).reshape(1, 1, horizontal_beams, vertical_beams)


def build_dynamic_obstacle_observation(
    position: torch.Tensor,
    target_direction: torch.Tensor,
    obstacle_position: torch.Tensor,
    obstacle_velocity: torch.Tensor,
    obstacle_size: torch.Tensor,
    robot_size: float,
) -> torch.Tensor:
    """Build the 10-D current obstacle state expected by current_geometry."""
    pos = torch.nan_to_num(obstacle_position.to(position.device), nan=0.0, posinf=0.0, neginf=0.0)
    vel = torch.nan_to_num(obstacle_velocity.to(position.device), nan=0.0, posinf=0.0, neginf=0.0)
    size = torch.nan_to_num(obstacle_size.to(position.device), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    direction = safe_direction(target_direction).to(position.device)

    relative = pos - position.reshape(1, 3)
    valid = size[:, 2] > 0.0
    relative = torch.where(valid.unsqueeze(-1), relative, torch.zeros_like(relative))
    relative[:, 2] = torch.where(size[:, 2] > 1.0, torch.zeros_like(relative[:, 2]), relative[:, 2])
    relative_goal = vec_to_new_frame(relative.unsqueeze(0), direction.reshape(1, 3)).squeeze(0)
    distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    distance_2d = relative_goal[..., :2].norm(dim=-1, keepdim=True)
    distance_z = relative_goal[..., 2:3]
    velocity_goal = vec_to_new_frame(vel.unsqueeze(0), direction.reshape(1, 3)).squeeze(0)
    normalized_relative = relative_goal / distance

    width = torch.maximum(size[:, 0], size[:, 1]) + 2.0 * float(robot_size)
    width = torch.clamp(torch.ceil(width / 0.25) - 1.0, min=0.0, max=3.0)
    width = torch.where(valid, width, torch.zeros_like(width))
    height = size[:, 2]
    height = torch.where((height > 0.0) & (height <= 1.0), torch.ones_like(height), height)
    height = torch.where(height > 1.0, torch.zeros_like(height), height)

    state = torch.cat(
        [normalized_relative, distance_2d, distance_z, velocity_goal,
         width.unsqueeze(-1), height.unsqueeze(-1)], dim=-1
    )
    return torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0).unsqueeze(0)


def build_state_observation(
    position: torch.Tensor,
    velocity: torch.Tensor,
    goal: torch.Tensor,
    target_direction: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    direction = safe_direction(target_direction).to(position.device)
    relative = goal - position
    distance_2d = relative[:2].norm().reshape(1)
    distance_z = relative[2].reshape(1)
    relative_unit = relative / relative.norm().clamp_min(1e-6)
    relative_goal = vec_to_new_frame(relative_unit, direction).reshape(-1)
    velocity_goal = vec_to_new_frame(velocity, direction).reshape(-1)
    state = torch.cat([relative_goal, distance_2d, distance_z, velocity_goal]).reshape(1, 8)
    return state, direction.reshape(1, 3)
