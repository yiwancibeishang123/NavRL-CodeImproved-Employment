#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        input_vector = torch.nan_to_num(input_vector).clamp(-1e4, 1e4)
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        input_vector = torch.nan_to_num(input_vector).clamp(-1e4, 1e4)
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        input_vector = torch.nan_to_num(input_vector).clamp(-10.0, 10.0)
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        out = out.clamp(-1e4, 1e4)
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive}
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features: torch.Tensor):
        features = torch.nan_to_num(features)
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        alpha = alpha.clamp(1.001, 20.0)
        beta = beta.clamp(1.001, 20.0)
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor

    def forward(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step]
                + self.gamma * next_value[:, step] * not_done[:, step]
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae)
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]


def make_sequence_batch(tensordict: TensorDict, num_minibatches: int):
    num_envs = tensordict.shape[0]
    usable_envs = (num_envs // num_minibatches) * num_minibatches
    perm = torch.randperm(usable_envs, device=tensordict.device).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]


def apply_history_ablation(tensordict: TensorDict, mode: str) -> TensorDict:
    if mode == "none":
        return tensordict

    hist_key = ("agents", "observation", "dynamic_obstacle_hist")
    if hist_key not in tensordict.keys(True, True):
        return tensordict

    obstacle_hist = tensordict.get(hist_key)
    if mode == "zero":
        tensordict.set(hist_key, torch.zeros_like(obstacle_hist))
        return tensordict

    if mode == "shuffle":
        rand = torch.rand(*obstacle_hist.shape[:-1], device=obstacle_hist.device)
        perm = rand.argsort(dim=-1)
        shuffled_hist = obstacle_hist.gather(
            dim=-2,
            index=perm.unsqueeze(-1).expand_as(obstacle_hist),
        )
        tensordict.set(hist_key, shuffled_hist)
        return tensordict

    raise ValueError(f"Unsupported history ablation mode: {mode}")

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0,
    exploration_type: ExplorationType=ExplorationType.MEAN,
    history_ablation_mode: str="none",
):
    env.enable_render(False)
    env.eval()
    env.set_seed(seed)
    if hasattr(policy, "reset_temporal_state"):
        policy.reset_temporal_state()
    tensordict = env.reset()
    done_recorded = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
    traj_stats = None

    with set_exploration_type(exploration_type):
        for _ in range(env.max_episode_length):
            tensordict = apply_history_ablation(tensordict, history_ablation_mode)
            tensordict = policy(tensordict)
            step_tensordict, tensordict = env.step_and_maybe_reset(tensordict)
            step_done = step_tensordict[("next", "done")]
            step_stats = step_tensordict[("next", "stats")]

            if traj_stats is None:
                traj_stats = {
                    k: torch.zeros_like(v, device=v.device)
                    for k, v in step_stats.items()
                }

            first_done = step_done & (~done_recorded)
            if first_done.any():
                mask = first_done.expand_as(step_done)
                for k, v in step_stats.items():
                    traj_stats[k][mask] = v[mask]
                done_recorded |= first_done

            if done_recorded.all():
                break

    if traj_stats is None:
        traj_stats = {
            k: v.clone()
            for k, v in tensordict[("stats")].items()
        }

    metric_prefix = "eval/stats."
    if history_ablation_mode != "none":
        metric_prefix = f"eval_ablation/{history_ablation_mode}/stats."
    info = {
        metric_prefix + k: torch.mean(v.float()).item()
        for k, v in traj_stats.items()
    }
    env.train()
    env.reset()

    return info


def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)

    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)

    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)

    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"
