#!/usr/bin/env python3
import rospy
import hydra
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from ppo import PPO
import numpy as np
import rospkg
from navigation_runner.srv import GetPolicyInference
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict.tensordict import TensorDict

class policy_server:
    def __init__(self, cfg):
        self.cfg = cfg
        requested_device = str(rospy.get_param("~device", cfg.device))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            rospy.logwarn("Requested %s but CUDA is unavailable; using CPU", requested_device)
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.cfg.device = str(self.device)
        self.lidar_hbeams = int(360/self.cfg.sensor.lidar_hres)
        self.raycast_vres = ((self.cfg.sensor.lidar_vfov[1] - self.cfg.sensor.lidar_vfov[0]))/(self.cfg.sensor.lidar_vbeams - 1) * np.pi/180.0
        self.raycast_hres = self.cfg.sensor.lidar_hres * np.pi/180.0
        self.policy = self.init_model()
        self.policy.eval()

    def init_model(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.device), 
            })
        }).expand(1, action_dim).to(self.device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.device)

        source_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "step3-80new.pt")
        try:
            package_checkpoint = os.path.join(rospkg.RosPack().get_path("navigation_runner"), "scripts", "ckpts", "step3-80new.pt")
        except Exception:
            package_checkpoint = source_checkpoint
        default_checkpoint = source_checkpoint if os.path.isfile(source_checkpoint) else package_checkpoint
        checkpoint = rospy.get_param("~model_checkpoint", default_checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("Deployment checkpoint does not exist: {}".format(checkpoint))
        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError("Checkpoint mismatch: missing={}, unexpected={}".format(list(missing)[:10], list(unexpected)[:10]))
        print("[policy server]: model init success: {}".format(checkpoint))
        return policy
    
    def handle_inference(self, req):
        # Convert request data to input format for the model
        # Assume req contains observation data in the necessary format
        drone_state = torch.tensor(req.state, device=self.device).view(req.state_shape)
        lidar_scan = torch.tensor(req.lidar, device=self.device).view(req.lidar_shape)
        target_dir_2d = torch.tensor(req.direction, device=self.device).view(req.direction_shape)
        dyn_obs_states = torch.tensor(req.dynamic_obstacle, device=self.device).view(req.dynamic_obstacle_shape)
        
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state,
                    "lidar": lidar_scan,
                    "direction": target_dir_2d,
                    "dynamic_obstacle": dyn_obs_states
                })
            })
        })
        # Perform inference
        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            output = self.policy.act(obs) if hasattr(self.policy, "act") else self.policy(obs)
        vel_world = output["agents", "action"]
        # Return the result
        # print(vel_world.cpu().detach().squeeze(0).squeeze(0).numpy().tolist())
        return {'action': vel_world.cpu().detach().reshape(-1)[:3].numpy().tolist()}

    def start_server(self):
        rospy.Service('rl_navigation/GetPolicyInference', GetPolicyInference, self.handle_inference)


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts/cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    rospy.init_node('policy_server')
    ps = policy_server(cfg)
    ps.start_server()
    print("[policy server]: ready.")
    rospy.spin()

if __name__ == "__main__":
    main()
