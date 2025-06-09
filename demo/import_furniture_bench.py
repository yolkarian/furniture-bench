import torch
from furniture_bench.envs.furniture_sim_env import FurnitureSimRLEnv
from furniture_bench.sim_config import (
    sim_config,
)
from furniture_bench.envs.observation import (
    DEFAULT_VISUAL_OBS,FULL_OBS
)
from PIL import Image
import matplotlib.pyplot as plt



if __name__=="__main__":
    sim_config["robot"]["gripper_torque"] = 0.002
    is_reset = True
    sim = FurnitureSimRLEnv(furniture="one_leg", 
                          num_envs=1, 
                          parallel_in_single_scene=False, 
                          headless=False,
                          obs_keys=FULL_OBS, 
                          init_assembled=False,
                          enable_sensor=True,
                          camera_shader="default",
                          viewer_shader="default",
                          action_type="delta",
                          april_tags=True,)
    sim.reset()

    # NOTE: Currently please onlytime use lamp/one_leg for the simulation, since to use other furnitures
    #   file path change in the urdf file should be made.
    action = sim.franka_default_dof_pos[None,:].repeat(sim.num_envs,axis = 0)
    
    action = torch.zeros_like(sim.act_low, device=sim.device)

    while True:
        if sim.env_steps[0] >= 20 and sim.env_steps[0] < 120 :
            action[:, -1] -= 0.001
            action[:, 0] -= 0.05 * sim.dt

        if sim.env_steps[0] >= 120 and is_reset :
            sim.reset()
            action[:, :] = 0
            is_reset = False
        obs,reward,done,info = sim.step(action)
        # plt.imshow(obs["color_image2"][0].cpu().numpy().squeeze())
        # plt.show()
    

