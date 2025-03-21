import torch
import gymnasium as gym
from furniture_bench.sim_config import (
    sim_config,
)
from furniture_bench.envs.observation import (
    DEFAULT_VISUAL_OBS,FULL_OBS
)




if __name__=="__main__":
    sim_config["robot"]["gripper_torque"] = 0.002
    is_reset = True
    sim = gym.make(
                    id = "FurnitureSim-v0",
                        furniture="one_leg", 
                          num_envs=1, 
                          parallel_in_single_scene=False, 
                          headless=False, 
                          obs_keys=FULL_OBS, 
                          init_assembled=True,
                          enable_sensor=True, 
                          camera_shader="default",
                          viewer_shader="rt",
                          action_type="delta"
    )

    # NOTE: Currently please onlytime use lamp/one_leg for the simulation, since to use other furnitures
    #   file path change in the urdf file should be made.
    action = sim.franka_default_dof_pos[None,:].repeat(sim.num_envs,axis = 0)
    sim.reset()
    action = torch.zeros_like(sim.act_low, device=sim.device)

    while not sim.viewer.closed:

        if sim.env_steps[0] >= 200:
            action[:, -1] -= 0.001
            action[:, 0] -= 0.0005 * sim.dt

        if sim.env_steps[0] >= 1000 and is_reset :
            sim.reset()
            is_reset = False
        obs,reward,done,info = sim.step(action)
    

