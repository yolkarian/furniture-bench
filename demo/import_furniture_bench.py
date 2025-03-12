import torch
from furniture_bench.envs.sapien_envs import FurnitureSimEnv
from furniture_bench.sim_config_sapien import (
    sim_config,
)
from furniture_bench.envs.observation import (
    DEFAULT_VISUAL_OBS,FULL_OBS
)



if __name__=="__main__":
    sim_config["robot"]["gripper_torque"] = 0.002
    is_reset = True
    sim = FurnitureSimEnv(furniture="one_leg", num_envs=4, parallel_in_single_scene=False, headless=False, obs_keys=FULL_OBS, enable_sensor=False)

    # NOTE: Currently please onlytime use lamp/one_leg for the simulation, since to use other furnitures
    #   file path change in the urdf file should be made.
    action = sim.franka_default_dof_pos[None,:].repeat(sim.num_envs,axis = 0)
    sim.action_type = 1
    action = torch.zeros_like(sim.act_low, device=sim.device)

    while not sim.viewer.closed:

        # sim._fetch_all()
        # sim.get_jacobian_ee(sim.get_qpos())

        # # Random Policy for gripper
        # # Gripper is directly controlled with force
        # noise = np.random.r        # print(obs["parts_poses"].cpu().numpy()[:, -7:-4])and(*action.shape)  - 0.5
        # noise[:, :-2] = 0
        # action_np = action

        # action_torch = torch.from_numpy(action_np).to(sim.device, dtype=torch.float32)
        # noise_torch = torch.from_numpy(noise).to(sim.device, dtype = torch.float32) 
        # sim.physx_system.cuda_articulation_target_qpos.torch()[:,:7] = action_torch[:,:7]
        # # sim.physx_system.cuda_articulation_target_qvel.torch()[:, :] = torch.zeros_like(sim.physx_system.cuda_articulation_target_qvel.torch()).to(sim.device)
        
        # # Force Control of gripper
        # sim.physx_system.cuda_articulation_qf.torch()[:,-2:] = 10 * noise_torch[:,-2:]
        # # print(sim.physx_system.cuda_rigid_body_data.torch().shape)

        # # It seems the order of applying values does not matter
        # sim.physx_system.gpu_apply_articulation_qf()
        # sim.physx_system.gpu_apply_articulation_target_position()
        # # sim.physx_system.gpu_apply_articulation_target_velocity()
        if sim.env_steps[0] >= 200:
            action[:, -1] -= 0.001
            action[:, 0] -= 0.0005 * sim.dt

        if sim.env_steps[0] >= 1000 and is_reset :
            sim.reset()
            is_reset = False
        obs,reward,done,info = sim.step(action)
    
        # This should be put after the step in which we first set target
        # sim.physx_system.step()
        # sim.step_viewer()

