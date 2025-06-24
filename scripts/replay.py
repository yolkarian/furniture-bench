import os
import torch
from typing import Optional
from furniture_bench.envs.furniture_sim_env import FurnitureSimRLEnv
from furniture_bench.sim_config import (
    sim_config,
)
import safetensors.numpy
import numpy as np
from furniture_bench.envs.observation import (
     FULL_OBS, DEFAULT_REPLAY_KEYS
)
import sapien
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="one_leg")
    parser.add_argument("--record-path", type=Optional[str], default = None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", "-o", type=Optional[str], default = None)
    parser.add_argument("--save-output", action="store_true")

    args = parser.parse_args()

    if args.record_path is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        dir_files = os.listdir(script_path)
        record_paths = [ os.path.join(script_path, dir_file) for dir_file in dir_files if dir_file.endswith(".safetensors")]
        if len(record_paths) == 0:
            raise ValueError("Cannot find any path to replay!")
        record_path = record_paths[0]
    else:
        record_path = args.record_path
    if args.output is None:
        output = record_path.rsplit(".", maxsplit=1)[0] + "_color_images.safetensors"
    else:
        output = args.output


    tensor_dict = safetensors.numpy.load_file(record_path)

    num_epoch = tensor_dict["nobs"].shape[0]
    max_steps = tensor_dict["nobs"].shape[1]
    num_envs = 1 # tensor_dict["nobs"].shape[2] # Currently only 1

    data_keys = list(tensor_dict.keys())

    if not all([(k in data_keys) for k in DEFAULT_REPLAY_KEYS]):
        raise ValueError("Recorded data does not contain joint info.")

    if "parts_poses" not in data_keys:
        raise ValueError("Recorded data does not contain parts info.")

    qpos_list = [tensor_dict[k] for k in DEFAULT_REPLAY_KEYS ]

    qpos = np.concatenate(qpos_list, axis=-1).astype(np.float32)
    parts_poses = tensor_dict["parts_poses"].astype(np.float32)

    env = FurnitureSimRLEnv(furniture=args.task,
                            num_envs=num_envs,
                            parallel_in_single_scene=False,
                            headless=args.headless,
                            obs_keys=FULL_OBS,
                            init_assembled=False,
                            enable_sensor=True,
                            camera_shader="rt",
                            viewer_shader="rt",
                            action_type="delta",
                            april_tags=True,
                            record=True)
    obs = env.reset()

    # NOTE: Currently please onlytime use lamp/one_leg for the simulation, since to use other furnitures
    #   file path change in the urdf file should be made.

    epoch_idx = 0
    if args.save_output:
        color_image1 = np.zeros((num_epoch, max_steps, num_envs, *obs["color_image1"].shape[-3:]), dtype=np.uint8)
        color_image2 =np.zeros((num_epoch, max_steps, num_envs, *obs["color_image2"].shape[-3:]), dtype=np.uint8)

    for i in range(max_steps):
        if i >= 0:
            env.rand_parts_rendering(0.6)
            env.rand_light(0.6)
            env.rand_franka_rendering(0.2)
            env.rand_obstacle_rendering(0.3)
        obs = env.render_only_step(torch.from_numpy(qpos[epoch_idx,i]).to("cuda"), parts_poses[epoch_idx , i])
        if args.save_output:
            color_image1[epoch_idx, i] = obs["color_image1"].cpu()
            color_image2[epoch_idx, i] = obs["color_image2"].cpu()

    if args.save_output:
        safetensors.numpy.save_file({
            "color_image1":color_image1,
            "color_image2":color_image2,
        }, output)


    del env

