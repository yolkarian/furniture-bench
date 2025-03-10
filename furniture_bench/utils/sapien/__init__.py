import sapien
import sapien.physx
import sapien.wrapper
import sapien.render
import os
import numpy as np
import scipy.spatial

import sapien.wrapper.scene
import sapien.wrapper.urdf_loader
from sapien.wrapper.articulation_builder import ActorBuilder, ArticulationBuilder
from sapien.render import RenderCameraComponent 
from furniture_bench.sim_config_sapien import SimParams, AssetOptions
from .urdf_loader import URDFLoader
from typing import List, Tuple, Union, Optional, Dict, Callable
import torch

# Valid for minimal shader
OBS_KEY_2_PICTURE_NAME:Dict[str, str] = {
    "color":"Color",
    "depth":"PositionSegmentation",
    "position":"PositionSegmentation"
    }

OBS_KEY_2_TRANSFORM:Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "color": lambda data: data[..., :3],
    "depth": lambda data: -data[..., [2]],
}

# Temporily Scene loading for the scene
def load_scene_config(scene: sapien.Scene, cfg: SimParams) -> None:
    
    scene_config = sapien.physx.get_scene_config()
    shape_config = sapien.physx.get_shape_config()
    body_config = sapien.physx.get_body_config()
    scene.physx_system.set_timestep(cfg.dt)
    scene_config.gravity = cfg.gravity
    # NOTE(Yuke): sapien doesn't support upaxis setting

    # NOTE(Yuke): sapien doesn't support substep settting
    # This should be called before
    # if cfg.use_gpu_pipeline:
    #     sapien.physx.enable_gpu()
    scene_config.bounce_threshold = cfg.physx.bounce_threshold_velocity

    if cfg.physx.solver_type == 0:
        scene_config.enable_tgs = False
    else:
        scene_config.enable_tgs = True

    # In sapien, the following parameters cannot be set in simulation level
    shape_config.contact_offset = cfg.physx.contact_offset
    shape_config.rest_offset = cfg.physx.rest_offset
    body_config.solver_position_iterations = cfg.physx.num_position_iterations
    body_config.solver_velocity_iterations = cfg.physx.num_velocity_iterations
    # NOTE(Yuke): sapien currently no support for firction_offset_threshold, friction_correlation_distance
    # NOTE(Yuke): max_depenetration_velocity must be set to each robot individually.

    cpu_workers = (
        cfg.physx.num_threads
    )  # TODO(Yuke): is they are exactly the same thing?


     # NOTE: When directly pass scene_config to the API, cpu_workers cannot be set.
    sapien.physx.set_scene_config(
        gravity=scene_config.gravity, 
        bounce_threshold=scene_config.bounce_threshold,
        enable_pcm=scene_config.enable_pcm,
        enable_tgs=scene_config.enable_tgs,
        enable_ccd=scene_config.enable_ccd,
        enable_enhanced_determinism=scene_config.enable_enhanced_determinism,
        enable_friction_every_iteration=scene_config.enable_friction_every_iteration,
        cpu_workers=cpu_workers
    )
    sapien.physx.set_shape_config(shape_config)
    sapien.physx.set_body_config(body_config)


def generate_builder_with_options(loader:URDFLoader,asset_root:str, asset_path:str, options:AssetOptions) -> Union[ArticulationBuilder, ActorBuilder]:
    loader.fix_root_link = options.fix_base_link
    full_path = os.path.join(asset_root, asset_path)
    articulator_builders, actor_builders,_ = loader.parse(full_path)
    if len(articulator_builders)!=0:
        builder = articulator_builders[0]
        for link_builder in builder.link_builders:
            link_builder.disable_gravity = options.disable_gravity  
            link_builder.linear_damping = options.linear_damping 
            link_builder.angular_damping = options.angular_damping
            for collision_record in link_builder.collision_records:
                collision_record.density = options.density
            link_builder.joint_record.armature = options.armature
        return builder 
        
    builder = actor_builders[0]
    builder.disable_gravity = options.disable_gravity  
    builder.linear_damping = options.linear_damping 
    builder.angular_damping = options.angular_damping
    for collision_record in builder.collision_records:
        collision_record.density = options.density
    # there is no joint for actor
    return builder

def generate_builder_with_options_(loader:URDFLoader,urdf_path:str,options:AssetOptions, package_dir:Optional[str]=None) -> Union[ArticulationBuilder, ActorBuilder]:
    loader.fix_root_link = options.fix_base_link
    articulator_builders, actor_builders,_ = loader.parse(urdf_path, package_dir=package_dir)
    if len(articulator_builders)!=0:
        builder = articulator_builders[0]
        for link_builder in builder.link_builders:
            link_builder.disable_gravity = options.disable_gravity  
            link_builder.linear_damping = options.linear_damping 
            link_builder.angular_damping = options.angular_damping
            for collision_record in link_builder.collision_records:
                collision_record.density = options.density
            link_builder.joint_record.armature = options.armature
        return builder 
        
    builder = actor_builders[0]
    builder.disable_gravity = options.disable_gravity  
    builder.linear_damping = options.linear_damping 
    builder.angular_damping = options.angular_damping
    for collision_record in builder.collision_records:
        collision_record.density = options.density
    # there is no joint for actor
    return builder


def camera_pose_from_look_at(pos:np.ndarray, target:np.ndarray)->sapien.Pose:
    vec = target.astype(np.float32) - pos.astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    pose = sapien.Pose(p = pos.astype(np.float32))
    rot,_ = scipy.spatial.transform.Rotation.align_vectors(vec, [1, 0, 0])
    if vec[0] < 0:
        quat = np.zeros(4, dtype = np.float32)
        quat[:3] = vec
        rot =  rot * scipy.spatial.transform.Rotation.from_quat(quat) 
    pose.set_rpy(rot.as_euler("xyz").astype(np.float32))
    return pose