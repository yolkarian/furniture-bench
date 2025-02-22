import sapien
import sapien.physx
import sapien.wrapper
import sapien.render
import os

import sapien.wrapper.scene
import sapien.wrapper.urdf_loader
from ..sim_config_sapien import SimParams, AssetOptions


# Temporily Scene loading for the scene
def load_scene_config(scene: sapien.Scene, cfg: SimParams) -> None:
    
    scene_config = sapien.physx.get_scene_config()
    shape_config = sapien.physx.get_shape_config()
    body_config = sapien.physx.get_body_config()

    scene.physx_system.set_timestep(cfg.dt)
    scene_config.gravity = cfg.gravity
    # NOTE(Yuke): sapien doesn't support upaxis setting

    # NOTE(Yuke): sapien doesn't support substep settting
    if cfg.use_gpu_pipeline:
        sapien.physx.enable_gpu()
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


def load_asset_with_options(loader:sapien.wrapper.urdf_loader.URDFLoader,asset_root:str, asset_path:str, options:AssetOptions) -> sapien.physx.PhysxArticulation:
    asset_file = os.path.join(asset_root, asset_path)
    tmp_fix_root_link = loader.fix_root_link
    loader.fix_root_link = options.fix_base_link  # dynamic 
    asset = loader.load(asset_file, package_dir=asset_root)
    loader.fix_root_link = tmp_fix_root_link

    for link in asset.links:
        link.set_angular_damping(options.angular_damping)
        link.set_linear_damping(options.linear_damping)
        link.set_disable_gravity(options.disable_gravity)
        for collision_shape in link.collision_shapes:
            collision_shape.set_density(options.density)  # TODO(Yuke): is this implementation correct or not?
    for joint in asset.joints:
        joint.set_armature(options.armature)
    
    return asset