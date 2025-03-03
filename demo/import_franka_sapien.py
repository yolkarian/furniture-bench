import sapien
from furniture_bench.utils.sapien.urdf_loader import URDFLoader
import numpy as np
import os
from pathlib import Path
import scipy.spatial
import torch

ASSET_ROOT = str(Path(__file__).parent.parent / "furniture_bench" / "assets_no_tags"/ "franka_description_ros")

def demo(fix_root_link, disable_gravity, balance_passive_force):
    sapien.physx.enable_gpu()
    sapien.physx.set_scene_config(gravity=np.array([0.0, 0.0, -9.8],dtype=np.float32))
    px = sapien.physx.PhysxGpuSystem()
    rs = sapien.render.RenderSystem(device=sapien.Device("cuda"))
    rs.get_cubemap()
    scene = sapien.Scene([px, rs])
    scene.add_ground(0, render_material=(0.0, 1.0, 1.0))

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    pose = sapien.Pose(p=[-2, 0, 1])
    vec = np.array([0, 0, 0],dtype=np.float32) - pose.get_p()
    vec = vec / np.linalg.norm(vec)
    print(vec)
    rot,_ = scipy.spatial.transform.Rotation.align_vectors(vec, [1, 0, 0])
    pose.set_rpy(rot.as_euler("xyz").astype(np.float32))
    # pose.set_q(np.roll(rot.as_quat(), 1).astype(np.float32))
    viewer.set_camera_pose(pose) # only this is valid


    # Load URDF
    loader = URDFLoader()
    loader.set_scene(scene)
    loader.fix_root_link = fix_root_link

    franka_asset_file = (
        "franka_description/robots/franka_panda.urdf"
    )

    asset_file = os.path.join(ASSET_ROOT, franka_asset_file)
    franka = loader.load(asset_file, package_dir=ASSET_ROOT)
    for link in franka.links:
        link.set_disable_gravity(disable_gravity) 
    for joint in franka.joints:
        joint.set_armature(np.ones_like(joint.armature) * 0.01)
    # The robot mesh should be flipped

    robot = franka
    robot.set_root_pose(sapien.Pose([0, 0, 2], [1, 0, 0, 0]))

    px.gpu_init()

    # px.gpu_fetch_articulation_link_pose()

    # px.gpu_apply_rigid_dynamic_data()
    # px.gpu_apply_articulation_qpos()
    # px.gpu_apply_articulation_qvel()
    # px.gpu_apply_articulation_qf()
    # px.gpu_apply_articulation_root_pose()
    # px.gpu_apply_articulation_root_velocity()
    # px.gpu_apply_articulation_target_position()
    # px.gpu_apply_articulation_target_velocity()

    # px.gpu_update_articulation_kinematics()
    # px.gpu_fetch_articulation_link_pose()

    # px.gpu_fetch_articulation_link_pose()
    # px.gpu_fetch_articulation_link_velocity()
    # px.gpu_fetch_articulation_qpos()
    # px.gpu_fetch_articulation_qvel()
    # px.gpu_fetch_articulation_qacc()
    # px.gpu_fetch_articulation_target_qpos()
    # px.gpu_fetch_articulation_target_qvel()

    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=not disable_gravity,  # By default, gravity is disabled
                    coriolis_and_centrifugal=True,
                )
                
                qf_torch = torch.from_numpy(qf).to(device="cuda")
                px.sync_poses_gpu_to_cpu()
                px.gpu_apply_articulation_qf(qf_torch)
        px.step()
        px.sync_poses_gpu_to_cpu()
        viewer.window.update_render()
        viewer.render()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-root-link", action="store_true")
    parser.add_argument("--balance-passive-force", action="store_true")
    parser.add_argument("--disable-gravity", action="store_true")
    args = parser.parse_args()

    demo(
        fix_root_link=args.fix_root_link,
        disable_gravity=args.disable_gravity,
        balance_passive_force=args.balance_passive_force,
    )


if __name__ == "__main__":
    main()