import sapien
from furniture_bench.utils.sapien.urdf_loader import URDFLoader
import numpy as np
import os
from pathlib import Path
import scipy.spatial

ASSET_ROOT = str(Path(__file__).parent.parent / "furniture_bench" / "assets_no_tags")

def demo(fix_root_link, disable_gravity, balance_passive_force):
    scene = sapien.Scene()
    scene.add_ground(0, render_material=(0.0, 1.0, 1.0))

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    pose = sapien.Pose(p=[-2, 0, 1])
    vec = np.array([0, 0, 0],dtype=np.float32) - pose.get_p()
    vec = vec / np.linalg.norm(vec)
    print(vec)
    rot,_ = scipy.spatial.transform.Rotation.align_vectors([1, 0, 0], vec)
    pose.set_rpy(rot.as_euler("xyz").astype(np.float32))
    print(pose)
    viewer.set_camera_pose(pose) # only this is valid


    # Load URDF
    loader = URDFLoader()
    loader.set_scene(scene)
    loader.fix_root_link = fix_root_link

    franka_asset_file = (
        "franka_description_ros/franka_description/robots/franka_panda.urdf"
    )

    asset_file = os.path.join(ASSET_ROOT, franka_asset_file)
    franka = loader.load(asset_file, package_dir=ASSET_ROOT)
    for link in franka.links:
        link.set_disable_gravity(disable_gravity) 
    for joint in franka.joints:
        joint.set_armature(np.ones_like(joint.armature) * 0.01)
    # The robot mesh should be flipped

    robot = franka
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    # Set initial joint positions
    # arm_init_qpos = [4.71, 2.84, 0, 0.75, 4.62, 4.48, 4.88]
    # gripper_init_qpos = [0, 0, 0, 0, 0, 0]
    # init_qpos = arm_init_qpos + gripper_init_qpos
    # robot.set_qpos(init_qpos)

    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=not disable_gravity,  # By default, gravity is disabled
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()
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