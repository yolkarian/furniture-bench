"""Load the Franka Panda URDF in SAPIEN and render an interactive viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

DEFAULT_ASSET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "furniture_bench"
    / "assets_no_tags"
    / "franka_description_ros"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the SAPIEN Franka viewer CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--fix-root-link", action="store_true")
    parser.add_argument("--balance-passive-force", action="store_true")
    parser.add_argument("--disable-gravity", action="store_true")
    parser.add_argument("--root-height", type=float, default=2.0)
    parser.add_argument(
        "--steps-per-render",
        type=int,
        default=4,
        help="Physics steps to execute before each viewer render.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Open the Franka Panda URDF in a standalone SAPIEN viewer."""
    args = parse_args(argv)

    # Delay simulator imports until after parsing so ``--help`` stays cheap.
    import numpy as np
    import sapien
    import torch
    from scipy.spatial.transform import Rotation

    from furniture_bench.utils.sapien.urdf_loader import URDFLoader

    asset_root = args.asset_root.resolve()
    asset_file = asset_root / "franka_description" / "robots" / "franka_panda.urdf"
    if not asset_file.exists():
        raise FileNotFoundError(f"Cannot find Franka URDF at {asset_file}")

    sapien.physx.enable_gpu()
    sapien.physx.set_scene_config(
        gravity=np.array([0.0, 0.0, -9.8], dtype=np.float32)
    )
    physx_system = sapien.physx.PhysxGpuSystem()
    render_system = sapien.render.RenderSystem(device=sapien.Device("cuda"))
    render_system.get_cubemap()
    scene = sapien.Scene([physx_system, render_system])
    scene.add_ground(0, render_material=(0.0, 1.0, 1.0))
    scene.set_ambient_light([0.1, 0.1, 0.1])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

    viewer = scene.create_viewer()
    camera_pose = sapien.Pose(p=[-2.0, 0.0, 1.0])
    forward = np.array([0.0, 0.0, 0.0], dtype=np.float32) - camera_pose.get_p()
    forward /= np.linalg.norm(forward)
    rotation, _ = Rotation.align_vectors([forward], [[1.0, 0.0, 0.0]])
    camera_pose.set_rpy(rotation.as_euler("xyz").astype(np.float32))
    viewer.set_camera_pose(camera_pose)

    loader = URDFLoader()
    loader.set_scene(scene)
    loader.fix_root_link = args.fix_root_link
    robot = loader.load(str(asset_file), package_dir=str(asset_root))

    for link in robot.links:
        link.set_disable_gravity(args.disable_gravity)
    for joint in robot.joints:
        joint.set_armature(np.ones_like(joint.armature) * 0.01)

    robot.set_root_pose(
        sapien.Pose([0.0, 0.0, args.root_height], [1.0, 0.0, 0.0, 0.0])
    )
    physx_system.gpu_init()

    while not viewer.closed:
        for _ in range(args.steps_per_render):
            if args.balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=not args.disable_gravity,
                    coriolis_and_centrifugal=True,
                )
                physx_system.sync_poses_gpu_to_cpu()
                physx_system.gpu_apply_articulation_qf(
                    torch.from_numpy(qf).to(device="cuda")
                )
            physx_system.step()

        physx_system.sync_poses_gpu_to_cpu()
        viewer.window.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
