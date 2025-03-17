import sapien.core
import sapien
import sapien.physx
import sapien.render
import pytorch_kinematics as pk
from furniture_bench.utils.sapien.urdf_loader import URDFLoader
from sapien.wrapper.urdf_exporter import export_kinematic_chain_urdf
import sapien.utils.viewer.control_window
from furniture_bench.utils.sapien import (
    generate_builder_with_options_,
    camera_pose_from_look_at,
)
from furniture_bench.utils.sapien.camera import (
    SHADER_DICT,
    set_shader,
    SAPIEN_SHADERS_DIR,
)
from sapien.utils.viewer import Viewer
import os
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union
from pathlib import Path

from furniture_bench.furniture.furniture import Furniture
from furniture_bench.utils.recorder import VideoRecorder
import torch
import gymnasium as gym

import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.envs.initialization_mode import Randomness, str_to_enum
from furniture_bench.controllers.diffik import diffik_factory

from furniture_bench.furniture import furniture_factory

# from furniture_bench.sim_config import sim_config
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.envs.observation import (
    DEFAULT_VISUAL_OBS,
    FULL_OBS,
    DEFAULT_STATE_OBS,
)
from furniture_bench.robot.robot_state import ROBOT_STATE_DIMS, ROBOT_STATES
from furniture_bench.furniture.parts.part import Part

from ipdb import set_trace as bp
from typing import Union, Tuple, List, Optional, Literal, Set

from furniture_bench.sim_config import (
    sim_config,
    AssetOptions,
    SimParams,
    PhysxParams,
    CameraCfg,
)
from furniture_bench.utils.sapien.actor_builder import ActorBuilder
from furniture_bench.utils.sapien.articulation_builder import ArticulationBuilder


ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets_no_tags")

# TODO:
#       Add randomness of obstacle, given that the full observations include the pose of obstacles
#       Investigation the difference between Isaac Gym and Sapien in Operation
#       Implement Sensor with different Shader (For different Shader, the operation to read data is different)
#       Turn off RenderGroup when num_env == 1

# NOTE(Yuke): Regarding the unit of depth image, please check the comment in `furniture_bench.utils.sapien.camera`


class FurnitureSimEnv(gym.Env):
    def __init__(
        self,
        furniture: str,
        num_envs: int = 1,
        obs_keys: List[str] = None,
        act_rot_repr: Literal["quat", "rot_6d", "axis"] = "quat",
        parts_poses_in_robot_frame: bool = False,  # Or Apriltag Frame
        randomness: Union[str, Randomness] = "low",
        headless: bool = False,
        init_assembled: bool = False,
        enable_sensor: bool = True,
        action_type: Literal["delta", "pos"] = "delta",
        ctrl_mode: Literal["diffik"] = "diffik",
        parallel_in_single_scene: bool = False,
        manual_done: bool = False,
        camera_shader: Optional[Literal["default", "minimal", "rt"]] = None,
        viewer_shader: Optional[Literal["default", "minimal", "rt"]] = None,
        record: bool = False,  #  TODO: Video Recording for this repository
        save_camera_input: bool = False,
        enable_reward: bool = False,
        april_tags: bool = False,
        perturbation_prob: float = 0.01,
        **kwargs: dict,  # dict which is used to catch extra params
    ):
        self.furniture_name = furniture
        self.task_name = furniture
        self.num_envs = num_envs
        if act_rot_repr == "axis":
            self.__act_rot_repr: Literal[0, 1, 2] = 2
        elif act_rot_repr == "rot_6d":
            self.__act_rot_repr: Literal[0, 1, 2] = 1
        else:
            self.__act_rot_repr: Literal[0, 1, 2] = 0

        self.obs_keys = obs_keys or DEFAULT_STATE_OBS
        self.parts_poses_in_robot_frame = parts_poses_in_robot_frame
        self.randomness = str_to_enum(randomness)
        self.headless = headless
        self.enable_sensor = enable_sensor
        self.parallel_in_single_scene = parallel_in_single_scene
        self.manual_done = manual_done
        self.init_assembled: bool = init_assembled  # If true, the environment is initialized and reset without randomness
        self.camera_shader = (
            SHADER_DICT[camera_shader]
            if camera_shader is not None
            else SHADER_DICT["minimal"]
        )  #TODO: Only allow camera rendering with ray-tracing in 1 env mode
        self.viewer_shader = (
            SHADER_DICT[viewer_shader]
            if viewer_shader is not None
            else SHADER_DICT["minimal"]
        )
        self.ctrl_mode = ctrl_mode
        self.enable_reward = enable_reward
        if april_tags:
            global ASSET_ROOT
            ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")
        self.perturbation_prob = perturbation_prob
        self.device = torch.device("cuda")
        self.sapien_device = sapien.Device(self.device.type)
        self.assemble_idx = 0
        self.move_neutral = False
        # osc is not supported
        assert self.ctrl_mode == "diffik"

        if self.randomness == Randomness.LOW:
            self.max_force_magnitude = 0.2
            self.max_torque_magnitude = 0.007
            self.max_obstacle_offset = 0.02
            self.franka_joint_rand_lim_deg = np.radians(5)
        elif self.randomness == Randomness.MEDIUM:
            self.max_force_magnitude = 0.5
            self.max_torque_magnitude = 0.01
            self.max_obstacle_offset = 0.04
            self.franka_joint_rand_lim_deg = np.radians(10)
        elif self.randomness == Randomness.HIGH:
            self.max_force_magnitude = 0.75
            self.max_torque_magnitude = 0.015
            self.max_obstacle_offset = 0.06
            self.franka_joint_rand_lim_deg = np.radians(13)
        else:
            raise ValueError("Invalid randomness level")

        print(
            f"Max force magnitude: {self.max_force_magnitude} "
            f"Max torque magnitude: {self.max_torque_magnitude} "
            f"Obstacle range: {self.max_obstacle_offset} "
            f"Franka joint randomization limit: {self.franka_joint_rand_lim_deg}"
        )
        super(FurnitureSimEnv, self).__init__()
        if self.furniture_name == "one_leg":
            force_mul = [25, 1, 1, 1, 1]
            torque_mul = [70, 1, 1, 1, 1]
        elif self.furniture_name == "lamp":
            force_mul = [8, 15, 30]
            torque_mul = [16, 20, 60]
        elif self.furniture_name == "round_table":
            force_mul = [30, 4, 20]
            torque_mul = [60, 4, 10]
        elif self.furniture_name == "square_table":
            force_mul = [25, 1, 1, 1, 1]
            torque_mul = [70, 1, 1, 1, 1]
        elif self.furniture_name == "mug_rack":
            force_mul = [50, 20]
            torque_mul = [150, 30]
        elif self.furniture_name == "factory_peg_hole":
            force_mul = [0.001, 0.001]
            torque_mul = [0.001, 0.001]
        elif self.furniture_name == "factory_nut_bolt":
            force_mul = [0.001, 0.001]
            torque_mul = [0.001, 0.001]
        else:
            raise ValueError(
                f"Have not set up the random force/torque multipliers for furniture {self.furniture_name}"
            )

        print(f"Force multiplier: {force_mul}")
        print(f"Torque multiplier: {torque_mul}")

        self.force_multiplier = torch.tensor(force_mul, device=self.device).unsqueeze(
            -1
        )
        self.torque_multiplier = torch.tensor(torque_mul, device=self.device).unsqueeze(
            -1
        )

        # Pair for the reward computation
        if self.furniture_name == "one_leg":
            self.pairs_to_assemble = [(0, 4)]
        elif self.furniture_name == "lamp":
            self.pairs_to_assemble = [(0, 1), (0, 2)]
        elif self.furniture_name == "round_table":
            self.pairs_to_assemble = [(0, 1), (1, 2)]
        elif self.furniture_name == "square_table":
            self.pairs_to_assemble = [(0, 1), (0, 2), (0, 3), (0, 4)]
        elif self.furniture_name == "mug_rack":
            self.pairs_to_assemble = [(0, 1)]
        elif self.furniture_name == "factory_peg_hole":
            self.pairs_to_assemble = [(0, 1)]
        elif self.furniture_name == "factory_nut_bolt":
            self.pairs_to_assemble = [(0, 1)]
        else:
            raise ValueError(
                f"Have not set up the pairs to assemble for furniture {self.furniture_name}"
            )

        # Predefined parameters
        self.pose_dim = 7
        self.stiffness = 1000.00
        self.damping = 200.0
        self.restitution = 0.000
        self.pos_scalar: float = 1.0
        self.rot_scalar: float = 1.0
        self.__action_type: Literal[0, 1] = (
            0 if action_type == "pos" else 1
        )  # 0: pos mode, 1: delta mode
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)
        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open and close actions.
        self.max_gripper_width = config["robot"]["max_gripper_width"][
            self.furniture_name
        ]

        self.include_parts_poses: bool = "parts_poses" in self.obs_keys

        self.robot_state_keys = [
            k.split("/")[1] for k in self.obs_keys if k.startswith("robot_state")
        ]

        self.furnitures: List[Furniture] = [
            furniture_factory(self.furniture_name) for _ in range(num_envs)
        ]
        if self.num_envs == 1:
            self.furniture = self.furnitures[0]
        else:
            self.furniture = furniture_factory(self.furniture_name)

        self.from_skill = 0  # NOTE(Yuke): Unknow parameters
        rel_poses_arr = np.asarray(
            [
                self.furniture.assembled_rel_poses[pair_key]
                for pair_key in self.pairs_to_assemble
            ],
        )

        # Size (num_pairs) x (num_poses) x 4 x 4
        self.assembled_rel_poses = (
            torch.from_numpy(rel_poses_arr).float().to(self.device)
        )
        self.already_assembled = torch.zeros(
            (self.num_envs, len(self.pairs_to_assemble)),
            dtype=torch.bool,
            device=self.device,
        )

        if "factory" in self.furniture_name:
            # Adjust simulation parameters
            sim_config["sim_params"].dt = 1.0 / 120.0
            sim_config["sim_params"].substeps = 4
            sim_config["sim_params"].physx.max_gpu_contact_pairs = 6553600
            sim_config["sim_params"].physx.default_buffer_size_multiplier = 8.0

            # Adjust part friction
            sim_config["parts"]["friction"] = 0.25

        # Predefined params to load objs, actors and articulations
        self.static_obj_dict: Dict[str, str] = {
            "base_tag": "furniture/urdf/base_tag.urdf",
            "obstacle_front": "furniture/urdf/obstacle_front.urdf",
            "obstacle_right": "furniture/urdf/obstacle_side.urdf",
            "obstacle_left": "furniture/urdf/obstacle_side.urdf",
            "background": "furniture/urdf/background.urdf",
            "table": "furniture/urdf/table.urdf",
        }
        # Pose setting of the scene in the simulation
        self.scene_spacing = 3.0
        self.table_pos = np.array([0.8, 0.8, 0.4], dtype=np.float32)

        table_half_width = 0.015
        table_surface_z: float = self.table_pos[2] + table_half_width
        self.franka_pose = sapien.Pose(
            p=[0.5 * -self.table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT],
        )

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p[0], self.franka_pose.p[1], self.franka_pose.p[2]],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat: NDArray[np.float32] = config["robot"][
            "tag_base_from_robot_base"
        ]

        self.base_tag_pose = sapien.Pose()
        base_tag_pos = T.pos_from_mat(config["robot"]["tag_base_from_robot_base"])
        self.base_tag_pose.p = self.franka_pose.p + np.array(
            [base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT], dtype=np.float32
        )
        self.base_tag_pose.p[2] = table_surface_z

        self.obstacle_front_pose = sapien.Pose()  # in Sim Coordinate
        self.obstacle_front_pose.p = np.array(
            [self.base_tag_pose.p[0] + 0.37 + 0.01, 0.0, table_surface_z + 0.015],
            dtype=np.float32,
        )
        self.obstacle_front_pose.set_rpy([0, 0, 0.5 * np.pi])
        self.obstacle_right_pose = sapien.Pose()
        self.obstacle_right_pose.p = np.array(
            [
                self.obstacle_front_pose.p[0] - 0.075,
                self.obstacle_front_pose.p[1] - 0.175,
                self.obstacle_front_pose.p[2],
            ],
            dtype=np.float32,
        )
        self.obstacle_right_pose.q = self.obstacle_front_pose.q

        self.obstacle_left_pose = sapien.Pose()
        self.obstacle_left_pose.p = np.array(
            [
                self.obstacle_front_pose.p[0] - 0.075,
                self.obstacle_front_pose.p[1] + 0.175,
                self.obstacle_front_pose.p[2],
            ],
            dtype=np.float32,
        )
        self.obstacle_left_pose.q = self.obstacle_front_pose.q

        # Define parameters of the camera
        self.resize_img = True
        self.img_size = sim_config["camera"][
            "resized_img_size" if self.resize_img else "color_img_size"
        ]

        # %% General Setup of Simulator
        sim_params: SimParams = sim_config["sim_params"]
        sapien.physx.enable_gpu()
        sapien.physx.set_scene_config(
            gravity=sim_params.gravity,
            bounce_threshold=sim_params.physx.bounce_threshold_velocity,
            enable_tgs=True if sim_params.physx.solver_type == 1 else False,
            cpu_workers=sim_params.physx.num_threads,
        )
        sapien.physx.set_shape_config(
            contact_offset=sim_params.physx.contact_offset,
            rest_offset=sim_params.physx.rest_offset,
        )
        sapien.physx.set_body_config(
            solver_position_iterations=sim_params.physx.num_position_iterations,
            solver_velocity_iterations=sim_params.physx.num_velocity_iterations,
        )

        self.physx_system = sapien.physx.PhysxGpuSystem(self.sapien_device)
        self.urdf_loader = URDFLoader()  # just used to generate builder
        self.render_system_group: Optional[Union[sapien.render.RenderSystemGroup, sapien.render.RenderSystem]] = None
        # Simulation Step
        self.env_steps = np.zeros(self.num_envs, dtype=np.int32)

        # %% Create builder

        self._create_static_obj_builders()
        self._create_ground_builder()
        self._create_franka_builder()
        self._create_part_builders()

        # load Scene configs
        self.physx_system.set_timestep(sim_params.dt)
        self.dt = sim_params.dt

        self._create_scenes()
        self._add_light()

        # initializing Physx simulation scene on GPU (Load/Reset all qpos to the default qpos)
        self._init_sim()

        if not self.headless:
            # viewer uses different pipeline
            self._init_viewer()
        else:
            self.viewer = None
        if not self.parallel_in_single_scene and self.enable_sensor:
            self._load_sensors()
            self._init_render()

        # TODO: Set up Random Number Generator here to ensure the determinism
        self._init_ctrl()

        try:
            gym.logger.set_level(gym.logger.INFO)
        except Exception:
            pass

        self.act_low = torch.from_numpy(self.action_space.low).to(device=self.device)
        self.act_high = torch.from_numpy(self.action_space.high).to(device=self.device)
        self.sim_steps = int(
            1.0 / config["robot"]["hz"] / sim_config["sim_params"].dt + 0.1
        )

    def _create_static_obj_builders(self):
        # Create builders for all static objs (Actorbuilder/ArticulationBuilder)
        self.static_obj_builders: Dict[str, ActorBuilder] = {}
        for key, value in self.static_obj_dict.items():
            opts = AssetOptions()
            opts.fix_base_link = True
            self.static_obj_builders[key] = generate_builder_with_options_(
                self.urdf_loader, os.path.join(ASSET_ROOT, value), opts
            )
            self.static_obj_builders[key].set_physx_body_type("kinematic")
            # NOTE(Yuke): Use kinematic instead of static here, because it may reload objs at different position
            self.static_obj_builders[key].set_name(name=key)

        # Setup Config for objs
        self.static_obj_builders["table"].set_initial_pose(
            sapien.Pose(p=[0, 0, self.table_pos[2]])
        )
        for collision_record in self.static_obj_builders["table"].collision_records:
            mt = sapien.physx.PhysxMaterial(
                1.05 * sim_config["table"]["friction"],
                sim_config["table"]["friction"],
                self.restitution,
            )
            collision_record.material = mt
        self.static_obj_builders["base_tag"].set_initial_pose(self.base_tag_pose)
        self.static_obj_builders["background"].set_initial_pose(
            sapien.Pose(p=[-0.8, 0, 0.75])
        )
        self.static_obj_builders["obstacle_front"].set_initial_pose(
            self.obstacle_front_pose
        )
        self.static_obj_builders["obstacle_left"].set_initial_pose(
            self.obstacle_left_pose
        )
        self.static_obj_builders["obstacle_right"].set_initial_pose(
            self.obstacle_right_pose
        )

    def _create_ground_builder(self):
        # Do we need to create a ground for each scene ? ()
        self.ground_builder = sapien.ActorBuilder()
        self.ground_builder.set_name("ground")
        self.ground_builder.set_physx_body_type("static")
        self.ground_builder.add_plane_collision(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0])
        )
        self.ground_builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        self.ground_builder.add_plane_visual(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]),
            [10, 10, 10],
            [0.0, 1.0, 1.0],
            "ground_visual",
        )

    def _create_franka_builder(self):
        urdf_file = "franka_description_ros/franka_description/robots/franka_panda.urdf"
        opts = AssetOptions()
        opts.armature = 0.01
        opts.thickness = 0.001
        opts.density = 5 * opts.density
        opts.fix_base_link = True
        opts.disable_gravity = True
        opts.flip_visual_attachments = True
        self.frank_builder: ArticulationBuilder = generate_builder_with_options_(
            self.urdf_loader, os.path.join(ASSET_ROOT, urdf_file), opts
        )
        for link_builder in self.frank_builder.link_builders:
            link_builder.joint_record.damping = self.damping
            # Different setting for the body and gripper
            if link_builder.name.endswith("finger"):
                link_builder.joint_record.limits = (0, self.max_gripper_width / 2)
                link_builder.joint_record.friction = (
                    1.05 * sim_config["robot"]["gripper_frictions"]
                )
            else:
                link_builder.joint_record.friction = sim_config["robot"][
                    "arm_frictions"
                ]
        self.frank_builder.set_initial_pose(self.franka_pose)

    def _create_part_builders(self):
        self.part_builders: Dict[str, ActorBuilder] = {}
        self.part_default_pose: Dict[str, np.ndarray] = {}
        self.urdf_loader.load_nonconvex_collisions_from_file = True # Some meshes of parts are nonconvex
        for part in self.furniture.parts:
            part_builder = generate_builder_with_options_(
                self.urdf_loader,
                os.path.join(ASSET_ROOT, part.asset_file),
                sim_config["asset"][part.name],
            )
            part_builder.set_name(part.name)
            self.part_builders[part.name] = part_builder
            pos, ori = self.get_reset_pose_part(part)
            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = sapien.Pose()
            part_pose.set_p(
                [part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]]
            )
            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.set_q(
                np.roll(T.mat2quat(reset_ori[:3, :3]), 1, axis=-1).astype(np.float32)
            )
            part_builder.set_initial_pose(part_pose)
            part_pose_np = np.zeros(7, dtype=np.float32)

            # Store default pose for reseting
            part_pose_np[:3] = part_pose.p
            part_pose_np[3:] = part_pose.q
            self.part_default_pose[part.name] = part_pose_np

            for collision_record in part_builder.collision_records:
                collision_record.material = sapien.physx.PhysxMaterial(
                    sim_config["parts"]["friction"],
                    sim_config["parts"]["friction"],
                    self.restitution,
                )
        self.urdf_loader.load_nonconvex_collisions_from_file = False

    def _create_scenes(self):
        # %% Create Scenes
        self.scenes: List[sapien.Scene] = []
        # Some info to store
        self.franka_entities: List[sapien.physx.PhysxArticulation] = []

        self.static_obj_entites: Dict[str, List[sapien.Entity]] = {}
        self.part_entities: Dict[str, List[sapien.Entity]] = {}
        for key in self.static_obj_dict.keys():
            self.static_obj_entites[key] = []
        for part in self.furniture.parts:
            self.part_entities[part.name] = []
        self.part_actors: Dict[str, List[sapien.Entity]] = {}
        self.scene_offsets_np: np.ndarray = np.zeros(
            (self.num_envs, 3), dtype=np.float32
        )
        scene_grid_length = int(np.ceil(np.sqrt(self.num_envs)))

        if self.parallel_in_single_scene:
            self.scenes.append(
                sapien.Scene(
                    systems=[
                        self.physx_system,
                        sapien.render.RenderSystem(self.sapien_device),
                    ]  # cuda also for the rendering
                )
            )
        for i in range(self.num_envs):
            scene_x, scene_y = (
                i % scene_grid_length - scene_grid_length // 2,
                i // scene_grid_length - scene_grid_length // 2,
            )
            scene_offset = np.array(
                [
                    scene_x * self.scene_spacing,
                    scene_y * self.scene_spacing,
                    0,
                ],
                dtype=np.float32,
            )
            self.scene_offsets_np[i] = scene_offset
            if self.parallel_in_single_scene:
                scene = self.scenes[0]
            else:
                scene = sapien.Scene(
                    systems=[
                        self.physx_system,
                        sapien.render.RenderSystem(self.sapien_device),
                    ]  # cuda also for the rendering
                )
                self.physx_system.set_scene_offset(
                    scene,
                    scene_offset,
                )

            self.ground_builder.set_scene(scene)
            # Set collision group to avoid collision between different scene
            self.ground_builder.collision_groups[3] &= 0xFFFF
            self.ground_builder.collision_groups[3] |= i << 16
            if self.parallel_in_single_scene:
                tmp_name = self.ground_builder.name
                tmp_initial_p = self.ground_builder.initial_pose.p.copy()
                self.ground_builder.set_name(f"scene_{i}_{tmp_name}")
                self.ground_builder.initial_pose.set_p(scene_offset + tmp_initial_p)
            self.ground_builder.build()
            if self.parallel_in_single_scene:
                self.ground_builder.set_name(tmp_name)
                self.ground_builder.initial_pose.set_p(tmp_initial_p)

            for key, value in self.static_obj_builders.items():
                value.set_scene(scene)
                value.collision_groups[3] &= 0xFFFF
                value.collision_groups[3] |= i << 16
                if self.parallel_in_single_scene:
                    tmp_name = value.name
                    tmp_initial_p = value.initial_pose.p.copy()
                    value.set_name(f"scene_{i}_{tmp_name}")
                    value.initial_pose.set_p(scene_offset + tmp_initial_p)
                obj_entity = value.build()
                self.static_obj_entites[key].append(obj_entity)
                if self.parallel_in_single_scene:
                    value.set_name(tmp_name)
                    value.initial_pose.set_p(tmp_initial_p)

            for key, value in self.part_builders.items():
                value.set_scene(scene)
                value.collision_groups[3] &= 0xFFFF
                value.collision_groups[3] |= i << 16
                if self.parallel_in_single_scene:
                    tmp_name = value.name
                    tmp_initial_p = value.initial_pose.p.copy()
                    value.set_name(f"scene_{i}_{tmp_name}")
                    value.initial_pose.set_p(scene_offset + tmp_initial_p)
                part_entity = value.build()
                self.part_entities[key].append(part_entity)
                if self.parallel_in_single_scene:
                    value.set_name(tmp_name)
                    value.initial_pose.set_p(tmp_initial_p)

            self.frank_builder.set_scene(scene)
            if self.parallel_in_single_scene:
                tmp_initial_p = self.frank_builder.initial_pose.p.copy()
                self.frank_builder.initial_pose.set_p(
                    scene_offset + self.frank_builder.initial_pose.get_p()
                )
            for link_builder in self.frank_builder.link_builders:
                link_builder.collision_groups[3] &= 0xFFFF
                link_builder.collision_groups[3] |= i << 16
                # links without one articulation haven't been changed in this case
            franka_entity = (
                self.frank_builder.build()
            )  # Actually, this object is not Entity
            if self.parallel_in_single_scene:
                franka_entity.set_name(f"scene_{i}_franka")
                self.frank_builder.initial_pose.set_p(tmp_initial_p)
            else:
                franka_entity.set_name("franka")

            if i == 0:
                for link in franka_entity.links:
                    if link.name.endswith("k_ee_link"):
                        self.ee_link_index: int = link.get_index()
                # NOTE(Yuke): we don't use integrated pinocchino of sapien to compute jacobian of end effector
                #           we use pytorch_kinematics
                # self.franka_pinocchio_model = franka_entity.create_pinocchio_model(gravity=sim_params.gravity)
                self.franka_ee_chain = pk.build_serial_chain_from_urdf(
                    export_kinematic_chain_urdf(
                        franka_entity, force_fix_root=True
                    ).encode("utf-8"),
                    end_link_name=f"link_{self.ee_link_index}",
                ).to(dtype=torch.float32, device=self.device)
                self.franka_num_dof = franka_entity.get_dof()
                self.franka_default_dof_pos = np.zeros(
                    self.franka_num_dof, dtype=np.float32
                )
                self.franka_default_dof_pos[:7] = np.array(
                    config["robot"]["reset_joints"], dtype=np.float32
                )
                self.franka_default_dof_pos[7:] = self.max_gripper_width / 2

            # NOTE(Yuke): Refer to mani_skill utils structs articulations.py Articulation qpos
            # Direct setting with entity object is impossible for GPU Simulator
            # The setting of initial qpos must be conducted after the initialization of GPU physx_system.gpu_init()
            # franka_entity.set_qpos(self.franka_default_dof_pos)
            self.franka_entities.append(franka_entity)
            for i, joint in enumerate(franka_entity.active_joints):
                if i < 7:
                    joint.set_drive_properties(
                        stiffness=self.stiffness, damping=self.damping
                    )
                else:
                    # Direct qf control. Stiffness and dampling should be zero.
                    joint.set_drive_properties(stiffness=0, damping=0)

            self.scenes.append(scene)
        self.scene_offsets_torch = torch.from_numpy(self.scene_offsets_np).to(
            self.device
        )

    def _add_light(self):
        for light in sim_config["lights"]:
            for scene in self.scenes:
                scene.render_system.ambient_light = light["ambient"]
            for scene in self.scenes:
                entity = sapien.Entity()
                entity.name = "directional_light"
                direct_light = sapien.render.RenderDirectionalLightComponent()
                entity.add_component(direct_light)
                direct_light.set_color(light["color"])
                light_position = np.zeros(3, dtype=np.float32)
                direct_light.set_entity_pose(
                    sapien.Pose(
                        light_position,
                        sapien.math.shortest_rotation([1, 0, 0], light["direction"]),
                    )
                )

                scene.add_entity(entity)
                # NOTE(Yuke): for rendering in a single scenario
                if self.parallel_in_single_scene:
                    break

    def _load_sensors(self):
        self.sensors: Dict[str, List[sapien.render.RenderCameraComponent]] = {}
        self.sensor_keys: Dict[str, Set[str]] = {}
        # camera_obs = {}
        # This camera can access depth information as well
        set_shader(self.camera_shader)

        def create_camera(name: str, i: int) -> sapien.render.RenderCameraComponent:
            scene = self.scenes[i]
            cfg = CameraCfg()

            if name == "wrist":
                if self.resize_img:
                    cfg.fovy = 55.0
                pose = sapien.Pose(p=[-0.04, 0, -0.05])
                pose.set_rpy([0, np.radians(-70.0), 0])
                camera = sapien.render.RenderCameraComponent(
                    cfg.width, cfg.height, shader_dir=self.camera_shader.shader_dir
                )
                camera.set_fovy(cfg.fovy)
                camera.set_far(cfg.far)
                camera.set_near(cfg.near)
                camera.set_name(name)
                camera.set_gpu_pose_batch_index(
                    self.end_effector_gpu_index[i].cpu().item()
                )
                self.franka_entities[i].links[self.ee_link_index].entity.add_component(
                    camera
                )
                camera.set_local_pose(pose)
            elif name == "front":
                pose = camera_pose_from_look_at(
                    np.array([0.9, -0.00, 0.65]), np.array([-1, -0.00, 0.3])
                )
                camera = sapien.render.RenderCameraComponent(
                    cfg.width, cfg.height, shader_dir=self.camera_shader.shader_dir
                )
                camera.set_fovy(cfg.fovy)
                camera.set_far(cfg.far)
                camera.set_near(cfg.near)
                camera.set_name(name)
                camera_mount = sapien.Entity()
                camera_mount.add_component(camera)
                camera_mount.set_name(name)
                scene.add_entity(camera_mount)
                camera.set_local_pose(pose)
            elif name == "rear":
                pose = sapien.Pose(
                    p=[self.franka_pose.p[0] + 0.08, 0, self.franka_pose.p[2] + 0.2]
                )
                pose.set_rpy([0, np.radians(35), 0])
                camera = sapien.render.RenderCameraComponent(
                    cfg.width, cfg.height, shader_dir=self.shader_dir
                )
                camera.set_fovy(cfg.fovy)
                camera.set_far(cfg.far)
                camera.set_near(cfg.near)
                camera.set_name(name)
                camera_mount = sapien.Entity()
                camera_mount.add_component(camera)
                camera_mount.set_name(name)
                scene.add_entity(camera_mount)
            return camera

        self.camera_names_dict = {"1": "wrist", "2": "front"}
        if not self.parallel_in_single_scene:
            for i, scene in enumerate(self.scenes):
                for k in self.obs_keys:
                    if k.startswith("color"):
                        camera_name = self.camera_names_dict[k[-1]]
                        obs_key = "color"
                    elif k.startswith("depth"):
                        camera_name = self.camera_names_dict[k[-1]]
                        obs_key = "depth"
                    else:
                        continue
                    if camera_name not in self.sensor_keys:
                        self.sensor_keys[camera_name] = set()
                    self.sensor_keys[camera_name].add(
                        self.camera_shader.obs_keys_2_texture_name[obs_key]
                    )
                    if camera_name not in self.sensors:
                        self.sensors[camera_name] = []

                    if len(self.sensors[camera_name]) <= i:
                        self.sensors[camera_name].append(create_camera(camera_name, i))

    def _init_render(self):
        for rb, gpu_pose_index in self._get_render_bodies():
            if rb is not None:
                for shape in rb.render_shapes:
                    shape.set_gpu_pose_batch_index(gpu_pose_index)
        if self.num_envs == 1:
            self.render_system_group = self.scenes[0].render_system
            self.render_system_group.step()
            for sensors in self.sensors.values():
                sensors[0].take_picture()
            return
            
        self.render_system_group = sapien.render.RenderSystemGroup(
            [scene.render_system for scene in self.scenes]
        )
        self.render_system_group.set_cuda_poses(self.physx_system.cuda_rigid_body_data)

        self.sensor_groups: Dict[str, sapien.render.RenderCameraGroup] = {}
        # init sensors
        for name, sensor in self.sensors.items():
            self.sensor_groups[name] = self.render_system_group.create_camera_group(
                sensor, picture_names=list(self.sensor_keys[name])
            )
        self.render_system_group.update_render()

    def get_sensor_obs(self) -> Dict[str, torch.Tensor]:
        """Obtain observation from Sensor

        Returns:
            Dict[str, torch.Tensor]: Dictionary of Data. "color":[H x W x 3], "depth":[H x W x 1]
        """
        sensor_obs = {}
        for camera_name, sensors in self.sensors.items():
            print(len(sensors))

        if isinstance(self.render_system_group, sapien.render.RenderSystem):
            sensor_raw_obs = {
            camera_name: {
                picture_name: sensors[0].get_picture_cuda(picture_name).torch().clone()[None, ...]
                for picture_name in self.sensor_keys[camera_name]
            }
            for camera_name, sensors in self.sensors.items()
        }
        else:
            sensor_raw_obs = {
                camera_name: {
                    picture_name: sensor_group.get_picture_cuda(picture_name).torch()
                    for picture_name in self.sensor_keys[camera_name]
                }
                for camera_name, sensor_group in self.sensor_groups.items()
            }

        for key in self.obs_keys:
            if key.startswith("color"):
                camera_name = self.camera_names_dict.get(key[-1])
                obs_key = "color"
            elif key.startswith("depth"):
                camera_name = self.camera_names_dict.get(key[-1])
                obs_key = "depth"
            else:
                continue
            if camera_name is None:
                # Warning might be included here.
                continue
            sensor_obs[key] = self.camera_shader.output_transform[
                self.camera_shader.obs_keys_2_texture_name[obs_key]
            ](
                sensor_raw_obs[camera_name][
                    self.camera_shader.obs_keys_2_texture_name[obs_key]
                ]
            )[obs_key]
        return sensor_obs

    def _init_viewer(self):
        # If parallel_in_one_scene is enabled, all articulations and actors will be loaded
        # in one scene and rendered in only one render system
        # If it is disabled, only one scene containing one instance of simulation will be
        # display in the viewer
        sapien.render.set_viewer_shader_dir(self.viewer_shader.shader_dir)
        set_shader(self.viewer_shader)
        self.viewer: Optional[Viewer] = Viewer()
        self.viewer.set_scene(self.scenes[0])
        control_window = self.viewer.control_window
        control_window.show_joint_axes = False
        control_window.show_camera_linesets = False

        pose = camera_pose_from_look_at(
            np.array([0.97, 0, 0.74]), np.array([-1, 0, 0.62])
        )
        self.viewer.set_camera_pose(pose)

        # Initial rendering
        self.physx_system.sync_poses_gpu_to_cpu()
        self.viewer.render()

    def _get_render_bodies(self) -> List[Tuple[sapien.render.RenderBodyComponent, int]]:
        """Get the render bodies and the indices for initing GPU rendering. Static Objects are not included

        Returns:
            List[Tuple[sapien.render.RenderBodyComponent, int]]: List of pair (RenderBodyComponent, index)
        """
        render_bodies = []
        render_bodies += [
            (
                part_entity.find_component_by_type(sapien.render.RenderBodyComponent),
                part_entity.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                ).gpu_pose_index,
            )
            for part_entities in self.part_entities.values()
            for part_entity in part_entities
        ]
        render_bodies += [
            (
                link.entity.find_component_by_type(sapien.render.RenderBodyComponent),
                link.gpu_pose_index,
            )
            for franka_entity in self.franka_entities
            for link in franka_entity.links
        ]

        return render_bodies

    def get_reset_pose_part(self, part: Part) -> Tuple[np.ndarray, np.ndarray]:
        """Get reset pose of the furniture part

        Args:
            part (Part): Part to obtain reset pose

        Returns:
            Tuple[np.ndarray, np.ndarray]: [position, orientation] of the reset pose. orientation is xyzw quaternion.
        """
        if self.init_assembled:
            if part.name == "chair_seat":
                # Special case handling for chair seat since the assembly of chair back is not available from initialized pose.
                part.reset_pos = [[0, 0.16, -0.035]]
                part.reset_ori = [rot_mat([np.pi, 0, 0], hom=True)]
            attach_to: Optional[Part] = None
            for assemble_pair in self.furniture.should_be_assembled:
                if part.part_idx == assemble_pair[1]:
                    attach_to = self.furniture.parts[assemble_pair[0]]
                    break
            if attach_to is not None:
                attach_part_pos = self.furniture.parts[attach_to.part_idx].reset_pos[0]
                attach_part_ori = self.furniture.parts[attach_to.part_idx].reset_ori[0]
                attach_part_pose = get_mat(attach_part_pos, attach_part_ori)
                if part.default_assembled_pose is not None:
                    pose = attach_part_pose @ part.default_assembled_pose
                    pos = pose[:3, 3]
                    ori = T.to_hom_ori(pose[:3, :3])
                else:
                    pos = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0][:4, 3]
                    )
                    pos = pos[:3]
                    ori = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0]
                    )
                part.reset_pos[0] = pos
                part.reset_ori[0] = ori
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
            return pos, ori
        pos = part.reset_pos[self.from_skill]
        ori = part.reset_ori[self.from_skill]
        return pos, ori

    def _get_ee_pose(self) -> torch.Tensor:
        """Obtain the Pose of End Effector in world frame (Rotation is in the world frame.
        Position is the relative position w.r.t the base)
        You should only read pose after you call `gpu_fetch_*` functions

        Returns:
            torch.Tensor: [num_envs, 7], orientation is wxyz quaternion.
        """
        end_effector_pose = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        end_effector_pose[:] = self.physx_system.cuda_rigid_body_data.torch()[
            self.end_effector_gpu_index, :7
        ]
        end_effector_pose[:, :3] -= self.physx_system.cuda_rigid_body_data.torch()[
            self.root_link_gpu_index, :3
        ]
        return end_effector_pose

    def get_ee_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtain the pose of End Effector on GPU

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Position and Quaternion (xyzw)
        """

        ee_pose = self._get_ee_pose()
        return ee_pose[:, :3], ee_pose[:, 3:7].roll(-1, dims=-1)

    def get_ee_lin_vel(self) -> torch.Tensor:
        """Get linear velocity of end effector

        Returns:
            torch.Tensor: Linear velocity Shape: [num_envs ,3]
        """
        ee_lin_vel = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        ee_lin_vel[:] = self.physx_system.cuda_rigid_body_data.torch()[
            self.end_effector_gpu_index, 7:10
        ]
        return ee_lin_vel

    def get_ee_rot_vel(self) -> torch.Tensor:
        """Get rotational velocity of end effector

        Returns:
            torch.Tensor: Rotational velocity Shape: [num_envs ,3]
        """
        ee_rot_vel = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        ee_rot_vel[:] = self.physx_system.cuda_rigid_body_data.torch()[
            self.end_effector_gpu_index, 10:13
        ]
        return ee_rot_vel

    def get_qpos(self) -> torch.Tensor:
        """
        Obtain qpos of franka
        You should only read qpos after you call `gpu_fetch_*` functions
        """
        qpos = torch.zeros(
            (self.num_envs, self.franka_num_dof),
            dtype=torch.float32,
            device=self.device,
        )
        qpos[:] = self.physx_system.cuda_articulation_qpos.torch()[
            self.franka_gpu_index, : self.franka_num_dof
        ]
        return qpos

    def get_qvel(self) -> torch.Tensor:
        """Get velocities of dof in all envs

        Returns:
            torch.Tensor: Tensor of Velocites
        """
        qvel = torch.zeros(
            (self.num_envs, self.franka_num_dof),
            dtype=torch.float32,
            device=self.device,
        )
        qvel[:] = self.physx_system.cuda_articulation_qvel.torch()[
            self.franka_gpu_index, : self.franka_num_dof
        ]
        return qvel

    def get_robot_state(self) -> Dict[str, torch.Tensor]:
        # Robot State
        qpos = self.get_qpos()
        joint_positions = qpos[:, :7]
        joint_velocities = self.get_qvel()[:, :7]
        # NOTE(Yuke): This is applied joint torch
        joint_torques = self.stiffness * (
            self.physx_system.cuda_articulation_target_qpos.torch()[:, :7]
            - joint_positions
        ) + self.damping * (
            self.physx_system.cuda_articulation_target_qvel.torch()[:, :7]
            - joint_velocities
        )

        # State of end effector
        ee_pose = self._get_ee_pose()
        ee_pos = ee_pose[:, :3]
        ee_quat = ee_pose[:, 3:].roll(-1, dims=-1)  # wxyz to xyzw
        ee_lin_vel = self.get_ee_lin_vel()
        ee_rot_vel = self.get_ee_rot_vel()

        gripper_width = qpos[:, 7:8] + qpos[:, 8:9]

        robot_state = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_pos_vel": ee_lin_vel,
            "ee_ori_vel": ee_rot_vel,
            "gripper_width": gripper_width,
            "gripper_finger_1_pos": qpos[:, 7:8],
            "gripper_finger_2_pos": qpos[:, 8:9],
        }
        return {key: robot_state[key] for key in self.robot_state_keys}

    def get_parts_poses(self, sim_coord=False, robot_coord=False):
        """Get furniture parts poses in the AprilTag frame.

        Args:
            sim_coord: If True, return the poses in the simulator coordinate. Otherwise, return the poses in the AprilTag coordinate.

        Returns:
            parts_poses: (num_envs, num_parts * pose_dim). The poses of all parts in the AprilTag frame.
            founds: (num_envs, num_parts). Always 1 since we don't use AprilTag for detection in simulation.
        """
        parts_poses = self.physx_system.cuda_rigid_body_data.torch()[
            self.parts_gpu_index_tensor, :7
        ].clone()
        if self.parallel_in_single_scene:
            parts_poses[..., :3] -= self.scene_offsets_torch.unsqueeze(1)
        # parts_poses Shape: (num_envs, num_parts, 7)
        parts_poses[..., 3:7] = parts_poses[..., 3:7].roll(-1, dims=-1)  # wxyz to xyzw
        if sim_coord:
            return parts_poses.reshape(self.num_envs, -1)

        if robot_coord:
            robot_coord_poses = self.sim_pose_to_robot_pose(parts_poses)
            return robot_coord_poses.view(self.num_envs, -1)

        april_coord_poses = self.sim_pose_to_april_pose(parts_poses)
        parts_poses = april_coord_poses.view(self.num_envs, -1)

        return parts_poses

    def get_obstacle_pose(self, sim_coord=False, robot_coord=False):
        obstacle_front_poses = self.physx_system.cuda_rigid_body_data.torch()[
            self.obstacle_gpu_index["obstacle_front"].unsqueeze(1), :7
        ].clone()
        # NOTE:(Yuke) offset should also be considered only in parallel in single scene mode
        if self.parallel_in_single_scene:
            obstacle_front_poses[..., :3] -= self.scene_offsets_torch.unsqueeze(1)
        obstacle_front_poses[..., 3:7] = obstacle_front_poses[..., 3:7].roll(
            -1, dims=-1
        )
        if sim_coord:
            return obstacle_front_poses.reshape(self.num_envs, -1)

        if robot_coord:
            robot_coord_poses = self.sim_pose_to_robot_pose(obstacle_front_poses)
            return robot_coord_poses.view(self.num_envs, -1)
        april_coord_poses = self.sim_pose_to_april_pose(obstacle_front_poses)
        return april_coord_poses.view(self.num_envs, -1)

    def get_assembly_action(self) -> torch.Tensor:
        """Scripted furniture assembly logic.

        Returns:
            Tuple (action for the assembly task, skill complete mask)
        """
        assert self.num_envs == 1  # Only support one environment for now.
        if self.furniture_name not in ["one_leg", "cabinet", "lamp", "round_table"]:
            raise NotImplementedError(
                "[one_leg, cabinet, lamp, round_table] are supported for scripted agent"
            )

        if self.assemble_idx > len(self.furniture.should_be_assembled):
            return torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=self.device)

        ee_pos, ee_quat = self.get_ee_pose()
        gripper_width = self.gripper_width()
        ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()

        if self.move_neutral:
            if ee_pos[2] <= 0.15 - 0.01:
                gripper = torch.tensor([-1], dtype=torch.float32, device=self.device)
                goal_pos = torch.tensor(
                    [ee_pos[0], ee_pos[1], 0.15], device=self.device
                )
                delta_pos = goal_pos - ee_pos
                delta_quat = torch.tensor([0, 0, 0, 1], device=self.device)
                action = torch.concat([delta_pos, delta_quat, gripper])
                return action.unsqueeze(0), 0
            else:
                self.move_neutral = False
        part_idx1, part_idx2 = self.furniture.should_be_assembled[self.assemble_idx]

        part1: Part = self.furniture.parts[part_idx1]
        part1_name = part1.name
        part1_pose = C.to_homogeneous(
            self.rb_states[self.parts_gpu_index[part1_name][0], :3],
            C.quat2mat(self.rb_states[self.parts_gpu_index[part1_name][0], 3:7]),
        )
        part2: Part = self.furniture.parts[part_idx2]
        part2_name = part2.name
        part2_pose = C.to_homogeneous(
            self.rb_states[self.parts_gpu_index[part2_name][0], :3],
            C.quat2mat(self.rb_states[self.parts_gpu_index[part2_name][0], 3:7]),
        )
        rel_pose: torch.Tensor = torch.linalg.inv(part1_pose) @ part2_pose
        assembled_rel_poses = self.furniture.assembled_rel_poses[(part_idx1, part_idx2)]
        if self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
            self.assemble_idx += 1
            self.move_neutral = True
            return (
                torch.tensor(
                    [0, 0, 0, 0, 0, 0, 1, -1], dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                1,
            )  # Skill complete is always 1 when assembled.
        parts_gpu_index = self.parts_gpu_index.copy()
        parts_gpu_index.update(self.obstacle_gpu_index)
        if not part1.pre_assemble_done:
            goal_pos, goal_ori, gripper, skill_complete = part1.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                parts_gpu_index,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
            )
        elif not part2.pre_assemble_done:
            goal_pos, goal_ori, gripper, skill_complete = part2.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                parts_gpu_index,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
            )
        else:
            goal_pos, goal_ori, gripper, skill_complete = self.furniture.parts[
                part_idx2
            ].fsm_step(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                parts_gpu_index,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
                self.furniture.parts[part_idx1].name,
            )

        delta_pos = goal_pos - ee_pos

        # Scale translational action.
        delta_pos_sign = delta_pos.sign()
        delta_pos = torch.abs(delta_pos) * 2
        for i in range(3):
            if delta_pos[i] > 0.03:
                delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * np.random.normal(1.5, 0.1)
        delta_pos = delta_pos * delta_pos_sign

        # Clamp too large action.
        max_delta_pos = 0.11 + 0.01 * torch.rand(3, device=self.device)
        max_delta_pos[2] -= 0.04
        delta_pos = torch.clamp(delta_pos, min=-max_delta_pos, max=max_delta_pos)

        delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), goal_ori)
        # Add random noise to the action.
        if (
            self.furniture.parts[part_idx2].state_no_noise()
            and np.random.random() < 0.50
        ):
            delta_pos = torch.normal(delta_pos, 0.005)
            delta_quat = C.quat_multiply(
                delta_quat,
                torch.tensor(
                    T.axisangle2quat(
                        [
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                        ]
                    ),
                    device=self.device,
                ),
            ).to(self.device)
        action = torch.concat([delta_pos, delta_quat, gripper])
        return action.unsqueeze(0), skill_complete

    def get_observation(self) -> dict:
        obs = {}

        obs["robot_state"] = self.get_robot_state()
        if self.render_system_group is not None:
            obs.update(self.get_sensor_obs().items())

        if self.include_parts_poses:
            # Part poses in AprilTag.
            parts_poses = self.get_parts_poses(
                sim_coord=False, robot_coord=self.parts_poses_in_robot_frame
            )
            obstacle_poses = self.get_obstacle_pose(
                sim_coord=False, robot_coord=self.parts_poses_in_robot_frame
            )

            obs["parts_poses"] = torch.cat([parts_poses, obstacle_poses], dim=1)

        return obs

    def _init_sim(self):
        # torch seed can be added before
        self.physx_system.gpu_init()

        # NOTE(Yuke): Setting the initial qpos must be complete after gpu_init
        #   and according to my understanding we should fetch first. Otherwise, the values stored
        #   in the buffer are undefined
        # Fetch the preloaded info into the GPU buffer
        # self.physx_system.gpu_fetch_rigid_dynamic_data()
        # self.physx_system.gpu_fetch_articulation_link_pose()
        self._fetch_all()
        self._config_parts()
        # set Vel of all actors to zero
        self.physx_system.cuda_rigid_body_data.torch()[:, 7:] = torch.zeros_like(
            self.physx_system.cuda_rigid_body_data.torch()[:, 7:]
        )
        self.physx_system.cuda_rigid_body_force.torch()[:, :] = 0
        self.physx_system.cuda_rigid_body_torque.torch()[:, :] = 0

        self._config_franka()

        self.obstacle_gpu_index = {
            obj_name: torch.tensor(
                [
                    entity.find_component_by_type(
                        sapien.physx.PhysxRigidDynamicComponent
                    ).get_gpu_index()
                    for entity in entities
                ]
            )
            for obj_name, entities in self.static_obj_entites.items()
            if obj_name.startswith("obstacle")
        }

        self.physx_system.gpu_apply_rigid_dynamic_data()
        self.physx_system.gpu_apply_rigid_dynamic_force()
        self.physx_system.gpu_apply_rigid_dynamic_torque()
        self.physx_system.gpu_apply_articulation_root_pose()
        self.physx_system.gpu_apply_articulation_root_velocity()
        self.physx_system.gpu_apply_articulation_qpos()
        self.physx_system.gpu_apply_articulation_qvel()
        self.physx_system.gpu_update_articulation_kinematics()  #  ensure all updates have been applied to the system

    def _init_ctrl(self):
        self.step_ctrl = diffik_factory(
            real_robot=False,
            pos_scalar=self.pos_scalar,
            rot_scalar=self.rot_scalar,
        )
        self.ctrl_started = True

    def _config_franka(self):
        # Since franka is fixed. No need to reset the root Pose
        self.franka_gpu_index = torch.tensor(
            [frank_entity.get_gpu_index() for frank_entity in self.franka_entities],
            dtype=torch.int32,
            device=self.device,
        )
        self.physx_system.cuda_articulation_qpos.torch()[
            self.franka_gpu_index, : self.franka_num_dof
        ] = torch.from_numpy(self.franka_default_dof_pos).to(self.device)
        self.physx_system.cuda_articulation_qvel.torch()[:, :] = torch.zeros_like(
            self.physx_system.cuda_articulation_qvel.torch()
        ).to(self.device)
        # NOTE(Yuke): Set the target position to the default position as well.
        #   To avoid the error in some scenarios. For example, some joints directly are controlled
        #   with force and others are controlled with Physx PD Controller. After initialization, if someone
        #   only modifies force without giving the target qpos, the robot can bounce to some unknow
        #   state because of the existence of PD Controller of Physx ()
        self.physx_system.cuda_articulation_target_qpos.torch()[
            self.franka_gpu_index, : self.franka_num_dof
        ] = torch.from_numpy(self.franka_default_dof_pos).to(self.device)
        self.physx_system.cuda_articulation_link_data.torch()[:, 7:] = torch.zeros_like(
            self.physx_system.cuda_articulation_link_data.torch()[:, 7:]
        ).to(self.device)
        self.root_link_gpu_index = torch.zeros((self.num_envs), dtype=torch.int32)
        self.end_effector_gpu_index = torch.zeros((self.num_envs), dtype=torch.int32)
        for i, frank_entity in enumerate(self.franka_entities):
            for link in frank_entity.links:
                if link.name.endswith("panda_link0"):
                    self.root_link_gpu_index[i] = link.get_gpu_pose_index()
                elif link.name.endswith("k_ee_link"):
                    self.end_effector_gpu_index[i] = link.get_gpu_pose_index()

    def _config_parts(self):
        self.parts_gpu_index: Dict[str, torch.Tensor] = {}
        for key, value in self.part_entities.items():
            part_gpu_index = torch.tensor(
                [
                    part_entity.find_component_by_type(
                        sapien.physx.PhysxRigidDynamicComponent
                    ).get_gpu_index()
                    for part_entity in value
                ]
            )
            self.parts_gpu_index[key] = part_gpu_index
            part_pose_np = np.zeros((self.num_envs, 7), dtype=np.float32)
            part_pose_np[:] = self.part_default_pose[key]  # broadcast
            if self.parallel_in_single_scene:
                part_pose_np[:, :3] += self.scene_offsets_np
            self.physx_system.cuda_rigid_body_data.torch()[part_gpu_index, :7] = (
                torch.from_numpy(part_pose_np).to(self.device)
            )
        # self.parts_gpu_index_tensor: Shape (num_envs, num_parts)
        self.parts_gpu_index_tensor = self.furniture_rb_indices = torch.stack(
            [self.parts_gpu_index[part.name] for part in self.furniture.parts],
            dim=0,
        ).T.to(self.device)

    def reset(self, env_idxs: Optional[torch.Tensor] = None):
        print("In orignal reset")
        if env_idxs is None:
            env_idxs = torch.arange(
                self.num_envs, device=self.device, dtype=torch.int32
            )
        for i in env_idxs:
            self.reset_env(i)
        self.step_ctrl.reset()
        self.physx_system.step()
        self._fetch_all()
        self.update_render()
        self.assemble_idx = 0
        obs = self.get_observation()

        self.reward = torch.zeros((self.num_envs, 1), dtype=torch.float32)
        self.done = torch.zeros((self.num_envs, 1), dtype=torch.bool)
        return obs

    def refresh(self):
        self.physx_system.step()
        self._fetch_all()
        self.update_render()

    def reset_env_to(self, env_idx: int, state: dict):
        """Reset to a specific state. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx:int: Environment index.
            state: A dict containing the state of the environment.
        """
        self.furnitures[env_idx].reset()
        dof_pos = np.concatenate(
            [
                state["robot_state"]["joint_positions"],
                np.array(
                    [
                        state["robot_state"]["gripper_finger_1_pos"],
                        state["robot_state"]["gripper_finger_2_pos"],
                    ]
                ),
            ],
        )

        self._reset_franka(
            env_idx,
            torch.from_numpy(dof_pos).to(dtype=torch.float32, device=self.device),
        )
        self._reset_parts(env_idx, state["parts_poses"])
        self.env_steps[env_idx] = 0
        self.move_neutral = False

    def reset_env(
        self, env_idx: int, reset_franka: bool = True, reset_parts: bool = True
    ):
        """Reset the environment

        Args:
            env_idx (int): The index of environment
            reset_franka (bool, optional): Whether to reset franka robot or not. Defaults to True.
            reset_parts (bool, optional): Whether to reset the part poses. Defaults to True.
        """
        furniture: Furniture = self.furnitures[env_idx]
        furniture.reset()
        if self.randomness == Randomness.LOW and not self.init_assembled:
            furniture.randomize_init_pose(
                self.from_skill, pos_range=[-0.015, 0.015], rot_range=15
            )

        if self.randomness == Randomness.LOW and not self.init_assembled:
            furniture.randomize_init_pose(
                self.from_skill, pos_range=[-0.015, 0.015], rot_range=15
            )

        if self.randomness == Randomness.MEDIUM:
            furniture.randomize_init_pose(self.from_skill)
        elif self.randomness == Randomness.HIGH:
            furniture.randomize_high(self.high_random_idx)
        if reset_parts:
            self._reset_parts(env_idx)
        if reset_franka:
            self._reset_franka(env_idx)

        self.env_steps[env_idx] = 0
        self.already_assembled[env_idx] = 0
        self.move_neutral = False

    def _reset_franka(
        self, env_idx: Optional[int] = None, dof_pos: Optional[torch.Tensor] = None
    ):
        if dof_pos is None:
            dof_pos = torch.from_numpy(self.franka_default_dof_pos).to(
                dtype=torch.float32, device=self.device
            )
        if env_idx is None:
            franka_gpu_index = self.franka_gpu_index
        else:
            franka_gpu_index = self.franka_gpu_index[env_idx]
        self.physx_system.cuda_articulation_qpos.torch()[
            franka_gpu_index, : self.franka_num_dof
        ] = dof_pos
        self.physx_system.cuda_articulation_qvel.torch()[
            franka_gpu_index, : self.franka_num_dof
        ] = torch.zeros_like(
            self.physx_system.cuda_articulation_qvel.torch()[
                franka_gpu_index, : self.franka_num_dof
            ],
            device=self.device,
        )

        # Set the target as well, for PD controller
        self.physx_system.cuda_articulation_target_qpos.torch()[
            franka_gpu_index, : self.franka_num_dof
        ] = dof_pos
        self.physx_system.cuda_articulation_target_qvel.torch()[
            franka_gpu_index, : self.franka_num_dof
        ] = torch.zeros_like(
            self.physx_system.cuda_articulation_target_qvel.torch()[
                franka_gpu_index, : self.franka_num_dof
            ],
            device=self.device,
        )
        self.physx_system.cuda_articulation_qf.torch()[
            franka_gpu_index, : self.franka_num_dof
        ] = torch.zeros_like(
            self.physx_system.cuda_articulation_qf.torch()[
                franka_gpu_index, : self.franka_num_dof
            ],
            device=self.device,
        )
        # Apply Changes
        self.physx_system.gpu_apply_articulation_qpos()
        self.physx_system.gpu_apply_articulation_qvel()
        self.physx_system.gpu_update_articulation_kinematics()
        self.physx_system.gpu_apply_articulation_qf()
        self.physx_system.gpu_apply_articulation_root_pose()
        self.physx_system.gpu_apply_articulation_root_velocity()
        self.physx_system.gpu_apply_articulation_target_position()
        self.physx_system.gpu_apply_articulation_target_velocity()

    def _reset_parts(
        self,
        env_idx: int,
        parts_poses: Optional[np.ndarray] = None,
        skip_set_state: bool = False,
    ):
        """Resets furniture parts to the initial pose.
        part_poses: quaternion wxyz"""
        for part_idx, part in enumerate(self.furnitures[env_idx].parts):
            # Use the given pose.
            if parts_poses is not None:
                part_pose = parts_poses[part_idx * 7 : (part_idx + 1) * 7]

                pos = part_pose[:3]
                ori = T.to_homogeneous(
                    [0, 0, 0], T.quat2mat(np.roll(part_pose[3:], shift=-1, axis=-1))
                )  # Dummy zero position.
            else:
                pos, ori = self.get_reset_pose_part(part)

            # NOTE(Yuke): recalculation instead of directly using self.part_default_pose,
            #          since self._get_reset_pose_part can obtain pos and ori with randomness
            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = sapien.Pose()
            part_pose.set_p(
                [part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]]
            )
            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.set_q(
                np.roll(T.mat2quat(reset_ori[:3, :3]), 1, axis=-1).astype(np.float32)
            )
            idxs = self.parts_gpu_index[part.name][env_idx]

            # Load the Pose into buffer (offset)
            if self.parallel_in_single_scene:
                self.physx_system.cuda_rigid_body_data.torch()[idxs, :3] = torch.tensor(
                    part_pose.p + self.scene_offsets_np[env_idx], device=self.device
                )
            else:
                self.physx_system.cuda_rigid_body_data.torch()[idxs, :3] = torch.tensor(
                    part_pose.p, device=self.device
                )
            self.physx_system.cuda_rigid_body_data.torch()[idxs, 3:7] = torch.tensor(
                part_pose.q,
                device=self.device,
            )
            # linear vel and rot vel to zero
            self.physx_system.cuda_rigid_body_data.torch()[idxs, 7:] = 0
            self.physx_system.cuda_rigid_body_force.torch()[:, :] = 0
            self.physx_system.cuda_rigid_body_torque.torch()[:, :] = 0

        # Get the obstacle poses, last 7 numbers in the parts_poses tensor
        if parts_poses is not None:
            obstacle_pose = parts_poses[-7:]
            pos = obstacle_pose[:3]
            ori = T.to_homogeneous([0, 0, 0], T.quat2mat(obstacle_pose[3:]))
            # Convert the obstacle pose from AprilTag to simulator coordinate system
            obstacle_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            obstacle_pose = sapien.Pose()
            obstacle_pose.set_p(
                (
                    obstacle_pose_mat[0, 3],
                    obstacle_pose_mat[1, 3],
                    obstacle_pose_mat[2, 3],
                )
            )
            reset_ori = self.april_coord_to_sim_coord(ori)
            obstacle_pose.set_q(
                np.roll(T.mat2quat(reset_ori[:3, :3]), shift=1, axis=-1).astype(
                    np.float32
                )
            )
        else:
            obstacle_pose = self.obstacle_front_pose

        # Calculate the offsets for the front and side obstacles
        obstacle_right_offset = np.array((-0.075, -0.175, 0), dtype=np.float32)
        obstacle_left_offset = np.array((-0.075, 0.175, 0), dtype=np.float32)

        # Write to GPU buffer. Since obstacles are static objects, no need to reset velocity
        if self.parallel_in_single_scene:
            self.physx_system.cuda_rigid_body_data.torch()[
                self.obstacle_gpu_index["obstacle_front"][env_idx], :3
            ] = torch.from_numpy(obstacle_pose.p + self.scene_offsets_np[env_idx]).to(
                dtype=torch.float32, device=self.device
            )

            self.physx_system.cuda_rigid_body_data.torch()[
                self.obstacle_gpu_index["obstacle_right"][env_idx], :3
            ] = torch.from_numpy(
                obstacle_pose.p + obstacle_right_offset + self.scene_offsets_np[env_idx]
            ).to(dtype=torch.float32, device=self.device)

            self.physx_system.cuda_rigid_body_data.torch()[
                self.obstacle_gpu_index["obstacle_left"][env_idx], :3
            ] = torch.from_numpy(
                obstacle_pose.p + obstacle_left_offset + self.scene_offsets_np[env_idx]
            ).to(dtype=torch.float32, device=self.device)
        else:
            self.physx_system.cuda_rigid_body_data.torch()[
                self.obstacle_gpu_index["obstacle_front"][env_idx], :3
            ] = torch.from_numpy(obstacle_pose.p).to(
                dtype=torch.float32, device=self.device
            )

            self.physx_system.cuda_rigid_body_data.torch()[
                self.obstacle_gpu_index["obstacle_right"][env_idx], :3
            ] = torch.from_numpy(obstacle_pose.p + obstacle_right_offset).to(
                dtype=torch.float32, device=self.device
            )

            self.physx_system.cuda_rigid_body_data.torch()[
                self.obstacle_gpu_index["obstacle_left"][env_idx], :3
            ] = torch.from_numpy(obstacle_pose.p + obstacle_left_offset).to(
                dtype=torch.float32, device=self.device
            )

        # Quaternion unused

        if skip_set_state:
            return

        # Reset root state for actors in a single env
        self.physx_system.gpu_apply_rigid_dynamic_data()

    def step_viewer(self):
        if self.viewer is None:
            return
        self.physx_system.sync_poses_gpu_to_cpu()
        # self.scenes[0].update_render()  # This is not needed for viewer rendering
        self.viewer.render()

    def _fetch_all(self):
        # fetch data from the Physx
        self.physx_system.gpu_fetch_rigid_dynamic_data()
        self.physx_system.gpu_fetch_articulation_link_pose()
        self.physx_system.gpu_fetch_articulation_link_velocity()
        self.physx_system.gpu_fetch_articulation_link_incoming_joint_forces()
        self.physx_system.gpu_fetch_articulation_qpos()
        self.physx_system.gpu_fetch_articulation_qvel()
        self.physx_system.gpu_fetch_articulation_qacc()
        self.physx_system.gpu_fetch_articulation_target_qpos()
        self.physx_system.gpu_fetch_articulation_target_qvel()

    def _apply_all(self):
        self.physx_system.gpu_apply_rigid_dynamic_data()
        self.physx_system.gpu_apply_articulation_qpos()
        self.physx_system.gpu_apply_articulation_qvel()
        self.physx_system.gpu_apply_articulation_qf()
        self.physx_system.gpu_apply_articulation_root_pose()
        self.physx_system.gpu_apply_articulation_root_velocity()
        self.physx_system.gpu_apply_articulation_target_position()
        self.physx_system.gpu_apply_articulation_target_velocity()

    def get_jacobian_ee(self, qpos: torch.Tensor) -> torch.Tensor:
        # Pinocchio can only use loop to compute jacobian for multiple envs
        # jacobian_ee = self.franka_pinocchio_model.compute_single_link_local_jacobian(qpos=qpos.cpu().numpy(), index=self.ee_link_index)
        jacobian_ee = torch.zeros(
            (self.num_envs, 6, self.franka_num_dof),
            dtype=torch.float32,
            device=self.device,
        )
        jacobian_ee[:, :, :7] = self.franka_ee_chain.jacobian(
            qpos[:, :7]
        )  # last 2 dim is for gripper
        return jacobian_ee

    @torch.no_grad()
    def step(
        self, action: torch.Tensor, sample_perturbations: bool = False
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, dict]:
        obs = self.get_observation()
        self.update_action(action)
        self._apply_all()
        for _ in range(self.sim_steps):
            self.physx_system.step()
            self._fetch_all()
            self.update_render()
        reward = self._reward()
        done = self._done()
        self.env_steps += 1
        if sample_perturbations:
            self._random_perturbation_of_parts(
                self.max_force_magnitude,
                self.max_torque_magnitude,
            )
        return (
            obs,
            reward,
            done,
            {"obs_success": True, "action_success": True},
        )

    def update_action(self, action: torch.Tensor) -> torch.Tensor:
        """Calculate the raw action variables and load action into the GPU buffer.

        Args:
            action (torch.Tensor): action of the franka in all envs. Shape: [num_envs, 8]  (Pose of end effector + Gripper Control: 7 + 1, Quaternion format: xyzw)
        """

        action = torch.clamp(action, self.act_low, self.act_high)

        ee_pose = self._get_ee_pose()
        if self.__act_rot_repr == 0:
            # Real part is the last element in the quaternion.
            action_quat_xyzw = action[:, 3:7]

        elif self.__act_rot_repr == 1:
            rot_6d = action[:, 3:9]
            rot_mat = C.rotation_6d_to_matrix(rot_6d)
            # Real part is the first element in the quaternion.
            action_quat_xyzw = C.matrix_to_quaternion_xyzw(rot_mat)

        else:
            # Convert axis angle to quaternion.
            action_quat_xyzw = C.matrix_to_quaternion_xyzw(
                C.axis_angle_to_matrix(action[:, 3:6])
            )

        # Delta Control
        if self.__action_type == 1:
            goal_pose = ee_pose.clone()
            goal_pose[:, :3] += action[:, :3]
            goal_pose[:, 3:] = C.quaternion_multiply(
                goal_pose[:, 3:].roll(-1, dims=-1), action_quat_xyzw
            ).roll(1, dims=-1)
        # Absolute Control
        elif self.__action_type == 0:
            goal_pose = torch.zeros(
                (self.num_envs, 7), dtype=torch.float32, device=self.device
            )
            goal_pose[:, :3] = action[:, :3]
            goal_pose[:, 3:7] = action_quat_xyzw.roll(1, dims=-1)

        # SAPIEN uses wxyz, diffik uses xyzw
        self.step_ctrl.set_goal(goal_pose[:, :3], goal_pose[:, 3:7].roll(-1, dims=-1))
        target_qpos = torch.zeros(
            (self.num_envs, self.franka_num_dof),
            dtype=torch.float32,
            device=self.device,
        )
        target_qf = torch.zeros(
            (self.num_envs, self.franka_num_dof),
            dtype=torch.float32,
            device=self.device,
        )
        gripper_action = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )

        grasp = action[:, -1]

        # Avoid Oscilation of Grip
        grip_sep = torch.where(
            (torch.sign(grasp) != torch.sign(self.last_grasp))
            & (torch.abs(grasp) > self.grasp_margin),
            torch.where(
                grasp < 0,
                self.max_gripper_width,
                torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            ),
            torch.where(
                self.last_grasp < 0,
                self.max_gripper_width,
                torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            ),
        )
        self.last_grasp = grasp
        gripper_action[:, -1] = grip_sep
        qpos = self.get_qpos()

        self.jacobian_ee = self.get_jacobian_ee(qpos)
        state_dict = {
            "ee_pos": ee_pose[:, :3],
            "ee_quat": ee_pose[:, 3:7].roll(-1, dims=-1),
            "joint_positions": qpos[:, :7],
            "jacobian_diffik": self.jacobian_ee[:, :, :7],
        }

        gripper_action_mask = (grip_sep > 0).unsqueeze(1)

        target_qf[:, 7:9] = torch.where(
            gripper_action_mask,
            sim_config["robot"]["gripper_torque"],
            -sim_config["robot"]["gripper_torque"],
        )

        target_qpos[:, :7] = self.step_ctrl(state_dict)["joint_positions"]
        target_qpos[:, 7:9] = torch.where(
            gripper_action_mask,
            self.max_gripper_width / 2,
            torch.zeros_like(target_qpos[:, 7:9]),
        )  # No use since it is qf control
        # print(target_qpos.cpu().numpy())

        # Write changes to buffer
        self.physx_system.cuda_articulation_qf.torch()[:, :] = target_qf
        self.physx_system.cuda_articulation_target_qpos.torch()[:, :] = target_qpos
        self.physx_system.cuda_articulation_target_qvel.torch()[:, :] = torch.zeros(
            (self.num_envs, self.franka_num_dof),
            dtype=torch.float32,
            device=self.device,
        )

    def update_render(self):
        self.step_viewer()
        self.step_sensor()

    def step_sensor(self):
        if self.render_system_group is None:
            return
        elif isinstance(self.render_system_group, sapien.render.RenderSystem):
            self.render_system_group.step()
            for sensors in self.sensors.values():
                sensors[0].take_picture()
            return
        self.render_system_group.update_render()

    def _reward(self) -> torch.Tensor:
        """Reward is 1 if two parts are newly assembled."""
        rewards = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        if not self.enable_reward:
            return rewards

        parts_poses = self.get_parts_poses(sim_coord=True)

        # Reshape parts_poses to (num_envs, num_parts, 7)
        num_parts = parts_poses.shape[1] // 7
        parts_poses = parts_poses.view(self.num_envs, num_parts, 7)

        # Compute the rewards based on the newly assembled parts
        newly_assembled_mask = torch.zeros(
            (self.num_envs, len(self.pairs_to_assemble)),
            dtype=torch.bool,
            device=self.device,
        )
        # Loop over parts to be assembled (relatively small number)
        for i, pair in enumerate(self.pairs_to_assemble):
            # Compute the relative pose for the specific pair of parts that should be assembled
            pose_mat1 = C.pose_from_vector(parts_poses[:, pair[0]])
            pose_mat2 = C.pose_from_vector(parts_poses[:, pair[1]])
            rel_pose = torch.matmul(torch.inverse(pose_mat1), pose_mat2)

            # Leading dimension is for checking if rel pose matches on of many possible assembled poses
            if pair in self.furniture.position_only:
                similar_rot = torch.tensor([True] * self.num_envs, device=self.device)
            else:
                similar_rot = C.is_similar_rot(
                    rel_pose[..., :3, :3],
                    self.assembled_rel_poses[i, :, None, :3, :3],
                    self.furniture.ori_bound,
                )
            similar_pos = C.is_similar_pos(
                rel_pose[..., :3, 3],
                self.assembled_rel_poses[i, :, None, :3, 3],
                torch.tensor(
                    self.furniture.assembled_pos_threshold, device=self.device
                ),
            )
            assembled_mask = similar_rot & similar_pos

            # Check if the parts are newly assembled (.any() over the multiple possibly matched assembled posees)
            newly_assembled_mask[:, i] = (
                assembled_mask.any(dim=0) & ~self.already_assembled[:, i]
            )

            # Update the already_assembled tensor
            self.already_assembled[:, i] |= newly_assembled_mask[:, i]

        # Compute the rewards based on the newly assembled parts
        rewards = newly_assembled_mask.any(dim=1).float().unsqueeze(-1)

        # print(f"Already assembled: {self.already_assembled.sum(dim=1)}")
        # print(
        #     f"Done envs: {torch.where(self.already_assembled.sum(dim=1) == len(self.pairs_to_assemble))[0]}"
        # )

        if self.manual_done and (rewards == 1).any():
            return print("Part assembled!")

        return rewards

    def _done(self):
        if self.manual_done:
            return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        return (
            self.already_assembled.sum(dim=1) == len(self.pairs_to_assemble)
        ).unsqueeze(1)

    def _random_perturbation_of_parts(
        self,
        max_force_magnitude: float,
        max_torque_magnitude: float,
    ):
        num_parts = len(self.part_entities)
        total_parts = len(self.part_entities) * self.num_envs

        # Generate a random mask to select parts with a 1% probability
        selected_part_mask = torch.rand(total_parts, device=self.device) < self.perturbation_prob

        # Generate random forces in the xy plane for the selected parts
        force_theta = (
            torch.rand((self.num_envs, num_parts, 1), device=self.device) * 2 * np.pi
        )
        force_magnitude = (
            torch.rand((self.num_envs, num_parts, 1), device=self.device)
            * max_force_magnitude
        )
        forces = torch.cat(
            [
                force_magnitude * torch.cos(force_theta),
                force_magnitude * torch.sin(force_theta),
                torch.zeros_like(force_magnitude),
            ],
            dim=-1,
        )

        # Scale the forces by the mass of the parts
        forces = (forces * self.force_multiplier).view(-1, 3)

        # Random torques
        # Generate random torques for the selected parts in the z direction
        z_torques = max_torque_magnitude * (
            torch.rand((self.num_envs, num_parts, 1), device=self.device) * 2 - 1
        )

        # Apply the torque multiplier
        z_torques = (z_torques * self.torque_multiplier).view(-1, 1)

        torques = torch.cat(
            [
                torch.zeros_like(z_torques),
                torch.zeros_like(z_torques),
                z_torques,
            ],
            dim=-1,
        )

        # Fill the appropriate indices with the generated forces and torques based on the selected part mask
        # Create tensors to hold forces and torques for all rigid bodies
        rigid_body_count = self.physx_system.cuda_rigid_body_data.torch().shape[0]

        all_forces = torch.zeros((rigid_body_count, 3), device=self.device)
        all_torques = torch.zeros((rigid_body_count, 3), device=self.device)

        # Fill the appropriate indices with the generated forces and torques based on the selected part mask
        all_forces[self.parts_gpu_index_tensor.view(-1)[selected_part_mask]] = forces[
            selected_part_mask
        ]
        all_torques[self.parts_gpu_index_tensor.view(-1)[selected_part_mask]] = torques[
            selected_part_mask
        ]

        # Apply the forces and torques to the rigid bodies
        self.physx_system.cuda_rigid_body_force.torch()[:, :3] = all_forces
        self.physx_system.cuda_rigid_body_torque.torch()[:, :3] = all_torques

        self.physx_system.gpu_apply_rigid_dynamic_force()
        self.physx_system.gpu_apply_rigid_dynamic_torque()

    def gripper_width(self) -> torch.Tensor:
        return (
            self.physx_system.cuda_articulation_qpos.torch()[:, 7:8]
            - self.physx_system.cuda_articulation_qpos.torch()[:, 8:9]
        )

    def sim_pose_to_april_pose(self, parts_poses: torch.Tensor) -> torch.Tensor:
        part_poses_mat = C.pose2mat_batched(
            parts_poses[:, :, :3], parts_poses[:, :, 3:7], device=self.device
        )

        april_coord_poses_mat = self.sim_coord_to_april_coord(part_poses_mat)
        april_coord_poses = torch.cat(C.mat2pose_batched(april_coord_poses_mat), dim=-1)
        return april_coord_poses

    def sim_pose_to_robot_pose(self, parts_poses: torch.Tensor) -> torch.Tensor:
        part_poses_mat = C.pose2mat_batched(
            parts_poses[:, :, :3], parts_poses[:, :, 3:7], device=self.device
        )

        robot_coord_poses_mat = self.sim_coord_to_robot_coord(part_poses_mat)
        robot_coord_poses = torch.cat(C.mat2pose_batched(robot_coord_poses_mat), dim=-1)
        return robot_coord_poses

    def sim_coord_to_robot_coord(self, sim_coord_mat: torch.Tensor) -> torch.Tensor:
        return self.sim_to_robot_mat @ sim_coord_mat

    def sim_coord_to_april_coord(self, sim_coord_mat: torch.Tensor) -> torch.Tensor:
        return self.sim_to_april_mat @ sim_coord_mat

    def april_coord_to_sim_coord(
        self, april_coord_mat: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return self.april_to_sim_mat @ april_coord_mat

    def filter_and_concat_robot_state(self, robot_state: Dict[str, torch.Tensor]):
        current_robot_state = []
        for rs in ROBOT_STATES:
            if rs not in robot_state:
                continue

            # if rs == "gripper_width":
            #     robot_state[rs] = robot_state[rs].reshape(-1, 1)
            current_robot_state.append(robot_state[rs])
        return torch.cat(current_robot_state, dim=-1)

    @property
    def envs(self) -> List[sapien.Scene]:
        return self.scenes

    @property
    def april_to_sim_mat(self) -> NDArray[np.float32]:
        return self.franka_from_origin_mat @ self.base_tag_from_robot_mat

    @property
    def sim_to_robot_mat(self) -> torch.Tensor:
        return torch.tensor(self.franka_from_origin_mat, device=self.device)

    @property
    def sim_to_april_mat(self) -> torch.Tensor:
        return torch.tensor(
            np.linalg.inv(self.base_tag_from_robot_mat)
            @ np.linalg.inv(self.franka_from_origin_mat),
            device=self.device,
        )

    @property
    def action_space(self):
        # Action space to be -1.0 to 1.0.
        if self.__act_rot_repr == 0:
            pose_dim = 7
        elif self.__act_rot_repr == 1:
            pose_dim = 9
        else:  # axis
            pose_dim = 6

        low = np.array([-1] * pose_dim + [-1], dtype=np.float32)
        high = np.array([1] * pose_dim + [1], dtype=np.float32)

        low = np.tile(low, (self.num_envs, 1))
        high = np.tile(high, (self.num_envs, 1))

        return gym.spaces.Box(low, high, (self.num_envs, pose_dim + 1))

    @property
    def action_type(self) -> Literal["delta", "pos"]:
        if self.__action_type:  # 1
            return "delta"
        return "pos"

    @property
    def rb_states(self) -> torch.Tensor:
        rb_states = self.physx_system.cuda_rigid_body_data.torch().clone()
        rb_states[..., 3:7] = rb_states[..., 3:7].roll(-1, dims=-1)
        return rb_states

    @property
    def april_to_robot_mat(self):
        return torch.tensor(self.base_tag_from_robot_mat, device=self.device)

    @property
    def n_parts_assemble(self):
        return len(self.furniture.should_be_assembled)

    @property
    def observation_space(self):
        low, high = -np.inf, np.inf
        dof = self.franka_num_dof
        img_size = config["furniture"]["env_img_size"]
        observation_space = {}
        full_robot_state_space = {
            "ee_pos": gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    3,
                ),
            ),  # (x, y, z)
            "ee_quat": gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    4,
                ),
            ),  #  (x, y, z, w)
            "ee_pos_vel": gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    3,
                ),
            ),
            "ee_ori_vel": gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    3,
                ),
            ),
            "joint_positions": gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    dof,
                ),
            ),
            "joint_velocities": gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    dof,
                ),
            ),
            "joint_torques": gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    dof,
                ),
            ),
            "gripper_width": gym.spaces.Box(low=low, high=high, shape=(self.num_envs,)),
            "gripper_finger_1_pos": gym.spaces.Box(
                low=low, high=high, shape=(self.num_envs,)
            ),
            "gripper_finger_2_pos": gym.spaces.Box(
                low=low, high=high, shape=(self.num_envs,)
            ),
        }
        robot_state_space = gym.spaces.Dict(
            {
                key: value
                for key, value in full_robot_state_space.items()
                if key in self.robot_state_keys
            }
        )
        observation_space["robot_state"] = robot_state_space
        if self.render_system_group is not None:
            for key in self.obs_keys:
                if key.startswith("color"):
                    observation_space[key] = gym.spaces.Box(
                        low=0, high=255, shape=(self.num_envs, *img_size, 3)
                    )
                if key.startswith("depth"):
                    observation_space[key] = gym.spaces.Box(
                        low=low, high=high, shape=(self.num_envs, *img_size)
                    )
        if self.include_parts_poses:
            observation_space["parts_poses"] = gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    self.num_envs,
                    self.furniture.num_parts * self.pose_dim,
                ),
            )
        return gym.spaces.Dict(observation_space)


class FurnitureSimFullEnv(FurnitureSimEnv):
    """FurnitureSim environment with all available observations."""

    def __init__(self, **kwargs):
        super().__init__(obs_keys=FULL_OBS, **kwargs)


class FurnitureSimStateEnv(FurnitureSimEnv):
    """FurnitureSim environment with state observations."""

    def __init__(self, **kwargs):
        obs_keys = DEFAULT_STATE_OBS
        super().__init__(obs_keys=obs_keys, concat_robot_state=True, **kwargs)
