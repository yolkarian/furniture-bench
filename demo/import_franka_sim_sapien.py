import sapien.core
import sapien
import sapien.physx
import sapien.render
from furniture_bench.utils.sapien.urdf_loader import URDFLoader
import sapien.utils.viewer.control_window
from furniture_bench.utils.sapien import (
    load_scene_config,
    generate_builder_with_options,
)
from sapien.utils.viewer import Viewer
import os
import numpy as np
import scipy
import scipy.spatial

from collections import defaultdict
from typing import Dict, Union
from datetime import datetime
from pathlib import Path

from furniture_bench.furniture.furniture import Furniture
from furniture_bench.utils.recorder import VideoRecorder
import torch
import gym
import numpy as np

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
)
from furniture_bench.robot.robot_state import ROBOT_STATE_DIMS, ROBOT_STATES
from furniture_bench.furniture.parts.part import Part

from ipdb import set_trace as bp
from typing import Union, Tuple, List

from furniture_bench.sim_config_sapien import sim_config, AssetOptions
from furniture_bench.utils.sapien.actor_builder import ActorBuilder
from furniture_bench.utils.sapien.articulation_builder import ArticulationBuilder


ASSET_ROOT = str(Path(__file__).parent.parent.absolute() /"furniture_bench"/ "assets_no_tags")

class FurnitureSimImport:
    """FurnitureSim base class."""

    def __init__(
        self,
        furniture: str,
        num_envs: int = 1,
        init_assembled: bool = False,
        np_step_out: bool = False,
        randomness: Union[str, Randomness] = "low",
        high_random_idx: int = 0,
        max_env_steps: int = 3000,
        headless: bool = False,
        ee_laser: bool = False,
        parts_poses_in_robot_frame=False,
    ):
        """
        Args:
            furniture (str): Specifies the type of furniture. Options are 'lamp', 'square_table', 'desk', 'drawer', 'cabinet', 'round_table', 'stool', 'chair', 'one_leg'.
            num_envs (int): Number of parallel environments.
            resize_img (bool): If true, images are resized to 224 x 224.
            obs_keys (list): List of observations for observation space (i.e., RGB-D image from three cameras, proprioceptive states, and poses of the furniture parts.)
            concat_robot_state (bool): Whether to return concatenated `robot_state` or its dictionary form in observation.
            manual_label (bool): If true, the environment reward is manually labeled.
            manual_done (bool): If true, the environment is terminated manually.
            headless (bool): If true, simulation runs without GUI.
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            init_assembled (bool): If true, the environment is initialized with assembled furniture.
            np_step_out (bool): If true, env.step() returns Numpy arrays.
            channel_first (bool): If true, color images are returned in channel first format [3, H, w].
            randomness (str): Level of randomness in the environment. Options are 'low', 'med', 'high'.
            high_random_idx (int): Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input (bool): If true, the initial camera inputs are saved.
            record (bool): If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps (int): Maximum number of steps per episode (default: 3000).
            act_rot_repr (str): Representation of rotation for action space. Options are 'quat', 'axis', or 'rot_6d'.
            ctrl_mode (str): 'diffik' (joint impedance, with differential inverse kinematics control)
        """
        super(FurnitureSimEnv, self).__init__()
        self.device = torch.device("cuda")

        self.assemble_idx = 0
        # Furniture for each environment (reward, reset).
        self.furnitures = [furniture_factory(furniture) for _ in range(num_envs)]

        if num_envs == 1:
            self.furniture = self.furnitures[0]
        else:
            self.furniture = furniture_factory(furniture)

        self.max_env_steps = max_env_steps
        self.furniture.max_env_steps = max_env_steps
        for furn in self.furnitures:
            furn.max_env_steps = max_env_steps

        self.furniture_name = furniture
        self.task_name = furniture
        self.num_envs = num_envs
        self.pose_dim = 7
        self.headless = headless


        self.move_neutral = False
        self.ctrl_started = False
        self.init_assembled = init_assembled
        self.np_step_out = np_step_out
        self.from_skill = (
            0  # TODO: Skill benchmark should be implemented in FurnitureSim.
        )
        self.randomness = str_to_enum(randomness)
        self.high_random_idx = high_random_idx
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)
        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open and close actions.
        self.max_gripper_width = config["robot"]["max_gripper_width"][furniture]


        print(f"Observation keys: {self.obs_keys}")

        if "factory" in self.furniture_name:
            # Adjust simulation parameters
            sim_config["sim_params"].dt = 1.0 / 120.0
            sim_config["sim_params"].substeps = 4
            sim_config["sim_params"].physx.max_gpu_contact_pairs = 6553600
            sim_config["sim_params"].physx.default_buffer_size_multiplier = 8.0

            # Adjust part friction
            sim_config["parts"]["friction"] = 0.25

        # Simulator setup.
        # NOTE(Yuke): as for sapien, the scene should not be load here
        # self.sapien_scene:sapien.Scene = sapien.Scene()
        # load_scene_config(self.sapien_scene, sim_config["sim_params"])
        # self.assets_loader = self.sapien_scene.create_urdf_loader() # load static root
        # self.assets_loader.fix_root_link = True
        if sim_config["sim_params"].use_gpu_pipeline:
            sapien.physx.enable_gpu() # According tp the comment in the C++ code
        self.urdf_builder_generator = URDFLoader()
        self.urdf_builder_generator.fix_root_link = True

        # our flags
        self.parts_poses_in_robot_frame = parts_poses_in_robot_frame

        self.import_assets()
        self._create_ground_plane()
        self.create_subscenes()

        self._setup_lights()
        self.set_viewer()
        # self.set_camera()
    def set_viewer(self):
        """Create the viewer."""
        

        if not self.headless:
            self.viewer = Viewer()
            self.viewer.set_scene(self.envs[0])

            control_window = self.viewer.control_window
            control_window.show_joint_axes = False
            control_window.show_camera_linesets = False
            self.viewer.set_camera_xyz(x=0.97, y=0, z=0.74)
            self.viewer.set_camera_rpy(r=0, p=-0.3, y=0)
            
            # Point camera at middle env.
            pose = sapien.Pose(p=[0.97,0,0.74])

            vec = np.array([-1, 0, 0.62],dtype=np.float32) - pose.get_p()
            vec = vec / np.linalg.norm(vec)
            rot,_ = scipy.spatial.transform.Rotation.align_vectors(vec, [1, 0, 0])
            pose.set_rpy(rot.as_euler("xyz").astype(np.float32))
            self.viewer.set_camera_pose(pose)
            # Other API also works, it redefined in control window
            # cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            # cam_target = gymapi.Vec3(-1, 0, 0.62)
            # middle_env = self.envs[0]
            # self.isaac_gym.viewer_camera_look_at(
            #     self.viewer, middle_env, cam_pos, cam_target
            # )


    def import_assets(self):
        self.base_tag_asset: ActorBuilder = self._import_base_tag_asset()
        self.background_asset: ActorBuilder = self._import_background_asset()
        self.table_asset: ActorBuilder = self._import_table_asset()
        self.obstacle_front_asset: ActorBuilder = self._import_obstacle_front_asset()
        self.obstacle_side_asset: ActorBuilder = self._import_obstacle_side_asset()
        self.franka_asset: ArticulationBuilder = self._import_franka_asset()
        self.part_assets: Dict[str, ActorBuilder] = self._import_part_assets()

    def _create_ground_plane(
        self,
        enable_render: bool = True,
        render_material: Union[
            sapien.render.RenderMaterial, None, Tuple[float, float, float]
        ] = None,
    ):
        """Creates ground plane."""
        self._ground_builder = sapien.ActorBuilder()
        self._ground_builder.set_physx_body_type("static")
        self._ground_builder.add_plane_collision(
            sapien.Pose(p=[0, 0, 0]), q=[0.7071068, 0, -0.7071068, 0]
        )
        self._ground_builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        if enable_render:
            self._ground_builder.add_plane_visual(
                sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]),
                [10, 10, 10],
                render_material,
                "ground_visual",
            )
        self._ground_builder.set_name("ground")

    def _setup_lights(self):
        for light in sim_config["lights"]:
            self.sapien_scene.add_directional_light(light["direction"], light["color"])
            self.sapien_scene.set_ambient_light(light["ambient"])

    @property
    def n_parts_assemble(self):
        return len(self.furniture.should_be_assembled)

    def create_subscenes(self):
        table_pos = np.array([0.8, 0.8, 0.4], dtype=np.float32)

        table_half_width = 0.015
        table_surface_z = table_pos[2] + table_half_width
        self.franka_pose = sapien.Pose(
            p=[0.5 * -table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT],
        )

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p[0], self.franka_pose.p[1], self.franka_pose.p[2]],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]

        # This is to get the actor in the following code
        # TODO: modify the code above
        # franka_link_dict = self.isaac_gym.get_asset_rigid_body_dict(self.franka_asset)
        # self.franka_ee_index = franka_link_dict["k_ee_link"]
        # self.franka_base_index = franka_link_dict["panda_link0"]

        # Create scenes.
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = np.array([-spacing, -spacing, 0.0], dtype=np.float32)
        env_upper = np.array([spacing, spacing, spacing], dtype=np.float32)
        self.envs:List[sapien.Scene] = []
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.entities = {}
        self.ee_idxs = []
        self.ee_entities = []

        self.base_tag_pose = sapien.Pose()
        base_tag_pos = T.pos_from_mat(config["robot"]["tag_base_from_robot_base"])
        self.base_tag_pose.p = self.franka_pose.p + np.array(
            [base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT], dtype=np.float32
        )
        self.base_tag_pose.p[2] = table_surface_z

        self.obstacle_front_pose = sapien.Pose()
        self.obstacle_front_pose.p = np.array(
            [self.base_tag_pose.p[0] + 0.37 + 0.01, 0.0, table_surface_z + 0.015],
            dtype=np.float32,
        )
        self.obstacle_front_pose.q = (
            scipy.spatial.transform.Rotation.from_rotvec(
                np.array([0, 0, 1]) * 0.5 * np.pi
            )
            .as_quat()
            .astype(np.float32)
        )

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

        self.obstacle_left_pose = sapien.Pose
        self.obstacle_left_pose.p = np.array(
            [
                self.obstacle_front_pose.p[0] - 0.075,
                self.obstacle_front_pose.p[1] + 0.175,
                self.obstacle_front_pose.p[2],
            ],
            dtype=np.float32,
        )
        self.obstacle_left_pose.q = self.obstacle_front_pose.q

        self.base_idxs = []
        self.part_idxs = defaultdict(list)
        self.franka_entities = []
        self.franka_actor_idx_all = []
        self.part_actor_idx_all = []  # global list of indices, when resetting all parts
        self.part_actor_idx_by_env = {}
        self.obstacle_actor_idxs_by_env = {}
        self.bulb_actor_idxs_by_env = {}
                    
        self.parts_entities = {} # entities of part in env 0
        self.obstacle_entities = [] # entities of obstacles in env 0
        self._subscenes = []
        scene_grid_length = int(np.ceil(np.sqrt(self.num_envs)))
        self.physx_system = sapien.physx.PhysxGpuSystem(
            device=sapien.Device(self.device.type)
        )  # cuda for the rendering
        self.obstacle_handles = []
        for i in range(self.num_envs):
            scene_x, scene_y = (
                i % scene_grid_length - scene_grid_length // 2,
                i // scene_grid_length - scene_grid_length // 2,
            )
            scene = sapien.Scene(
                systems=[
                    self.physx_system,
                    sapien.render.RenderSystem(self.device.type),
                ]  # cuda for the rendering
            )
            load_scene_config(scene, sim_config["sim_params"])
            self.physx_system.set_scene_offset(
                scene,
                [
                    scene_x * spacing,
                    scene_y * spacing,
                    0,
                ],
            )
            self._subscenes.append(scene)

            # Add workspace (table).
            table_pose = sapien.Pose()
            table_pose.p = np.array([0.0, 0.0, table_pos[2]], dtype=np.float32)

            self.table_asset.set_scene(scene)
            table_entity = self.table_asset.build(f"table_{i}")

            # table Compenent Set
            table_props = table_entity.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            table_props: sapien.physx.PhysxRigidDynamicComponent
            for collision_shape in table_props.collision_shapes:
                mt = collision_shape.get_physical_material()
                mt.set_static_friction(
                    1.01 * sim_config["table"]["friction"]
                )  # static friction is slightly larger that dynamic friction
                mt.set_dynamic_friction(sim_config["table"]["friction"])
                collision_shape.set_physical_material(mt)
            # TODO(Yuke): check whether actually the friction changes or not.

            # Base Tag
            self.base_tag_asset.set_scene(scene)
            bg_pose = sapien.Pose()
            bg_pose.set_p([-0.8, 0, 0.75])
            self.base_tag_asset.set_initial_pose(bg_pose)
            base_tag_entity = self.base_tag_asset.build(f"base_tag_{i}")

            self.background_asset.set_scene(scene)
            bg_entity = self.background_asset.build(f"background_{i}")

            # Make the obstacle
            # TODO: Make config
            self.obstacle_actor_idxs_by_env[i] = []
            for asset, pose, name in zip(
                [
                    self.obstacle_front_asset,
                    self.obstacle_side_asset,
                    self.obstacle_side_asset,
                ],
                [
                    self.obstacle_front_pose,
                    self.obstacle_right_pose,
                    self.obstacle_left_pose,
                ],
                [f"obstacle_front_{i}", f"obstacle_right_{i}", f"obstacle_left_{i}"],
            ):
                asset.set_scene(scene)
                asset.set_initial_pose(pose)
                obstacle_entity = asset.build(name)
                if i == 0:
                    self.obstacle_entities.append(obstacle_entity)

                # NOTE(Yuke): different ID might be needed
                self.obstacle_actor_idxs_by_env[i].append(obstacle_entity.get_global_id())
                self.part_idxs[name].append(obstacle_entity.get_global_id())

            # Add robot.
            self.franka_asset.set_scene(scene)
            franka_entity = self.franka_asset.build()
            franka_entity.set_name(f"franka_{i}")
            # TODO: whether it is actually the index we want
            self.franka_actor_idx_all(franka_entity.get_gpu_index())
            self.franka_num_dofs = franka_entity.get_dof()

            # TODO(Yuke): force sensor
            # self.isaac_gym.enable_actor_dof_force_sensors(env, franka_handle)
            self.franka_entities.append(franka_entity)

            # TODO(Yuke): To inspect whether the index here is global index or not. Get global index of hand and base.
            self.ee_idxs.append(franka_entity.find_link_by_name("panda_link0"))
            self.ee_entities.append(franka_entity.find_link_by_name("panda_link0"))
            self.base_idxs.append(
                franka_entity.find_link_by_name("k_ee_link").get_index()
            )
            # Set dof properties (active joints)
            for i, joint in enumerate(franka_entity.active_joints):
                if i < 7:
                    # NOTE(Yuke): use the internal PD controller to control
                    joint.set_drive_properties(
                        stiffness=self.stiffness, damping=self.damping, mode="force"
                    )
                    joint.set_friction(sim_config["robot"]["arm_frictions"])
                    continue
                joint.set_drive_properties(
                        stiffness=0, damping=0, mode="force"
                    )
                joint.set_friction(sim_config["robot"]["gripper_frictions"])
                joint.set_limits([0, self.max_gripper_width / 2])

            # Set initial dof states

            self.default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
            self.default_dof_pos[:7] = np.array(
                config["robot"]["reset_joints"], dtype=np.float32
            )
            self.default_dof_pos[7:] = self.max_gripper_width / 2
            franka_entity.set_qpos(self.default_dof_pos[:,])

            # Add furniture parts.
            poses = []
            self.part_actor_idx_by_env[i] = []
            self.bulb_actor_idxs_by_env[i] = []

            for part in self.furniture.parts:
                pos, ori = self._get_reset_pose(part)
                part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
                part_pose = sapien.Pose()
                part_pose.set_p(
                    [part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]]
                )
                reset_ori = self.april_coord_to_sim_coord(ori)
                part_pose.set_q(np.array(T.mat2quat(reset_ori[:3, :3]),dtype=np.float32))
                poses.append(part_pose)

                self.part_assets[part.name].set_scene(scene)
                part_entitiy = self.part_assets[part.name].build(part.name)
                component:sapien.physx.PhysxRigidBaseComponent = part_entitiy.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
                for collision_shape in component.collision_shapes:
                    # upper 16 must be the same to have collision
                    # TODO: need testing
                    groups = collision_shape.get_collision_groups()
                    collision_shape.get_physical_material()
                    groups[3] = i << 16 
                    collision_shape.set_collision_groups(groups)
                    mt = collision_shape.get_physical_material()
                    mt.set_static_friction(1.01*sim_config["parts"]["friction"])

                    mt.set_dynamic_friction(sim_config["parts"]["friction"])
                    collision_shape.set_physical_material(mt)

                

                part_idx = part_entitiy.get_global_id()
                if i == 0:
                    self.entities[part.name] = part_entitiy
                if part.name == "lamp_bulb":
                    self.bulb_actor_idxs_by_env[i].append(part_idx)
                self.part_actor_idx_all.append(part_idx)
                self.part_actor_idx_by_env[i].append(part_idx)
                self.part_idxs[part.name].append(part_idx)

        # Make a tensor that contains the RB indices of all the furniture parts.
        self.furniture_rb_indices = torch.stack(
            [torch.tensor(self.part_idxs[part.name]) for part in self.furniture.parts],
            dim=0,
        ).T

        if self.furniture_name == "lamp":
            self.lamp_bulb_rb_indices = torch.stack(
                [
                    torch.tensor(self.part_idxs[part.name])
                    for part in self.furniture.parts
                    if part.name == "lamp_bulb"
                ],
                dim=0,
            ).T


            self.hand_bulb_pos_thresh = torch.tensor(
                [0.03, 0.03, 0.03], dtype=torch.float32, device=self.device
            )
        # Make a tensor that contains the RB indices of all the furniture parts.
        # Add a dimension for the part number to be compatible with the parts RB indices.
        self.obstacle_front_rb_indices = torch.tensor(
            self.part_idxs["obstacle_front"]
        ).unsqueeze(1)

        # This only needs to happen once
        # This obstacle entities and part entities of env 0 have been added before
        # print(f'Getting the separate actor indices for the frankas and the furniture parts (not the handles)')

        self.franka_actor_idxs_all_t = torch.tensor(
            self.franka_actor_idx_all, device=self.device, dtype=torch.int32
        )
        self.part_actor_idxs_all_t = torch.tensor(
            self.part_actor_idx_all, device=self.device, dtype=torch.int32
        )
    def _import_base_tag_asset(self):
        asset_file = "furniture/urdf/base_tag.urdf"
        asset_options = AssetOptions()
        asset_options.fix_base_link = True
        return generate_builder_with_options(
            self.urdf_builder_generator, ASSET_ROOT, asset_file, asset_options
        )

    def _import_part_assets(self):
        part_builders = {}
        for part in self.furniture.parts:
            asset_option = sim_config["asset"][part.name]
            part_builders[part.name] = generate_builder_with_options(
                self.urdf_builder_generator, ASSET_ROOT, part.asset_file, asset_option
            )
        # TODO: Setup params for these parts
        return part_builders

    def _import_obstacle_asset(self):
        asset_file = "furniture/urdf/obstacle.urdf"
        asset_options = AssetOptions()
        asset_options.fix_base_link = True
        return generate_builder_with_options(
            self.urdf_builder_generator, ASSET_ROOT, asset_file, asset_options
        )

    def _import_background_asset(self):
        asset_file = "furniture/urdf/background.urdf"
        asset_options = AssetOptions()
        asset_options.fix_base_link = True
        return generate_builder_with_options(
            self.urdf_builder_generator, ASSET_ROOT, asset_file, asset_options
        )

    def _import_table_asset(self):
        asset_file = "furniture/urdf/table.urdf"
        asset_options = AssetOptions()
        asset_options.fix_base_link = True
        return generate_builder_with_options(
            self.urdf_builder_generator, ASSET_ROOT, asset_file, asset_options
        )

    def _import_obstacle_front_asset(self):
        asset_file = "furniture/urdf/obstacle_front.urdf"
        asset_options = AssetOptions()
        asset_options.fix_base_link = True
        return generate_builder_with_options(
            self.urdf_builder_generator, ASSET_ROOT, asset_file, asset_options
        )

    def _import_obstacle_side_asset(self):
        asset_file = "furniture/urdf/obstacle_side.urdf"
        asset_options = AssetOptions()
        asset_options.fix_base_link = True
        return generate_builder_with_options(
            self.urdf_builder_generator, ASSET_ROOT, asset_file, asset_options
        )

    def _import_franka_asset(self):
        self.franka_asset_file = (
            "franka_description_ros/franka_description/robots/franka_panda.urdf"
        )
        asset_options = AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_file = (
            "franka_description_ros/franka_description/robots/franka_panda.urdf"
        )
        asset_options = AssetOptions()
        return generate_builder_with_options(
            self.urdf_builder_generator, ASSET_ROOT, asset_file
        )
