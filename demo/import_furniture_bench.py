import sapien.core
import sapien
import sapien.physx
import sapien.render
from furniture_bench.utils.sapien.urdf_loader import URDFLoader
import sapien.utils.viewer.control_window
from furniture_bench.utils.sapien import (
    load_scene_config,
    generate_builder_with_options,
    generate_builder_with_options_,
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
from typing import Union, Tuple, List, Optional

from furniture_bench.sim_config_sapien import (
    sim_config,
    AssetOptions,
    SimParams,
    PhysxParams,
)
from furniture_bench.utils.sapien.actor_builder import ActorBuilder
from furniture_bench.utils.sapien.articulation_builder import ArticulationBuilder

ASSET_ROOT = str(Path(__file__).parent.parent / "furniture_bench" / "assets_no_tags")
# ASSET_ROOT = str(Path(__file__).parent.parent / "furniture_bench" / "assets")


# TODO:
#      1. Action Input and output
#      2. Shader

# NOTE(Yuke): 
# 1. Currently no such shared collision shapes which can collide with elements in all Scenes.
# 2. Some actors do not contain collision mesh.

class FurnitureEnv:
    def __init__(
        self,
        furniture_name: str,
        num_envs: int = 1,
        headless: bool = False,
        parallel_in_single_scene: bool = False,
    ):
        self.furniture_name = furniture_name
        self.num_envs = num_envs
        
        self.headless = headless
        self.parallel_in_single_scene = parallel_in_single_scene
        # assert self.parallel_in_single_scene

        # Predefined parameters
        self.device = torch.device("cuda")
        self.sapien_device = sapien.Device(self.device.type)
        self.pose_dim = 7
        self.stiffness = 1000.00
        self.damping = 200.0
        self.restitution = 0.5
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)
        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open and close actions.
        self.max_gripper_width = config["robot"]["max_gripper_width"][
            self.furniture_name
        ]
        self.furnitures = [furniture_factory(self.furniture_name) for _ in range(num_envs)]
        if self.num_envs == 1:
            self.furniture = self.furnitures[0]
        else:
            self.furniture = furniture_factory(self.furniture_name)

        self.from_skill = 0 # TODO: to investigate the role of this parameter
        self.scene_spacing = 3.0
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
            "obstacle_left": "furniture/urdf/obstacle_side.urdf",
            "obstacle_right": "furniture/urdf/obstacle_side.urdf",
            "background": "furniture/urdf/background.urdf",
            "obstacle_front": "furniture/urdf/obstacle_front.urdf",
            "table": "furniture/urdf/table.urdf",
        }
        self.table_pos = np.array([0.8, 0.8, 0.4], dtype=np.float32)

        table_half_width = 0.015
        table_surface_z:float = self.table_pos[2] + table_half_width
        self.franka_pose = sapien.Pose(
            p=[0.5 * -self.table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT],
        )

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p[0], self.franka_pose.p[1], self.franka_pose.p[2]],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]


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

        #%% General Setup
        sapien.physx.enable_gpu()
        self.physx_system = sapien.physx.PhysxGpuSystem(self.sapien_device)
        self.urdf_loader = URDFLoader()  # just used to generate builder
        self.render_system_group:Optional[sapien.render.RenderSystemGroup] = None

        # %% Create builder

        self._create_static_obj_builders()
        self._create_ground_builder()
        self._create_franka_builder()
        self._create_part_builders()


        #%% Create Scenes
        self.scenes: List[sapien.Scene] = []

        # load Scene configs
        sim_params: SimParams = sim_config["sim_params"]
        self.physx_system.set_timestep(sim_params.dt)
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

        # Some info to store
        self.franka_entities:List[sapien.physx.PhysxArticulation] = []
        self.static_obj_entites:Dict[str, List[sapien.Entity]] = {}
        self.part_entities:Dict[str, List[sapien.Entity]] = {}
        for key in self.static_obj_dict.keys():
            self.static_obj_entites[key] = []
        for part in self.furniture.parts:
            self.part_entities[part.name] = []
        self.part_actors:Dict[str, List[sapien.Entity]] = {}
        self.scene_offsets_np:List[np.ndarray] = []
        scene_grid_length = int(np.ceil(np.sqrt(self.num_envs)))

        if parallel_in_single_scene:
            self.scenes.append(sapien.Scene(
                    systems=[
                        self.physx_system,
                        sapien.render.RenderSystem(self.sapien_device),
                    ]  # cuda also for the rendering
            ))
        for i in range(self.num_envs):
            scene_x, scene_y = (
                i % scene_grid_length - scene_grid_length // 2,
                i // scene_grid_length - scene_grid_length // 2,
            )
            scene_offset = np.array(                [
                    scene_x * self.scene_spacing,
                    scene_y * self.scene_spacing,
                    0,
                ],dtype=np.float32)
            self.scene_offsets_np.append(scene_offset)
            if parallel_in_single_scene:
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
            self.ground_builder.collision_groups[3] &= 0xffff
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
                value.collision_groups[3] &= 0xffff
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
                value.collision_groups[3] &= 0xffff
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
                self.frank_builder.initial_pose.set_p(scene_offset + self.frank_builder.initial_pose.get_p())
            for link_builder in self.frank_builder.link_builders:
                link_builder.collision_groups[3] &= 0xffff
                link_builder.collision_groups[3] |= i << 16
                # links without one articulation haven't been changed in this case
            franka_entity = self.frank_builder.build() # Actually, this object is not Entity
            if self.parallel_in_single_scene:
                franka_entity.set_name(f"scene_{i}_franka")
                self.frank_builder.initial_pose.set_p(tmp_initial_p)
            else:
                franka_entity.set_name("franka")
            
            if i == 0:
                self.franka_num_dof = franka_entity.get_dof()
                self.franka_default_dof_pos = np.zeros(self.franka_num_dof, dtype=np.float32)
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
                    joint.set_drive_properties(stiffness=self.stiffness, damping=self.damping)
                else:
                    joint.set_drive_properties(stiffness=self.stiffness, damping=0)

            # For testing of the shape of Tensor
            # Automatically construct a tensor which contains q of all entities with the 
            # same configuration. (m, dof)
            # self.frank_builder.set_initial_pose(
            #     sapien.Pose(
            #         p = [2.0 + 0.5 * -self.table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT]
            #     )
            # )
            # franka_entity.set_name("franka_extra_test")
            # franka_entity = self.frank_builder.build()
            # self.franka_entities.append(franka_entity)

            self.scenes.append(scene)

        
        # TODO: add entities into the list

        self._add_light()
        self._set_viewer()

    def _create_static_obj_builders(self):
        # Create builders for all static objs (Actorbuilder/ArticulationBuilder)
        self.static_obj_builders: Dict[str, ActorBuilder] = {}
        for key, value in self.static_obj_dict.items():
            opts = AssetOptions()
            opts.fix_base_link = True
            self.static_obj_builders[key] = generate_builder_with_options_(
                self.urdf_loader, os.path.join(ASSET_ROOT, value), opts
            )
            self.static_obj_builders[key].set_name(name=key)

        # Setup Config for objs
        self.static_obj_builders["table"].set_initial_pose(
            sapien.Pose(p=[0, 0, self.table_pos[2]])
        )
        for collision_record in self.static_obj_builders["table"].collision_records:
            mt = sapien.physx.PhysxMaterial(sim_config["table"]["friction"], sim_config["table"]["friction"], self.restitution)
            collision_record.material = mt
        self.static_obj_builders["base_tag"].set_initial_pose(
            self.base_tag_pose
        )
        self.static_obj_builders["background"].set_initial_pose(
            sapien.Pose(p =[-0.8, 0, 0.75])
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
        opts.fix_base_link = True
        opts.disable_gravity = True
        opts.flip_visual_attachments = True
        self.frank_builder:ArticulationBuilder =  generate_builder_with_options_(
            self.urdf_loader,
            os.path.join(ASSET_ROOT, urdf_file),
            opts
            )
        for link_builder in self.frank_builder.link_builders:
            link_builder.joint_record.damping = self.damping
            # Different setting for the body and gripper
            if link_builder.name.endswith("finger"):
                link_builder.joint_record.limits = (0, self.max_gripper_width / 2)
                link_builder.joint_record.friction = sim_config["robot"]["gripper_frictions"]
            else:
                link_builder.joint_record.friction = sim_config["robot"]["arm_frictions"]
        self.frank_builder.set_initial_pose(self.franka_pose)

    def _create_part_builders(self):
        self.part_builders:Dict[str, ActorBuilder] = {}
        for part in self.furniture.parts:
            part_builder = generate_builder_with_options_(
                self.urdf_loader,
                os.path.join(ASSET_ROOT, part.asset_file),
                AssetOptions()
            )
            part_builder.set_name(part.name)
            self.part_builders[part.name] = part_builder
            pos, ori = self._get_reset_pose_part(part)
            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = sapien.Pose()
            part_pose.set_p(
                [part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]]
            )
            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.set_q(np.array(T.mat2quat(reset_ori[:3, :3]),dtype=np.float32))
            part_builder.set_initial_pose(part_pose)
            for collision_record in part_builder.collision_records:  
                collision_record.material = sapien.physx.PhysxMaterial(
                    sim_config["parts"]["friction"],
                    sim_config["parts"]["friction"],
                    self.restitution
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
                light_position = self.scene_offsets_np[0]
                direct_light.set_entity_pose(sapien.Pose(
                    light_position, sapien.math.shortest_rotation([1, 0, 0], light["direction"])
                ))
                
                scene.add_entity(entity)
                # NOTE(Yuke): for rendering in a single scenario
                if self.parallel_in_single_scene:
                    break


    def _set_viewer(self):
        # If parallel_in_one_scene is enabled, all articulations and actors will be loaded 
        # in one scene and rendered in only one render system
        # If it is disabled, only one scene containing one instance of simulation will be
        # display in the viewer
        self.viewer = Viewer()
        self.viewer.set_scene(self.scenes[0])
        control_window = self.viewer.control_window
        control_window.show_joint_axes = False
        control_window.show_camera_linesets = False
        
        pose = sapien.Pose(p=[0.97,0,0.74])

        vec = np.array([-1, 0, 0.62],dtype=np.float32) - pose.get_p()
        vec = vec / np.linalg.norm(vec)
        rot,_ = scipy.spatial.transform.Rotation.align_vectors(vec, [1, 0, 0])
        if vec[0] < 0:
            quat = np.zeros(4, dtype = np.float32)
            quat[:3] = vec
            rot =  rot * scipy.spatial.transform.Rotation.from_quat(quat) 
        pose.set_rpy(rot.as_euler("xyz").astype(np.float32))
        self.viewer.set_camera_pose(pose)

    def _get_reset_pose_part(self, part: Part):
        """Get the reset pose of the part.

        Args:
            part: The part to get the reset pose.
        """
        pos = part.reset_pos[self.from_skill]
        ori = part.reset_ori[self.from_skill]
        return pos, ori


    def init_sim(self):
        # torch seed can be added before
        self.physx_system.gpu_init() 

        # NOTE(Yuke): Setting the initial qpos must be complete after gpu_init
        #   and according to my understanding we should fetch first. Otherwise, the values stored
        #   in the buffer are undefined
        self.physx_system.gpu_fetch_rigid_dynamic_data()
        self.physx_system.gpu_fetch_articulation_link_pose()
        self.physx_system.cuda_rigid_body_data.torch()[:, 7:] = torch.zeros_like(self.physx_system.cuda_rigid_body_data.torch()[:, 7:])
        self.reset_franka_qpos()

        self.physx_system.gpu_apply_rigid_dynamic_data()
        self.physx_system.gpu_apply_articulation_root_pose()
        self.physx_system.gpu_apply_articulation_root_velocity()
        self.physx_system.gpu_apply_articulation_qpos()  
        self.physx_system.gpu_apply_articulation_qvel()
        if self.parallel_in_single_scene:
            pass
            # self.render_system_group = sapien.render.RenderSystemGroup([ s.render_system for s in self.scenes])
        
    def reset_franka_qpos(self):
        frank_entities_gpu_index = torch.tensor([frank_entity.get_gpu_index() for frank_entity in self.franka_entities],dtype=torch.int32, device=self.device)
        self.physx_system.cuda_articulation_qpos.torch()[frank_entities_gpu_index, :self.franka_num_dof] = torch.from_numpy(self.franka_default_dof_pos).to(self.device)
        self.physx_system.cuda_articulation_qvel.torch()[:, :] = torch.zeros_like(self.physx_system.cuda_articulation_qvel.torch())
    
    def april_coord_to_sim_coord(self, april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return self.april_to_sim_mat @ april_coord_mat
    @property
    def april_to_sim_mat(self):
        return self.franka_from_origin_mat @ self.base_tag_from_robot_mat



if __name__=="__main__":
    sim = FurnitureEnv(furniture_name="lamp", num_envs=4, parallel_in_single_scene=True)
    sim.init_sim()

    # TODO: Currently please only use lamp for the simulation, since to use other furnitures
    #   file path change in the urdf file should be made.
    action = sim.franka_default_dof_pos[None,:].repeat(sim.num_envs,axis = 0)

    while not sim.viewer.closed:
        sim.physx_system.step()
        sim.physx_system.sync_poses_gpu_to_cpu()
        sim.scenes[0].update_render()
        sim.viewer.render()
        # Random Policy
        noise = np.random.rand(*action.shape)  - 0.5
        noise[:, -2] = 0
        action_np = action + 0.02 * noise

        action_torch = torch.from_numpy(action_np).to(sim.device, dtype=torch.float32)
        
        sim.physx_system.cuda_articulation_target_qpos.torch()[:, :] = action_torch[:, :]
        # sim.physx_system.cuda_articulation_target_qvel.torch()[:, :] = torch.zeros_like(sim.physx_system.cuda_articulation_target_qvel.torch()).to(sim.device)
        sim.physx_system.gpu_apply_articulation_target_position()
        # sim.physx_system.gpu_apply_articulation_target_velocity()
