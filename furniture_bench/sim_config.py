"""Define additional simulator parameters for the SAPIEN backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from furniture_bench.config import config

sim_config = config.copy()


# Positional threshold for declaring the furniture assembled.
# This is a smaller value compared to the real-world config, since the detection can be more accurate in simulation.
# This will reduce the false positive rate.
sim_config["furniture"]["assembled_pos_threshold"] = [0.005, 0.005, 0.005]


# Timeout for # environment steps for each furniture model.
sim_config["scripted_timeout"] = {
    "one_leg": 700,  # Increased from 600
    "cabinet": 1500,
    "lamp": 1100,  # Increased from 1000
    "round_table": 1500,  # Increased from 1300
    "drawer": 1_000,  # Increased from 1300
    "stool": 1_000,  # Increased from 1300
}


def _next_power_of_two(value: int) -> int:
    """Round a positive integer up to the next power of two."""
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


@dataclass
class PhysxParams:
    """Low-level PhysX scene configuration."""

    solver_type: int = 1  # 0 PCS 1 TGS
    bounce_threshold_velocity: float = 0.002
    num_position_iterations: int = 10
    num_velocity_iterations: int = 1
    rest_offset: float = 0.0
    contact_offset: float = 0.009999999776482582
    friction_offset_threshold: float = 0.01  # These two
    friction_correlation_distance: float = 0.005
    max_depenetration_velocity: float = 10
    num_threads: int = 0
    use_gpu: bool = False
    max_gpu_contact_pairs: int = 16055314
    default_buffer_size_multiplier: float = 8.0

    # TODO: introduce params for contact control
    # sapien.physx.set_gpu_memory_config(max_rigid_contact_count=6553600)


@dataclass
class GPUMemoryConfig:
    """GPU buffer capacities sized from the active `num_envs` value."""
    temp_buffer_capacity: int = 2**17
    max_rigid_contact_count: int = 2**16
    max_rigid_patch_count: int = 2**13
    heap_capacity: int = 2**19
    found_lost_pairs_capacity: int = 2**18
    found_lost_aggregate_pairs_capacity: int = 2**4
    total_aggregate_pairs_capacity: int = 2**4
    collision_stack_size: int = 2**18

    def as_dict(self) -> dict[str, int]:
        """Return a plain dictionary accepted by `sapien.physx` helpers."""
        return {
            "temp_buffer_capacity": self.temp_buffer_capacity,
            "max_rigid_contact_count": self.max_rigid_contact_count,
            "max_rigid_patch_count": self.max_rigid_patch_count,
            "heap_capacity": self.heap_capacity,
            "found_lost_pairs_capacity": self.found_lost_pairs_capacity,
            "found_lost_aggregate_pairs_capacity": self.found_lost_aggregate_pairs_capacity,
            "total_aggregate_pairs_capacity": self.total_aggregate_pairs_capacity,
            "collision_stack_size": self.collision_stack_size,
        }

    def dict(self) -> dict[str, int]:
        """Backward-compatible alias for legacy call sites."""
        return self.as_dict()

    def scale_for_envs(self, num_envs: int) -> "GPUMemoryConfig":
        """Scale every GPU buffer linearly with `num_envs`.

        The values are rounded to the next power of two because PhysX GPU
        allocators behave better with aligned capacities and the historical
        defaults were already expressed as powers of two.
        """

        env_count = max(1, num_envs)
        scaled_fields = {
            field_name: _next_power_of_two(field_value * env_count)
            for field_name, field_value in self.as_dict().items()
        }
        return GPUMemoryConfig(**scaled_fields)


@dataclass
class SimParams:
    """High-level simulator configuration shared across scenes."""

    # up_axis
    gravity: NDArray[np.float32] = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.8], dtype=np.float32)
    )
    dt: float = 1.0 / 60.0
    substeps: int = 2
    use_gpu_pipeline: bool = True
    physx: PhysxParams = field(default_factory=PhysxParams)
    gpu_memory: GPUMemoryConfig = field(default_factory=GPUMemoryConfig)


@dataclass
class AssetOptions:
    """URDF loading options used while creating SAPIEN actors."""

    flip_visual_attachments: bool = False  # NOTE:(Yuke)
    fix_base_link: bool = False
    thickness: float = 0.0
    density: float = 600.0
    armature: float = 0.01
    linear_damping: float = 0.0
    max_linear_velocity: float = 1000.0
    angular_damping: float = 0.0
    max_angular_velocity: float = 1000.0
    disable_gravity: bool = False
    enable_gyroscopic_forces: bool = True
    # NOTE(YUKE): Sapien cannot set the following parameters: flip_visual_attachments, thickness, max_linear_velocity, max_angular_velocity, enable_gyroscopic_forces


@dataclass
class CameraCfg:
    """Simple camera specification used by the simulator."""

    name: Optional[str] = None
    width: int = 1280
    height: int = 720
    fovy: float = np.deg2rad(40)
    near: float = 0.001
    far: float = 2.0


# Simulator options.
sim_params = SimParams()
# sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = np.array([0.0, 0.0, -9.8])
sim_params.dt = 1.0 / 60.0

# Isaac Gym advances each 1/60 s simulation step with two 1/120 s solver
# substeps. SAPIEN exposes one PhysX step, so the environment flattens these
# substeps explicitly when configuring the GPU system.
sim_params.substeps = 2
sim_params.use_gpu_pipeline = True
sim_params.physx.solver_type = 1
sim_params.physx.bounce_threshold_velocity = 0.02

# Increasing this can make the simulation more stable.
sim_params.physx.num_position_iterations = 20
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.rest_offset = 0.0000
sim_params.physx.contact_offset = 0.002
sim_params.physx.friction_offset_threshold = 0.01
sim_params.physx.friction_correlation_distance = 0.0005
sim_params.physx.max_depenetration_velocity = 10
sim_params.physx.use_gpu = True

# Can set these if contacts are being weird
# sim_params.physx.max_gpu_contact_pairs = 6553600  # 50 * 1024 * 1024
# sim_params.physx.default_buffer_size_multiplier = 8.0


sim_config["sim_params"] = sim_params
sim_config["parts"] = {"friction": 0.15}
sim_config["table"] = {"friction": 0.10}
sim_config["asset"] = {}

# Parameters for the robot.
sim_config["robot"].update(
    {
        "kp": [90, 90, 90, 70.0, 60.0, 80.0],  # Default positional gains.
        # "kp": [270, 270, 270, 210, 180, 240],  # Cranked up gains
        "kv": None,  # Default velocity gains.
        "arm_frictions": 0.05,  # Default arm friction.
        "gripper_frictions": 15.0,
        "gripper_torque": 13,
    }
)

# Parameters for the light.
sim_config["lights"] = [
    {
        "color": [1.1, 1.1, 1.1],
        "ambient": [0.25, 0.25, 0.25],
        "direction": [1.0, -1.0, -2.0],
        "shadow": True,
    },
    {
        "color": [0.55, 0.55, 0.55],
        "ambient": [0.25, 0.25, 0.25],
        "direction": [-1.0, 0.5, -1.0],
        "shadow": False,
    },
]

"""
Set density for each furniture part.
  - The volume is estimated using Belnder.
  - The mass is estimated using 3D printer slicer.
"""


def default_asset_options() -> AssetOptions:
    """Build the default asset options shared by furniture parts."""
    asset_options = AssetOptions()
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = False
    asset_options.thickness = 0.0
    asset_options.density = 600.0
    asset_options.armature = 0.2
    asset_options.linear_damping = 0.001
    asset_options.max_linear_velocity = 1000.0
    asset_options.angular_damping = 0.001
    asset_options.max_angular_velocity = 1000.0
    asset_options.disable_gravity = False
    asset_options.enable_gyroscopic_forces = True

    return asset_options


# Volume: 302802 mm^3
# Mass: 151g
square_table_top_asset_options = default_asset_options()
square_table_top_asset_options.density = 498.68
sim_config["asset"]["square_table_top"] = square_table_top_asset_options

# Volume: 62435.mm^3
# Mass: 23.1g
leg_asset_options = default_asset_options()
leg_asset_options.density = 369.98
sim_config["asset"]["square_table_leg1"] = leg_asset_options
sim_config["asset"]["square_table_leg2"] = leg_asset_options
sim_config["asset"]["square_table_leg3"] = leg_asset_options
sim_config["asset"]["square_table_leg4"] = leg_asset_options

# Cabinet.
# Volume: 224623 mm^3
# Mass: 130.98g
cabinet_body_asset_options = default_asset_options()
cabinet_body_asset_options.density = 583.11
sim_config["asset"]["cabinet_body"] = cabinet_body_asset_options

# Volume: 73208 mm^3
# Mass: 30.2g
cabinet_door_left_asset_options = default_asset_options()
cabinet_door_left_asset_options.density = 412.52
sim_config["asset"]["cabinet_door_left"] = cabinet_door_left_asset_options
sim_config["asset"]["cabinet_door_right"] = cabinet_door_left_asset_options

# Volume: 192689 mm^3
# Mass: 60.29g
cabinet_top_asset_options = default_asset_options()
cabinet_top_asset_options.density = 312.89
sim_config["asset"]["cabinet_top"] = cabinet_top_asset_options

# Desk.
# Volume: 343624 mm^3
# Mass: 169.4g
desk_top_asset_options = default_asset_options()
desk_top_asset_options.density = 492.98
sim_config["asset"]["desk_top"] = desk_top_asset_options

# Volume: 181892 mm^3
# Mass: 56.2g
desk_leg1_asset_options = default_asset_options()
desk_leg1_asset_options.density = 308.92
sim_config["asset"]["desk_leg1"] = desk_leg1_asset_options
sim_config["asset"]["desk_leg2"] = desk_leg1_asset_options
sim_config["asset"]["desk_leg3"] = desk_leg1_asset_options
sim_config["asset"]["desk_leg4"] = desk_leg1_asset_options

# Round table.
# Volume: 257631 mm^3
# Mass: 121.69g
round_table_top_asset_options = default_asset_options()
round_table_top_asset_options.density = 472.34
sim_config["asset"]["round_table_top"] = round_table_top_asset_options

# Volume: 75321 mm^3
# Mass:  32.28g
round_table_leg_asset_options = default_asset_options()
round_table_leg_asset_options.density = 414.52
sim_config["asset"]["round_table_leg"] = round_table_leg_asset_options

# Volume: 81926 mm^3
# Mass: 33.96g
round_table_base_asset_options = default_asset_options()
round_table_base_asset_options.density = 533.11
sim_config["asset"]["round_table_base"] = round_table_base_asset_options

# Drawer
# Volume: 221853 mm^3
# Mass: 151.63g
drawer_box_asset_options = default_asset_options()
drawer_box_asset_options.density = 683.47
sim_config["asset"]["drawer_box"] = drawer_box_asset_options

# Volume:  92893 mm^3
# Mass: 59.37g
drawer_container_top_asset_options = default_asset_options()
drawer_container_top_asset_options.density = 639.1
sim_config["asset"]["drawer_container_top"] = drawer_container_top_asset_options
sim_config["asset"]["drawer_container_bottom"] = drawer_container_top_asset_options

# Chair
# Volume: 111594 mm^3
# MAss: 61.87g
chair_seat_asset_options = default_asset_options()
chair_seat_asset_options.density = 554.42
sim_config["asset"]["chair_seat"] = chair_seat_asset_options

# Volume: 354703 mm^3
# Mass: 123.16g
chair_back_asset_options = default_asset_options()
chair_back_asset_options.density = 347.22
sim_config["asset"]["chair_back"] = chair_back_asset_options

# Volume: 60139 mm^3
# MAss: 22.44g
chair_leg1_asset_options = default_asset_options()
chair_leg1_asset_options.density = 373.14
sim_config["asset"]["chair_leg1"] = chair_leg1_asset_options
sim_config["asset"]["chair_leg2"] = chair_leg1_asset_options

# Volume: 20083 mm^3
# Mass: 10.15g
chair_nut1_asset_options = default_asset_options()
chair_nut1_asset_options.density = 505.40
sim_config["asset"]["chair_nut1"] = chair_nut1_asset_options
sim_config["asset"]["chair_nut2"] = chair_nut1_asset_options

# Lamp
# Volume:  78694 mm^3
# Mass: 59.99g
lamp_hood_asset_options = default_asset_options()
lamp_hood_asset_options.density = 762.31
# lamp_hood_asset_options.density = 200
sim_config["asset"]["lamp_hood"] = lamp_hood_asset_options

# Volume:  174649 mm^3
# Mass: 59.65g
lamp_base_asset_options = default_asset_options()
lamp_base_asset_options.density = 341.54
sim_config["asset"]["lamp_base"] = lamp_base_asset_options

# Volume: 70576 mm^3
# Mass: 38.47g
lamp_bulb_asset_options = default_asset_options()
lamp_bulb_asset_options.density = 545.09
# lamp_bulb_asset_options.density = 369.98
# lamp_bulb_asset_options.density = 100
sim_config["asset"]["lamp_bulb"] = lamp_bulb_asset_options

# Stool
# Volume: 103515 mm^3
# Mass: 57.34g
stool_seat_asset_options = default_asset_options()
stool_seat_asset_options.density = 553.93
sim_config["asset"]["stool_seat"] = stool_seat_asset_options

# Volume: 81131 mm^3
# Mass: 27.07g
stool_leg1_asset_options = default_asset_options()
stool_leg1_asset_options.density = 333.66
sim_config["asset"]["stool_leg1"] = stool_leg1_asset_options
sim_config["asset"]["stool_leg2"] = stool_leg1_asset_options
sim_config["asset"]["stool_leg3"] = stool_leg1_asset_options

rack_asset_options = default_asset_options()
rack_asset_options.density = 100.00
sim_config["asset"]["rack"] = rack_asset_options

mug_asset_options = default_asset_options()
mug_asset_options.density = 100.00
sim_config["asset"]["mug"] = mug_asset_options

factory_nut_asset_options = default_asset_options()
# factory_nut_asset_options.density = 1000.00
factory_nut_asset_options.density = 500.00
sim_config["asset"]["factory_nut"] = factory_nut_asset_options

factory_bolt_asset_options = default_asset_options()
# factory_bolt_asset_options.density = 1000.00
factory_bolt_asset_options.density = 500.00
sim_config["asset"]["factory_bolt"] = factory_bolt_asset_options

factory_peg_asset_options = default_asset_options()
# factory_peg_asset_options.density = 1000.00
factory_peg_asset_options.density = 500.00
sim_config["asset"]["factory_peg"] = factory_peg_asset_options

factory_hole_asset_options = default_asset_options()
# factory_hole_asset_options.density = 1000.00
factory_hole_asset_options.density = 500.00
sim_config["asset"]["factory_hole"] = factory_hole_asset_options
