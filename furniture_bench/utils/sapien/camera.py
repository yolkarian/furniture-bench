import os
from dataclasses import dataclass
from typing import Dict, Callable, Optional
import torch
import sapien


SAPIEN_SHADERS_DIR = os.path.join(os.path.dirname(sapien.__file__),"vulkan_shader")


def default_position_texture_transform(data: torch.Tensor):
    position = data[..., :3]
    depth = -position[..., [2]]
    return {
        "depth": depth,
        "position": position,
    }

@dataclass
class ShaderConfig:
    shader_name:str
    shader_dir:str
    obs_keys_2_texture_name:Dict[str, str]
    output_transform:Dict[str, Callable[[torch.Tensor], Dict[str, torch.Tensor]]]
    shader_params:Optional[dict] = None

DEFAULT_SHADER_CONFIG =  ShaderConfig(
    shader_name="default",
    shader_dir=os.path.join(SAPIEN_SHADERS_DIR, "default"),
    obs_keys_2_texture_name={
        "color":"Color",
        "position":"Position",
        "depth":"Position",
        "segmentation": "Segmentation",
        "normal":"Normal",
        "albedo":"Albedo",
    },
    output_transform={
        "Color": lambda data: {"color": (data[..., :3] * 255).to(torch.uint8)},
        "Position": default_position_texture_transform,
        # note in default shader pack, 0 is visual shape / mesh, 1 is actor/link level, 2 is parallel scene ID, 3 is unused
        "Segmentation": lambda data: {"segmentation": data[..., 1][..., None]},
        "Normal": lambda data: {"normal": data[..., :3]},
        "Albedo": lambda data: {"albedo": (data[..., :3] * 255).to(torch.uint8)},
    },
)

MINIMAL_SHADER_CONFIG = ShaderConfig(
    shader_name="minimal",
    shader_dir=os.path.join(SAPIEN_SHADERS_DIR, "minimal"),
    obs_keys_2_texture_name={
        "color":"Color",
        "position":"PositionSegmentation",
        "depth":"PositionSegmentation",
        "segmentation": "PositionSegmentation",
        "normal":"Normal",
        "albedo":"Albedo",
    },
    output_transform={
            "Color": lambda data: {"color": data[..., :3]},
            "PositionSegmentation": lambda data: {
                "position": data[
                    ..., :3
                ].to(torch.float32) * 1e-3,  # position for minimal is in millimeters and is uint16
                "depth": -data[..., [2]].to(torch.float32) * 1e-3,
                "segmentation": data[..., [3]],
            },
        },
)


RT_SHADER_CONFIG =  ShaderConfig(
    shader_name="rt",
    shader_dir=os.path.join(SAPIEN_SHADERS_DIR, "rt"),
    obs_keys_2_texture_name={
        "color":"Color",
        "position":"Position",
        "depth":"Position",
        "segmentation": "Segmentation",
        "normal":"Normal",
        "albedo":"Albedo",
    },
    output_transform = {
        "Color": lambda data: {"color": (data[..., :3] * 255).to(torch.uint8)},
        "Position": default_position_texture_transform,
        # note in default shader pack, 0 is visual shape / mesh, 1 is actor/link level, 2 is parallel scene ID, 3 is unused
        "Segmentation": lambda data: {"segmentation": data[..., 1][..., None]},
        "Normal": lambda data: {"normal": data[..., :3]},
        "Albedo": lambda data: {"albedo": (data[..., :3] * 255).to(torch.uint8)},
    },

    shader_params = {
        "ray_tracing_samples_per_pixel": 2,
        "ray_tracing_path_depth": 1,
        "ray_tracing_denoiser": "oidn",
    },
)

SHADER_DICT = {
    DEFAULT_SHADER_CONFIG.shader_name: DEFAULT_SHADER_CONFIG,
    MINIMAL_SHADER_CONFIG.shader_name: MINIMAL_SHADER_CONFIG,
    RT_SHADER_CONFIG.shader_name:RT_SHADER_CONFIG,
}

def set_shader(shader_config: ShaderConfig):
    """sets a global shader pack for cameras. Used only for the 3.0 SAPIEN rendering system"""
    sapien.render.set_camera_shader_dir(shader_config.shader_dir)
    if shader_config.shader_name == "minimal":
        sapien.render.set_camera_shader_dir("minimal")
        sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
        sapien.render.set_picture_format("ColorRaw", "r8g8b8a8unorm")
        sapien.render.set_picture_format("PositionSegmentation", "r16g16b16a16sint")
    if shader_config.shader_name == "default":
        sapien.render.set_camera_shader_dir("default")
        sapien.render.set_picture_format("Color", "r32g32b32a32sfloat")
        sapien.render.set_picture_format("ColorRaw", "r32g32b32a32sfloat")
        sapien.render.set_picture_format(
            "PositionSegmentation", "r32g32b32a32sfloat"
        )
    if shader_config.shader_name[:2] == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(
            shader_config.shader_params["ray_tracing_samples_per_pixel"]
        )
        sapien.render.set_ray_tracing_path_depth(
            shader_config.shader_params["ray_tracing_path_depth"]
        )
        sapien.render.set_ray_tracing_denoiser(
            shader_config.shader_params["ray_tracing_denoiser"]
        )
