import gymnasium as gym
import mani_skill.envs

# Further investigation of the implementation of the parallel in one Scene. 
# This is exactly as the name indicates. It just put all objects in one Scene

env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    sim_backend="gpu",
    render_backend="gpu",
    num_envs=16,
    parallel_in_single_scene=False,
    viewer_camera_configs=dict(shader_pack="rt-fast"),
)
env.reset()
print(env.scene.px.cuda_articulation_qpos.torch().shape)
print(len(env.scene.sub_scenes[-1].get_entities()), len(env.scene.sub_scenes[0].get_entities()))
while True:
    env.step(env.action_space.sample())
    env.render_human()
