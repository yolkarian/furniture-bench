"""Offline IQL evaluation entry point for the refactored FurnitureBench project."""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import wrappers
from evaluation import evaluate
from learner import Learner
from furniture_bench.utils.checkpoint import download_ckpt_if_not_exists

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Checkpoint root directory.")
flags.DEFINE_string("run_name", "debug", "Run-specific name.")
flags.DEFINE_string("ckpt_step", None, "Specific checkpoint step.")
flags.DEFINE_string("randomness", "low", "Randomness mode.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("max_steps", int(1e6), "Training horizon used by the checkpoint.")
flags.DEFINE_integer("from_skill", 0, "Skill index to start from.")
flags.DEFINE_integer("skill", -1, "Skill index to evaluate.")
flags.DEFINE_integer("high_random_idx", 0, "Index for high-randomness presets.")
flags.DEFINE_boolean("tqdm", True, "Unused compatibility flag.")
flags.DEFINE_boolean("record", False, "Record evaluation video.")
flags.DEFINE_float("temperature", 0.0, "Policy sampling temperature.")
flags.DEFINE_boolean("headless", False, "Run in headless mode.")
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "Path to the training hyperparameter configuration.",
    lock_config=False,
)


def make_env(
    env_name: str,
    seed: int,
    record: bool,
    from_skill: int,
    skill: int,
    randomness: str,
    high_random_idx: int,
    headless: bool,
) -> gym.Env:
    """Create the evaluation environment for offline checkpoints."""
    if "Furniture" in env_name:
        import furniture_bench  # noqa: F401

        env_id, furniture_name = env_name.split("/")
        env = gym.make(
            env_id,
            furniture=furniture_name,
            use_all_cam=False,
            record=record,
            disable_env_checker=True,
            from_skill=from_skill,
            skill=skill,
            high_random_idx=high_random_idx,
            randomness=randomness,
            headless=headless,
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    return env


def main(_: object) -> None:
    """Load a local checkpoint and evaluate it in the target environment."""
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, "eval"), exist_ok=True)
    eval_path = os.path.join(
        FLAGS.save_dir, "eval", f"{FLAGS.run_name}.{FLAGS.seed}"
    )

    if "Sim" in FLAGS.env_name:
        import sapien  # noqa: F401

    # 1. Build the environment with raw observations only.
    env = make_env(
        FLAGS.env_name,
        FLAGS.seed,
        record=FLAGS.record,
        from_skill=FLAGS.from_skill,
        skill=FLAGS.skill,
        high_random_idx=FLAGS.high_random_idx,
        randomness=FLAGS.randomness,
        headless=FLAGS.headless,
    )

    # 2. Reconstruct the learner used during offline training.
    learner_kwargs = dict(FLAGS.config)
    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        use_encoder=False,
        **learner_kwargs,
    )

    # 3. Resolve and load a local checkpoint directory.
    checkpoint_dir = download_ckpt_if_not_exists(
        os.path.join(FLAGS.save_dir, "ckpt"), FLAGS.run_name, FLAGS.seed
    )
    agent.load(str(checkpoint_dir), FLAGS.ckpt_step or FLAGS.max_steps)

    evaluation_stats = evaluate(agent, env, FLAGS.eval_episodes, FLAGS.temperature)
    np.savetxt(eval_path, [evaluation_stats["return"]], fmt=["%.1f"])


if __name__ == "__main__":
    app.run(main)
