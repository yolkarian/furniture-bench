"""Offline IQL training entry point for the refactored FurnitureBench project."""

from __future__ import annotations

import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import D4RLDataset, FurnitureDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./checkpoints/", "TensorBoard logging directory.")
flags.DEFINE_string("run_name", "debug", "Run-specific name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000000, "Evaluation interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Checkpoint interval.")
flags.DEFINE_integer("batch_size", 256, "Mini-batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("data_path", "", "Path to the offline dataset.")
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "Path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_boolean("wandb", False, "Use Weights & Biases logging.")
flags.DEFINE_string("wandb_project", "", "Weights & Biases project name.")
flags.DEFINE_string("wandb_entity", "", "Weights & Biases entity name.")


def normalize(dataset: D4RLDataset) -> None:
    """Normalize reward scale for classic D4RL locomotion datasets."""
    trajectories = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(trajectory) -> float:
        return sum(reward for _, _, reward, _, _, _ in trajectory)

    trajectories.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajectories[-1]) - compute_returns(trajectories[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(
    env_name: str, seed: int, data_path: str
) -> Tuple[gym.Env, D4RLDataset]:
    """Create the evaluation environment and the matching offline dataset."""
    if "Furniture" in env_name:
        import furniture_bench  # noqa: F401

        env_id, furniture_name = env_name.split("/")
        env = gym.make(env_id, furniture=furniture_name, data_path=data_path)
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    if "Furniture" in env_name:
        dataset = FurnitureDataset(data_path)
    else:
        dataset = D4RLDataset(env)

    if "antmaze" in env_name:
        dataset.rewards -= 1.0
    elif "halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name:
        normalize(dataset)

    return env, dataset


def main(_: object) -> None:
    """Train an offline IQL policy and periodically evaluate it."""
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    tensorboard_dir = os.path.join(
        FLAGS.save_dir, "tb", f"{FLAGS.run_name}.{FLAGS.seed}"
    )
    checkpoint_dir = os.path.join(
        FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}"
    )

    # 1. Build the environment and offline dataset.
    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.data_path)
    learner_kwargs = dict(FLAGS.config)

    # 2. Configure experiment logging.
    if FLAGS.wandb:
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=f"{FLAGS.env_name}-{FLAGS.seed}-{FLAGS.data_path}",
            config=learner_kwargs,
            sync_tensorboard=True,
        )
    summary_writer = SummaryWriter(tensorboard_dir, write_to_disk=True)

    # 3. Initialize the learner with raw state/image observations only.
    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        use_encoder=False,
        **learner_kwargs,
    )

    evaluation_returns = []
    for step in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if step % FLAGS.log_interval == 0:
            for key, value in update_info.items():
                if value.ndim == 0:
                    summary_writer.add_scalar(f"training/{key}", value, step)
                else:
                    summary_writer.add_histogram(
                        f"training/{key}", np.array(value), step
                    )
            summary_writer.flush()

        if step % FLAGS.eval_interval == 0:
            evaluation_stats = evaluate(agent, env, FLAGS.eval_episodes)
            for key, value in evaluation_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{key}s", value, step)
            summary_writer.flush()

            evaluation_returns.append((step, evaluation_stats["return"]))
            np.savetxt(
                os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"),
                evaluation_returns,
                fmt=["%d", "%.1f"],
            )

        if step % FLAGS.ckpt_interval == 0:
            agent.save(checkpoint_dir, step)

    if step % FLAGS.ckpt_interval != 0:
        agent.save(checkpoint_dir, step)

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
