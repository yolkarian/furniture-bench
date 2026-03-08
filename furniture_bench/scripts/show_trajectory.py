"""Visualize a recorded FurnitureBench trajectory."""

from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Build the trajectory-visualization CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to collected data")
    parser.add_argument(
        "--data-path", help="Path to collected data of single pickle file."
    )
    parser.add_argument(
        "--channel-first", help="Path to collected data .pkl", action="store_true"
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--show-raw-images", action="store_true", help="Show original images."
    )
    parser.add_argument(
        "--speed-up", type=int, default=5, help="Speed up the video by this factor."
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def get_frames_from_video(video_path: Path) -> list[np.ndarray]:
    """Read every frame from a video file into memory."""
    video = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break
        frames.append(frame)
    video.release()
    return frames


def resolve_pickle_path(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Resolve the pickle file path and optional demo directory."""
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        return data_dir / f"{data_dir.name}.pkl", data_dir
    if args.data_path is not None:
        return Path(args.data_path), None
    raise ValueError("Either data_dir or data_path must be specified.")


def load_raw_media(data_dir: Path, length: int) -> tuple[list[np.ndarray], ...]:
    """Load the raw RGB videos and depth PNG sequences for a trajectory."""
    print("Start reading depth images...")
    depth_sequences: list[list[np.ndarray]] = []
    for camera_idx in range(1, 4):
        depth_dir = data_dir / f"{data_dir.name}_depth_image{camera_idx}"
        depth_paths = sorted(depth_dir.glob("*.png"))
        depth_sequences.append(
            [cv2.imread(str(path), -1) for path in depth_paths[:length]]
        )

    print("Start reading color videos...")
    video_paths = [
        data_dir / f"{data_dir.name}_color_image1.mp4",
        data_dir / f"{data_dir.name}_color_image2.mp4",
        data_dir / f"{data_dir.name}_color_image3.mp4",
    ]
    color_sequences = [get_frames_from_video(path) for path in video_paths]
    return (*color_sequences, *depth_sequences)


def colorize_depth(
    depth_image: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Convert a raw depth frame into a display-friendly colormap."""
    colored = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.1),
        cv2.COLORMAP_JET,
    )
    return cv2.resize(colored, target_shape)


def main(argv: Sequence[str] | None = None) -> None:
    """Render a stored trajectory in an OpenCV window."""
    np.set_printoptions(suppress=True)
    args = parse_args(argv)

    pickle_path, data_dir = resolve_pickle_path(args)

    video_writer: cv2.VideoWriter | None = None
    if args.save_video:
        output_path = Path(datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".avi")
        frame_size = (224 * 2, 224)
        video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            10,
            frame_size,
            True,
        )

    with open(pickle_path, "rb") as file_obj:
        data = pickle.load(file_obj)

    trajectory_length = len(data["actions"])
    rewards: list[float] = []
    sum_skills = 0

    raw_media: tuple[list[np.ndarray], ...] | None = None
    if data_dir is not None and args.show_raw_images:
        raw_media = load_raw_media(data_dir, trajectory_length)

    for step_idx in range(trajectory_length):
        observation = data["observations"][step_idx]

        if "color_image1" in data["observations"][0]:
            color_image1 = observation["color_image1"]
            color_image2 = observation["color_image2"]
        else:
            # Converted datasets may keep the image tensors under alternative keys.
            color_image1 = observation["image1"]
            color_image2 = observation["image2"]

        if args.channel_first:
            color_image1 = np.moveaxis(color_image1, 0, -1)
            color_image2 = np.moveaxis(color_image2, 0, -1)

        color_image1 = cv2.cvtColor(color_image1, cv2.COLOR_RGB2BGR)
        color_image2 = cv2.cvtColor(color_image2, cv2.COLOR_RGB2BGR)
        color_img = np.hstack([color_image1, color_image2])

        if data["rewards"][step_idx] != 0:
            rewards.append(data["rewards"][step_idx])
        print(observation["robot_state"])

        cv2.putText(
            np.ascontiguousarray(color_img),
            f"rewards: {rewards}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 255, 0),
        )

        if data["skills"][step_idx] != 0:
            sum_skills += data["skills"][step_idx]

        cv2.putText(
            np.ascontiguousarray(color_img),
            f"skills: {sum_skills}",
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 0, 255),
        )

        cv2.imshow("Trajectory", color_img)

        if raw_media is not None:
            (
                color_images1,
                color_images2,
                color_images3,
                depth_images1,
                depth_images2,
                depth_images3,
            ) = raw_media
            raw_color1 = cv2.cvtColor(color_images1[step_idx], cv2.COLOR_RGB2BGR)
            raw_color2 = cv2.cvtColor(color_images2[step_idx], cv2.COLOR_RGB2BGR)
            raw_color3 = cv2.cvtColor(color_images3[step_idx], cv2.COLOR_RGB2BGR)
            raw_color = np.hstack([raw_color1, raw_color2, raw_color3])

            target_shape = (raw_color1.shape[1], raw_color1.shape[0])
            raw_depth = np.hstack(
                [
                    colorize_depth(depth_images1[step_idx], target_shape),
                    colorize_depth(depth_images2[step_idx], target_shape),
                    colorize_depth(depth_images3[step_idx], target_shape),
                ]
            )
            raw_image = np.vstack([raw_color, raw_depth])
            cv2.imshow("Original Trajectory", raw_image)

        if video_writer is not None:
            video_writer.write(color_img)

        time.sleep(0.1 / args.speed_up)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

    if video_writer is not None:
        video_writer.release()


if __name__ == "__main__":
    main()
