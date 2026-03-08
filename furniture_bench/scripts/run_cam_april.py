"""Visualize AprilTag detections from the RealSense camera setup."""

from __future__ import annotations

import argparse
from typing import Sequence

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Build the AprilTag-visualization CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-depth", action="store_true", help="Show depth image.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def detect_draw(april_tag: object, cam: object) -> tuple[np.ndarray, np.ndarray]:
    """Run AprilTag detection on a single camera frame and draw the results."""
    from furniture_bench.utils.draw import draw_tags

    color_frame, depth_frame = cam.get_frame()
    depth_image = np.asanyarray(depth_frame.get_data()).copy()
    rgb_image = np.asanyarray(color_frame.get_data()).copy()

    tags = april_tag.detect(color_frame, cam.intr_param)
    draw_image = draw_tags(rgb_image.copy(), cam, tags)
    return draw_image, depth_image


def main(argv: Sequence[str] | None = None) -> None:
    """Open an interactive window showing AprilTag detections."""
    args = parse_args(argv)

    # Delay camera imports until after parsing so ``--help`` does not require
    # RealSense or AprilTag runtime dependencies.
    from furniture_bench.config import config
    from furniture_bench.perception.apriltag import AprilTag
    from furniture_bench.perception.realsense import RealsenseCam

    cam1 = RealsenseCam(
        config["camera"][1]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
    )
    cam2 = RealsenseCam(
        config["camera"][2]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
        None,
        disable_auto_exposure=True,
    )
    cam3 = RealsenseCam(
        config["camera"][3]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
    )
    april_tag = AprilTag(tag_size=0.0195)

    cv2.namedWindow("RealsenseAprilTag", cv2.WINDOW_AUTOSIZE)

    while True:
        color_img1, depth_image1 = detect_draw(april_tag, cam1)
        color_img2, depth_image2 = detect_draw(april_tag, cam2)
        color_img3, depth_image3 = detect_draw(april_tag, cam3)

        # Always show the RGB streams side by side.
        color_img = np.hstack([color_img1, color_img2, color_img3])
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        if args.show_depth:
            depth_image1 = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image1, alpha=0.1), cv2.COLORMAP_JET
            )
            depth_image2 = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image2, alpha=0.1), cv2.COLORMAP_JET
            )
            depth_image3 = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image3, alpha=0.1), cv2.COLORMAP_JET
            )

            depth_image1 = cv2.resize(
                depth_image1, (color_img1.shape[1], color_img1.shape[0])
            )
            depth_image2 = cv2.resize(
                depth_image2, (color_img1.shape[1], color_img2.shape[0])
            )
            depth_image3 = cv2.resize(
                depth_image3, (color_img1.shape[1], color_img3.shape[0])
            )
            depth_img = np.hstack([depth_image1, depth_image2, depth_image3])
            image = np.vstack([color_img, depth_img])
        else:
            image = color_img

        cv2.imshow("Detected tags", image)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
