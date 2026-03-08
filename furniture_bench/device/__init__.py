"""Device-interface factory helpers."""

from __future__ import annotations

from furniture_bench.device.device_interface import DeviceInterface


def make_device(device_name: str) -> DeviceInterface:
    """Create a supported teleoperation device interface.

    Args:
        device_name: One of ``keyboard``, ``oculus``, or ``keyboard-oculus``.

    Returns:
        The initialized device interface.
    """
    if device_name == "keyboard":
        from furniture_bench.device.keyboard_interface import KeyboardInterface

        device: DeviceInterface = KeyboardInterface()
    elif device_name == "oculus":
        from furniture_bench.device.oculus_interface import OculusInterface

        device = OculusInterface()
    elif device_name == "keyboard-oculus":
        from furniture_bench.device.keyboard_oculus_interface import (
            KeyboardOculusInterface,
        )

        device = KeyboardOculusInterface()
    else:
        raise ValueError(
            "Unrecognized device: "
            f"{device_name}. Choose one of 'keyboard', 'oculus', or 'keyboard-oculus'."
        )

    device.print_usage()
    return device


# Backward-compatible alias used by older scripts.
make = make_device
