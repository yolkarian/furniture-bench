from pathlib import Path
from setuptools import find_packages, setup


long_description = (Path(__file__).parent / "README.md").read_text()

core_requirements = [
    "opencv-python",
    # "gym==0.21.0",
    "gymnasium==0.29.1",
    "pynput==1.7.7",
    "joblib",
    "numba",
    "pyrealsense2",
    "dt-apriltags",
    "rich",
    "tqdm",
]

setup(
    name="furniture_bench",
    version="0.1",
    author="CLVR @ KAIST",
    author_email="clvr.kaist@gmail.com",
    url="https://github.com/clvrai/furniture-bench",
    description="FurnitureBench: Reproducible Real-World Furniture Assembly Benchmark (RSS 2023)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.8,<3.10",
    install_requires=core_requirements,
)
