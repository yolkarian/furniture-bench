# FurnitureBench: Reproducible Real-World Furniture Assembly Benchmark

[**Paper**](http://arxiv.org/abs/2305.12821)
| [**Website**](https://clvrai.com/furniture-bench/)
| [**Documentation**](https://clvrai.github.io/furniture-bench/docs/index.html)

![FurnitureBench](furniture_bench_banner.jpg)

FurnitureBench is the real-world furniture assembly benchmark, which aims at providing a reproducible and easy-to-use platform for long-horizon complex robotic manipulation.

This is a fork of FurnitureBench. The main difference in this fork is that it utilizes [SAPIEN](https://github.com/haosulab/SAPIEN) as the simulator for FurnitureSim with support for Python 3.9+. (For this installation, please refer to [FurnitureSim section](#FurnitureSim)) and also guide and tutorials in the [online document](https://clvrai.github.io/furniture-bench/docs/index.html) of the original author.

It features
- Long-horizon complex manipulation tasks
- Standardized environment setup
- Python-based robot control stack
- FurnitureSim: a simulated environment
- Large-scale teleoperation dataset (200+ hours)

Please check out our [website](https://clvrai.com/furniture-bench/) for more details.


## FurnitureBench

We elaborate on the real-world environment setup guide and tutorials in our [online document](https://clvrai.github.io/furniture-bench/docs/index.html).



## FurnitureSim

FurnitureSim in this fork is a simulator based on SAPIEN. FurnitureSim works on any Linux and Python 3.9. Original FurnitureSim is a simulator based on Isaac Gym. 


### SAPIEN Installation 
For the installation of FurnitureSim of this branch, please download the wheels in releases of this [repository](https://github.com/yolkarian/SAPIEN) (Customized SAPIEN simulator with fixed URDF loader) and use `pip` to install.

You might not need CUDA for the installation of SAPIEN but NVIDIA GPU is needed since we enforce GPU simulation for FurnitureSim. In addition, the NVIDIA GPU driver should meet some requirements. Please find the details on this [page](https://sapien-sim.github.io/docs/user_guide/getting_started/installation.html)

### FurnitureBench Installation

After installation of SAPIEN, in the folder of this repo, run:
```bash
pip install -e .
```

## Roadmap

### Tested

- Simulator with Data Collection
- Successful furniture assembly with handcrafted scripts.

### TODO

- More realistic Rasterization-based Render (New Shader)
- Removal of Isaacgym in bash scripts
- Recording during the simulation
- Evaluation with approaches from the original FurnitureBench
- Test reinforcement learning (fine-tuning) in the environment


## Citation

If you find FurnitureBench useful for your research, please cite this work:
```
@inproceedings{heo2023furniturebench,
    title={FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation},
    author={Minho Heo and Youngwoon Lee and Doohyun Lee and Joseph J. Lim},
    booktitle={Robotics: Science and Systems},
    year={2023}
}
```


## References

- Polymetis: https://github.com/facebookresearch/polymetis
- BC: Youngwoon's [robot-learning repo](https://github.com/youngwoon/robot-learning).
- IQL: https://github.com/ikostrikov/implicit_q_learning
- R3M: https://github.com/facebookresearch/r3m
- VIP: https://github.com/facebookresearch/vip
- Factory: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/factory.md
- OSC controller references: https://github.com/StanfordVL/perls2 and https://github.com/ARISE-Initiative/robomimic and https://github.com/ARISE-Initiative/robosuite

