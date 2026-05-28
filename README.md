# DroneAI

This is a repository for the DroneAI project(presented at a Conference, Defense M&S Seminar), which focuses on developing artificial intelligence algorithms for autonomous drones.

Note that Code base is [WindJammer6](https://github.com/WindJammer6/35.-Star-Wars-Reinforcement-Learning)

# Frontier-based Exploration
## Paper
- Yamauchi, B. (1997). *A frontier-based approach for autonomous exploration*. Proceedings of IEEE International Symposium on Computational Intelligence in Robotics and Automation (CIRA'97).

## Code References
- [**Frontier-Based-Exploration** (Python, educational)](https://github.com/Topiwala/Frontier-Based-Exploration)

# CPP (Coverage Path Planning)
## Paper
- Choset, H., & Pignon, P. (1998). *Coverage path planning: The boustrophedon cellular decomposition*. Proceedings of International Conference on Field and Service Robotics.
## Code References
- [**coverage-path-planning-python** (Python Robotics)](https://github.com/AtsushiSakai/PythonRobotics)

# Rainbow DQN
## Paper
- [Rainbow (AAAI 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11796) · [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)*
| [2] | Deep Reinforcement Learning with Double Q-learning | van Hasselt, H., Guez, A., Silver, D. | AAAI 2016 | [arXiv:1509.06461](https://arxiv.org/abs/1509.06461) |

**구성 컴포넌트 논문:**
- **Double DQN**: van Hasselt, H., et al. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI 2016.
  - https://arxiv.org/abs/1509.06461
- **Dueling DQN**: Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML 2016.
  - https://arxiv.org/abs/1511.06581
- **Prioritized Experience Replay**: Schaul, T., et al. (2016). *Prioritized Experience Replay*. ICLR 2016.
  - https://arxiv.org/abs/1511.05952
- **Noisy Networks**: Fortunato, M., et al. (2018). *Noisy Networks for Exploration*. ICLR 2018.
  - https://arxiv.org/abs/1706.10295
- **C51 (Distributional RL)**: Bellemare, M.G., et al. (2017). *A Distributional Perspective on Reinforcement Learning*. ICML 2017.
  - https://arxiv.org/abs/1707.06887
- **Maximum Entropy Inverse Reinforcement Learning**. Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). *AAAI Conference on Artificial Intelligence*.

- **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**. (Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*. arXiv:1801.01290. https://arxiv.org/abs/1801.01290)

## Code References
- [**dopamine** (Google)](https://github.com/google/dopamine)
- [**rainbow-is-all-you-need** (PyTorch, 교육용)](https://github.com/Curt-Park/rainbow-is-all-you-need)

# SAC
## Paper
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML 2018.

## Code references
- [**Stable-Baselines3** (Python, 공식)](https://github.com/DLR-RM/stable-baselines3)
- [**spinningup** (PyTorch/TensorFlow, OpenAI)](https://github.com/openai/spinningup)
- [**rlkit** (PyTorch, 연구용)](https://github.com/rail-berkeley/rlkit): Berkeley의 SAC 구현 (원저자 소속)

# PPO
## Paper
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** Proximal Policy Optimization Algorithms. *arXiv:1707.06347*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

## Code References
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

# PPO + ICM
## Paper
- **Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).** Curiosity-driven Exploration by Self-Supervised Prediction. *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*, Vol. 70, pp. 2778–2787. PMLR. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)

## Code References
- [**Stable-Baselines3 Team (2021).**](https://github.com/DLR-RM/stable-baselines3)

# Recurrent PPO
## Paper
- **Hochreiter, S., & Schmidhuber, J.** (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.  
https://doi.org/10.1162/neco.1997.9.8.1735
- **Ni, T., et al.** (2022). *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.* ICML 2022. https://blog.ml.cmu.edu/2022/08/26/recurrent-model-free-rl-can-be-a-strong-baseline-for-many-pomdps-2/

## Code References
- [*Recurrent PPO with Truncated BPTT — Reference Implementation.*](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt) 