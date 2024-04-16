# AI control of tokamak fusion reactor

Forked from [AI Tokamak Control](https://github.com/jaem-seo/AI_tokamak_control)

- KSTAR is a tokamak (donut-shaped nuclear fusion reactor) located in South Korea.
- This repository describes an AI that designs the tokamak operation trajectory to control the fusion plasma in KSTAR.
- Here, we would like to control 3 physics parameters; βp, q95 and li.
- I recommend you to see [KSTAR Tokamak Simulator](https://github.com/jaem-seo/KSTAR_tokamak_simulator) first. The manual control of it is replaced by AI here.

# Installation
- You can install by
```
git clone https://github.com/mht3/AI_tokamak_control.git
cd AI_tokamak_control
```
## Conda Environment

First, create the fusion environment with Python 3.8.
```
conda create -n fusion python=3.8
```

Activate the environment
```
conda activate fusion
```

Install the required packages
```
pip install -r requirements.txt
```

# 1. Target arrival in 4 s interval
- Open the GUI. It takes a bit (tens of secconds) depending on your environment.
```
$ python ai_control_v0.py
```
or
```
$ python ai_control_v1.py
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/166656005-c37156f7-a7a4-4e2c-b714-e0a6319387f7.png">
</p>

- Slide the toggles on the right to change the targets and click the "AI control" button (it takes tens of seconds).
- Then, the AI will design the tokamak operation trajectory to achieve the given target in 4 s.

# 2. Real-time feedback target tracking
- Open the GUI. It takes a bit (tens of secconds) depending on your environment.
```
$ python rt_control_v2.py
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/168571826-9464756c-cd0b-4430-90db-4139d177082c.png">
</p>

- Slide the toggles on the right to change the target state.
- Then, the AI will control the tokamak operation to track the targets in real-time.

# Note
- The AI was trained by reinforcement learning; [TD3](https://arxiv.org/abs/1802.09477) and [HER](https://arxiv.org/abs/1707.01495) implementation from [Stable Baselines](https://github.com/hill-a/stable-baselines).
- The AI control can fail if the target state is physically unfeasible (ex. high-βp, low-q95 and high-li).
- The tokamak simulation possesses most of the computation time, but the AI operation control is actually very fast (real-time capable in experiments).
- Deployment on the KSTAR control system will require further development.

# References
- A. Hill et al. ["Stable Baselines."](https://github.com/hill-a/stable-baselines) GitHub repository (2018).
- S. Fujimoto et al. ["Addressing Function Approximation Error in Actor-Critic Methods."](https://arxiv.org/abs/1802.09477) ICML (2018).
- M. Andrychowicz et al. ["Hindsight Experience Replay."](https://arxiv.org/abs/1707.01495) NIPS (2017).
- J. Seo, ["KSTAR tokamak simulator."](https://github.com/jaem-seo/KSTAR_tokamak_simulator) GitHub repository (2022).
- J. Seo, et al. "Feedforward beta control in the KSTAR tokamak by deep reinforcement learning." Nuclear Fusion [61 (2021): 106010.](https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta)
- J. Seo, et al. "Development of an operation trajectory design algorithm for control of multiple 0D parameters using deep reinforcement learning in KSTAR." Nuclear Fusion [62 (2022): 086049.](https://iopscience.iop.org/article/10.1088/1741-4326/ac79be/meta)
