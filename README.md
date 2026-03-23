# 🦤 DODO – Reinforcement Learning for Bipedal Locomotion

This project implements a **bipedal robot (Dodo)** trained with **model-free reinforcement learning (PPO)** using the **Genesis physics engine**.

The goal is to learn stable and efficient locomotion behaviors (e.g. walking or standing) and provide a clean pipeline for:

* simulation
* training
* evaluation
* sim2sim / sim2real transfer


Example of a trained walking policy:

https://github.com/user-attachments/assets/eed10890-74e9-4e20-ada0-640d650bdb05

---

## 🚀 Features

* ⚡ **Fast RL training** using Genesis (GPU accelerated)
* 🤖 **Custom biped robot (URDF / MJCF support)**
* 🧠 **PPO-based training (rsl-rl)**
* 🔁 **Config-driven system (dataclasses)**
* 🦶 Automatic extraction of:

  * joint names
  * foot links
* 📊 Integrated logging with **Weights & Biases**
* Support for:

  * flat terrain
  * uneven terrain
* 🔄 Export to **TorchScript (JIT)** for deployment

---

## 📁 Project Structure

```
.
├── main.py                     # Entry point (train / eval / debug)
├── classes/
│   ├── dodo_environment.py    # Core RL environment + simulation
│   ├── dodo_configs.py        # All training & env configs
│   ├── file_format_and_paths.py # Auto path + URDF parsing
├── robots/
│   └── ...                    # Robot models (URDF / XML)
├── logs/                      # Training outputs & checkpoints
```

---

## ⚙️ Installation

### 1. Clone repository

* Clone repository as usual.

### 2. Install dependencies

Make sure you have:

* Python ≥ 3.8
* CUDA (for GPU training)

Then install:

```bash
pip install -r requirements.txt
```

⚠️ Important:

```bash
pip install rsl-rl-lib==2.2.4
```

---

## ▶️ Usage

### 🧪 1. Test robot (debug simulation)

```bash
python main.py
```

Uncomment in `main.py`:

```python
dodo_env.import_robot_sim()
```

This runs a sinusoidal joint test to verify:

* URDF loading
* joint behavior
* simulation stability

---

### 🧍 2. Test standing controller

```python
dodo_env.import_robot_standing()
```

Useful for:

* PD tuning
* checking spawn pose
* verifying robot stability

---

### 🏋️ 3. Train a policy

```bash
python main.py --num_envs 4096 --max_iterations 500 --exp_name dodo-walking
```

Training uses:

* PPO (rsl-rl)
* parallel environments on GPU
* reward shaping (configurable)

---

### 👀 4. Evaluate trained model

```python
dodo_env.eval_trained_model(
    exp_name="your_experiment",
    v_x=0.2,
    v_y=0.0,
    v_ang=0.0,
)
```

* Opens a viewer
* Runs policy inference
* Shows robot behavior

---

### 💾 5. Export model (for deployment)

```python
dodo_env.export_checkpoint_to_jit(
    exp_name="your_experiment",
    model_name="model_final.pt"
)
```

Used for:

* sim2sim
* sim2real
* C++ integration

---

## Observations

The policy observes:

* base linear velocity
* base angular velocity
* projected gravity
* joint positions
* joint velocities
* last action (model output)
* command velocities
* optional: gait phase (sin/cos)

---

## Reward Design

The reward is composed of multiple terms:

* velocity tracking
* orientation stability
* base height
* periodic gait enforcement
* foot swing clearance
* energy penalty
* joint penalties

All rewards are:

* modular
* configurable via `dodo_configs.py`

---

## Configuration System

All parameters are defined using **dataclasses**:

* `EnvCfg` → robot + simulation
* `RewardCfg` → reward shaping
* `ObsCfg` → observation scaling
* `TrainCfg` → PPO hyperparameters

Configs are:

* automatically saved (`cfgs.pkl`)
* reused during evaluation

---

## Automatic Robot Parsing

The class `FileFormatAndPaths` automatically:

* finds the project root
* locates robot files
* extracts joint names from URDF/XML
* detects foot links

-> No manual mapping needed 

---

## Terrain Support

* Flat plane
* Random uneven terrain
* Configurable via `TerrainCfg`

---

## 📊 Logging

Training logs include:

* reward
* episode length
* losses
* individual reward terms

Stored in:

```
logs/<experiment_name>/
```

Visualization:

* Weights & Biases

---

## Example Commands

```bash
# Train walking
python main.py --num_envs 4096 --max_iterations 400 --exp_name dodo-walking

# Small test run
python main.py --num_envs 512 --max_iterations 50 --exp_name debug

# Continue training
python main.py --exp_name dodo-walking --max_iterations 200
```

---

## Key Design Ideas

* **Model-free RL (PPO)**
* **Phase-based gait learning**
* **Command-conditioned rewards**
* **High parallelization (thousands of envs)**
* **Minimalistic simulation pipeline**

---

## 🔮 Future Work

* sim2real deployment using a ROS2 pipeline
* better domain randomization
* curriculum learning
* more complex terrains
* multi-task policies

---

## Acknowledgements

* Genesis Physics Engine
* rsl-rl (PPO implementation)
