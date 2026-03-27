import os
import math
import copy
import wandb
import torch
import pickle
import shutil
import argparse
import numpy as np
from pathlib import Path
from importlib import metadata
from dataclasses import asdict, is_dataclass

import matplotlib
matplotlib.use("Agg")   # No Tkinter, only offscreen rendering
import matplotlib.pyplot as plt

import genesis as gs
from genesis import Scene
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

from classes.dodo_configs import *
from classes.file_format_and_paths import FileFormatAndPaths


# ---------------------------------------------------
# Reward Registry
# ---------------------------------------------------
REWARD_REGISTRY = {}

def register_reward():
    """Decorator for the Reward-Methods; the Key is automatically extracted from the reward-function-names."""
    def wrap(fn):
        key = fn.__name__.removeprefix("_reward_")
        REWARD_REGISTRY[key] = fn
        return fn
    return wrap

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class DodoEnvironment:
    def __init__(self, 
                 dodo_path_helper: FileFormatAndPaths,
                 exp_name: str = "dodo-walking",
                 num_envs: int = 4096,
                 max_iterations: int = 2500,
                 ):
        
        
        # -----------------------------------------------------------------------------
        # Public class variables
        # -----------------------------------------------------------------------------
        # set device and initialize path_helper
        self.device = gs.device
        self.dodo_path_helper: FileFormatAndPaths = dodo_path_helper
        self.exp_name: str = exp_name
        self.num_envs: int = num_envs
        self.max_iterations: int = max_iterations

        # extract joint names from path_helper, which extracts them from the robot file. So you don't have to hardcode them in multiple places and can easily change the robot file without worrying about inconsistencies in the joint naming.
        self.joint_names_unmapped = dodo_path_helper.joint_names 
        self.foot_link_names = dodo_path_helper.foot_link_names

        # Set observation space dimensions
        self._base_components = 3 + 3 + 3                              # lin_vel, ang_vel, proj_grav
        self._per_dof_components = 3 * len(self.joint_names_unmapped)  # pos, vel, last_action
        self._command_components = 3                                   # cmd_vel_x, cmd_vel_y, cmd_yaw_rate
        self._clock_components = 0 # TODO set to 2 if you want to use Clock/Phase (sin, cos) -based rewards (like periodic gait reward, bird hip phase reward, etc.)

        self.num_obs = (
            self._base_components
            + self._per_dof_components
            + self._command_components
            + self._clock_components
        )

        # Load the configs from the dataclasses defined in dodo_configs.py. This includes all relevant hyperparameters for the environment, the observations, the rewards, the commands and the training. If you want to change any hyperparameter, you can change it in dodo_configs.py and it will be automatically loaded here. 
        (self.env_config_dataclass,
        self.obs_config_dataclass,
        self.reward_config_dataclass,
        self.command_config_dataclass,
        self.train_config_dataclass) = init_dodo_configs(
            exp_name=self.exp_name,
            foot_link_names=self.foot_link_names,
            joint_names=self.joint_names_unmapped,
            max_iterations=self.max_iterations,
            num_obs=self.num_obs,
            robot_file_format=self.dodo_path_helper.robot_file_format,
            robot_file_path_relative=str(self.dodo_path_helper.robot_file_path_relative),
        )

        # sorted list of joint names (order is "left ...", "right" from top to bottom)
        self.joint_names = list(asdict(self.env_config_dataclass.joint_names_mapped).values()) 
        
        # Pre-compute joint indices (do this ONCE) -> Faster than comnputing it on demand.
        self.idx_left_thigh = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.left_thigh)
        self.idx_right_thigh = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.right_thigh)
        self.idx_left_hip = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.left_hip)
        self.idx_right_hip = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.right_hip)
        self.idx_left_knee = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.left_knee)
        self.idx_right_knee = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.right_knee)

        # Get some data from the configs
        self.num_actions = self.env_config_dataclass.num_actions
        self.num_commands = self.command_config_dataclass.num_commands
        self.simulate_action_latency = self.env_config_dataclass.simulate_action_latency
        self.dt = 0.01
        self.max_episode_length = math.ceil(self.env_config_dataclass.episode_length_s / self.dt)
        self.last_torques = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.obs_scales = self.obs_config_dataclass.obs_scales
        self.reward_scales = self.reward_config_dataclass.reward_scales

        # Initilize relevant variables.
        self.genesis_scene = None
        self.robot = None
        self.motors_dof_idx = None
        self.default_joint_angles = None
        self.kp = None
        self.kd = None

        # -----------------------------------------------------------------------------
        # Global logs (all relevant Reward‑Terms) 
        # -----------------------------------------------------------------------------
        self.iters = []
        self.val_loss = []
        self.surrogate_loss = []
        self.noise_std = []
        self.total_reward = []
        self.ep_length = []
        self.periodic_gait = []
        self.energy_penalty = []
        self.foot_swing_clearance = []
        self.forward_torso_pitch = []
        self.knee_extension_at_push = []
        self.bird_hip_phase = []
        self.hip_abduction_penalty = []
        self.lateral_drift_penalty = []

        # Dataclass -> convert to dict -> iterate over items -> get reward function from registry -> save function and scale in dicts for later use in the reward computation. 
        reward_scales_dict: dict[str, float] = asdict(self.reward_scales)
        self.reward_functions: dict[str, callable] = {}
        self.reward_scales: dict[str, float] = {}
        for name, scale in reward_scales_dict.items():
            if name not in REWARD_REGISTRY:
                raise KeyError(f"Reward '{name}' nicht implementiert.")
            # gebundenes Methoden-Objekt holen
            fn = REWARD_REGISTRY[name].__get__(self, type(self))
            self.reward_functions[name] = fn
            self.reward_scales[name] = scale
        # Episode-Summen pro Reward-Typ
        self.episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros((self.num_envs,), device=self.device)
            for name in self.reward_scales
        }
        
        self.disable_command_resampling = False

    def create_genesis_scene(
            self, 
            show_viewer: bool = False,
            sim_options: gs.options.SimOptions = gs.options.SimOptions(
                dt=0.01, 
                substeps=2,
                gravity=(0, 0, -9.81)
            ),
            viewer_options: gs.options.ViewerOptions = gs.options.ViewerOptions(
                max_FPS=100,
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40
            ),
            rigid_options: gs.options.RigidOptions = gs.options.RigidOptions(
                dt=0.01,
                constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
            ),
            vis_options: gs.options.VisOptions = gs.options.VisOptions(rendered_envs_idx=[0]),
            show_FPS=True,
            ) -> Scene:
        """
        Create a new genesis scene and save it inside the environment object as self.genesis_scene.
        """
        new_scene: Scene = Scene(
            show_viewer=show_viewer,
            sim_options=sim_options,
            viewer_options=viewer_options,
            rigid_options=rigid_options,
            vis_options=vis_options,
            show_FPS=show_FPS,
        )

        return new_scene

    #-------------------------------------------------------------------------------
    # Helper Function for adding different terrains
    #-------------------------------------------------------------------------------
    def _add_ground(self, scene: Scene, terrain_cfg):

        cfg = terrain_cfg

        # Choose terraintype
        if cfg.mode == "random":
            terrain_type = np.random.choice(cfg.options, p=cfg.probs)
        else:
            terrain_type = cfg.mode

        self.current_terrain_type = terrain_type  

        ground_mat = gs.surfaces.Default(color=(0.35, 0.35, 0.35))  # grau

        if terrain_type == "plane":
            scene.add_entity(
                gs.morphs.Plane(),
                surface=ground_mat,
            )
            return

        if terrain_type == "uneven":
            u = cfg.uneven

            n_x, n_y = u.n_subterrains
            sub_size_x, sub_size_y = u.subterrain_size
            c_x, c_y = n_x // 2, n_y // 2

            subterrain_types = []
            for i in range(n_x):
                row = []
                for j in range(n_y):
                    di = abs(i - c_x)
                    dj = abs(j - c_y)
                    if max(di, dj) <= u.spawn_flat_radius_sub:
                        row.append("flat_terrain")
                    else:
                        row.append("random_uniform_terrain")
                subterrain_types.append(row)

            if u.border_flat:
                for k in range(n_y):
                    subterrain_types[0][k] = "flat_terrain"
                    subterrain_types[n_x - 1][k] = "flat_terrain"
                for k in range(n_x):
                    subterrain_types[k][0] = "flat_terrain"
                    subterrain_types[k][n_y - 1] = "flat_terrain"

            total_x = n_x * sub_size_x
            total_y = n_y * sub_size_y
            terrain_pos = (-0.5 * total_x - 0.5 * sub_size_x, -0.5 * total_y - 0.5 * sub_size_y, 0.0)

            scene.add_entity(
                gs.morphs.Terrain(
                    pos=terrain_pos,
                    n_subterrains=u.n_subterrains,
                    subterrain_size=u.subterrain_size,
                    horizontal_scale=u.horizontal_scale,
                    vertical_scale=u.vertical_scale,
                    subterrain_types=subterrain_types,
                    randomize=u.randomize,
                ),
                surface=ground_mat,
            )
            return

        raise ValueError(f"Unknown terrain type: {terrain_type}")


    # -----------------------------------------------------------------------------
    # Import the robot into the scene and hardcode the joint movements
    # -----------------------------------------------------------------------------
    def _init_dodo_scene(self, scene: Scene, spawn_position: tuple[float, float, float], terrain_cfg:TerrainCfg):
        self.genesis_scene = scene

        self.default_joint_angles = list(asdict(self.env_config_dataclass.default_joint_angles).values())

        # add plane
        #self.create_and_add_plane(scene=self.genesis_scene)

        # add uneven terrain
        #self.create_and_add_uneven_terrain(scene=self.genesis_scene)

        self._add_ground(scene=self.genesis_scene, terrain_cfg=terrain_cfg) # Add Ground, that is defined inside the dodo_env_config - either plane or uneven terrain for example

        #Add robot to scene
        if self.dodo_path_helper.robot_file_format == "urdf":
            self.robot = self.genesis_scene.add_entity(
            gs.morphs.URDF(      
                file  = str(os.path.join(self.dodo_path_helper.relevant_paths_dict['urdf'], self.dodo_path_helper.robot_file_name)),
                fixed = False,
                pos   = spawn_position,
                #euler = (0, 0, 270),
                )
            )
        elif self.dodo_path_helper.robot_file_format == "xml":
            self.robot = self.genesis_scene.add_entity(
                gs.morphs.MJCF(
                    file  = str(os.path.join(self.dodo_path_helper.relevant_paths_dict['dodo_robot'], self.dodo_path_helper.robot_file_name)),
                    pos   = spawn_position,
                    #euler = (0, 0, 270),
                )
            )
        else:
            raise Exception("Neither 'URDF' nor 'XML' file was loaded. Therefore No robot is loaded into the simulation")
        
        # build genesis scene after adding all entities.
        self.genesis_scene.build(n_envs=self.num_envs)

        # Get the dofs indices of the motors by their joint names.
        self.motors_dof_idx  = [self.robot.get_joint(n).dof_start for n in self.joint_names]

        # Set the robot to its defined default pose
        self.robot.set_dofs_position(np.array(self.default_joint_angles), self.motors_dof_idx) 

        # Set Controller gains defined in the configs
        self.kp = list(asdict(self.env_config_dataclass.kp).values())
        self.kd = list(asdict(self.env_config_dataclass.kd).values())
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kd, self.motors_dof_idx)

        # Set torque limits defined in the configs
        max_torques = torch.tensor(list(asdict(self.env_config_dataclass.max_torques).values()), dtype=torch.float32, device=self.device)
        self.robot.set_dofs_force_range(
            lower=-max_torques,
            upper=max_torques,
            dofs_idx_local=self.motors_dof_idx,
        )


    # -----------------------------------------------------------------------------
    # Some Helper / Eval Functions
    # -----------------------------------------------------------------------------
    def import_robot_sim(self, manual_stepping: bool = False, total_steps: int = 2000, spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.55)):
        """
        Helper function to test if the robot is correctly imported into the Genesis scene and if the joints can be moved. 
        The joints will move in a sinusoidal pattern, which is defined in this function. 
        You can use this function to debug the robot import and the controller implementation before starting with the training. 
        """
        self.num_envs = 1

        scene = self.create_genesis_scene(show_viewer=True, show_FPS=True)
        self._init_dodo_scene(scene = scene, spawn_position = spawn_position, terrain_cfg=self.env_config_dataclass.terrain_cfg)

        n_dofs    = len(self.motors_dof_idx)
        q_amp  = 0.8
        freq   = 1.3
        omega  = 2 * np.pi * freq
        self.kp = list(asdict(self.env_config_dataclass.kp).values())
        self.kd = list(asdict(self.env_config_dataclass.kd).values())
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kd, self.motors_dof_idx)

        max_torques = torch.tensor(list(asdict(self.env_config_dataclass.max_torques).values()), dtype=torch.float32, device=self.device)
        self.robot.set_dofs_force_range(
            lower=-max_torques,
            upper=max_torques,
            dofs_idx_local=self.motors_dof_idx,
        )

        dt = self.genesis_scene.sim_options.dt

        try:
            for step in range(total_steps):
                t = step * dt
                q_des = q_amp * np.sin(omega * t) * np.ones(n_dofs, dtype=np.float32)

                self.robot.control_dofs_position(q_des, self.motors_dof_idx)
                if manual_stepping:
                    input("enter to continue…")   # keep this to step manually
                base_pos = self.robot.get_pos()
                if manual_stepping:
                    print(f"[pos ctrl] step {step:4d} → base height = {base_pos[0,2]:.4f} m")
                self.genesis_scene.step()
        except gs.GenesisException as e:
            if "Viewer closed" in str(e):
                print("Viewer closed – simulation finished.")
            else:
                raise

    # -----------------------------------------------------------------------------
    def import_robot_standing(self, manual_stepping: bool = False, total_steps: int = 2000, spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.55)):
        """
        Helper function to test if the robot can stand in the Genesis scene with a simple PD controller.
        You can evaluate a stable "standing" init position using this function.
        """
        
        self.num_envs = 1

        scene = self.create_genesis_scene(show_viewer=True, show_FPS=False)
        self._init_dodo_scene(scene=scene, spawn_position=spawn_position, terrain_cfg=self.env_config_dataclass.terrain_cfg)
        self._init_buffers()

        self.kp = list(asdict(self.env_config_dataclass.kp).values())
        self.kd = list(asdict(self.env_config_dataclass.kd).values())
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kd, self.motors_dof_idx)

        max_torques = torch.tensor(
            list(asdict(self.env_config_dataclass.max_torques).values()),
            dtype=torch.float32,
            device=self.device
        )
        self.robot.set_dofs_force_range(
            lower=-max_torques,
            upper=max_torques,
            dofs_idx_local=self.motors_dof_idx,
        )

        foot_link_names = self.env_config_dataclass.foot_link_names
        self.ankle_links = [self.robot.get_link(name) for name in foot_link_names]

        try:
            for step in range(total_steps):
                q_des = self.default_joint_angles
                self.robot.control_dofs_position(q_des, self.motors_dof_idx)

                if manual_stepping:
                    input("enter to continue…")

                self.genesis_scene.step()

                self.current_ankle_heights[:] = torch.stack(
                    [link.get_pos()[:, 2] for link in self.ankle_links],
                    dim=1
                )

                # print ankle heights and contact state
                print(self.current_ankle_heights[0])
                print((self.current_ankle_heights[0] < self.env_config_dataclass.contact_height).float())

        except gs.GenesisException as e:
            if "Viewer closed" in str(e):
                print("Viewer closed – simulation finished.")
            else:
                raise


    # -----------------------------------------------------------------------------
    def test_robot_controller(
        self,
        manual_stepping: bool = False,
        total_steps: int = 1000,
        spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.55),
        kp_value: float = 120.0,
        kd_value: float | None = None,
        q_amp: float = 0.25,
        freq: float = 2.0,
        test_joint_idx: int = 3,
        test_mode: str = "sine",   # "sine" oder "step"
    ):
        """
        Debug / Identification function for testing PD gains on the robot.

        Features:
        - Tests one selected joint around the default pose
        - Logs desired position, actual position, error, velocity and control torque
        - Computes MAE / RMSE / max error / torque saturation ratio
        - Optional plotting
        """
        self.num_envs = 1

        scene = self.create_genesis_scene(show_viewer=True, show_FPS=False)
        self._init_dodo_scene(
            scene=scene,
            spawn_position=spawn_position,
            terrain_cfg=self.env_config_dataclass.terrain_cfg,
        )

        n_dofs = len(self.motors_dof_idx)

        if kd_value is None:
            kd_value = 2.0 * np.sqrt(kp_value)

        kp = kp_value * np.ones(n_dofs, dtype=np.float32)
        kv = kd_value * np.ones(n_dofs, dtype=np.float32)
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kv, self.motors_dof_idx)
        
        max_torques = torch.tensor(list(asdict(self.env_config_dataclass.max_torques).values()), dtype=torch.float32, device=self.device)
        self.robot.set_dofs_force_range(
            lower=-max_torques,
            upper=max_torques,
            dofs_idx_local=self.motors_dof_idx,
        ) 

        dt = float(self.genesis_scene.sim_options.dt)
        omega = 2.0 * np.pi * freq

        if not (0 <= test_joint_idx < n_dofs):
            raise ValueError(
                f"test_joint_idx must be in [0, {n_dofs - 1}], got {test_joint_idx}"
            )

        joint_name = self.joint_names[test_joint_idx]
        default_pose = np.array(self.default_joint_angles, dtype=np.float32)

        print("\n[PD TEST] Starting import_robot_sim")
        print(f"  joint index      : {test_joint_idx}")
        print(f"  joint name       : {joint_name}")
        print(f"  mode             : {test_mode}")
        print(f"  kp               : {kp_value}")
        print(f"  kd               : {kd_value}")
        print(f"  torque_limit     : {torque_limit}")
        print(f"  q_amp            : {q_amp}")
        print(f"  freq             : {freq}")
        print(f"  dt               : {dt}")
        print(f"  default pose     : {default_pose}")

        # --- logging buffers ---
        t_log = []
        q_des_log = []
        q_act_log = []
        q_err_log = []
        qd_act_log = []
        tau_ctrl_log = []
        tau_sat_log = []
        base_height_log = []

        try:
            for step in range(total_steps):
                t = step * dt

                q_des = default_pose.copy()

                if test_mode == "sine":
                    q_des[test_joint_idx] = (
                        default_pose[test_joint_idx] + q_amp * np.sin(omega * t)
                    )

                elif test_mode == "step":
                    # 0-25%: default
                    # 25-50%: +amp
                    # 50-75%: -amp
                    # 75-100%: default
                    phase = step / max(total_steps - 1, 1)

                    if phase < 0.25:
                        delta = 0.0
                    elif phase < 0.50:
                        delta = q_amp
                    elif phase < 0.75:
                        delta = -q_amp
                    else:
                        delta = 0.0

                    q_des[test_joint_idx] = default_pose[test_joint_idx] + delta

                else:
                    raise ValueError("test_mode must be either 'sine' or 'step'")

                self.robot.control_dofs_position(q_des, self.motors_dof_idx)

                if manual_stepping:
                    input("Press Enter to continue...")

                self.genesis_scene.step()

                # --- read back states ---
                q_all = self.robot.get_dofs_position()
                qd_all = self.robot.get_dofs_velocity()

                # try to read control torque and actual torque
                tau_ctrl_all = None
                tau_act_all = None

                try:
                    tau_ctrl_all = self.robot.get_dofs_control_force()
                except Exception:
                    pass

                try:
                    tau_act_all = self.robot.get_dofs_force()
                except Exception:
                    pass

                q_actual = np.array(q_all[0, self.motors_dof_idx].detach().cpu().numpy(), dtype=np.float32)
                qd_actual = np.array(qd_all[0, self.motors_dof_idx].detach().cpu().numpy(), dtype=np.float32)

                if tau_ctrl_all is not None:
                    tau_ctrl = np.array(
                        tau_ctrl_all[0, self.motors_dof_idx].detach().cpu().numpy(),
                        dtype=np.float32,
                    )
                else:
                    tau_ctrl = np.full(n_dofs, np.nan, dtype=np.float32)

                if tau_act_all is not None:
                    tau_act = np.array(
                        tau_act_all[0, self.motors_dof_idx].detach().cpu().numpy(),
                        dtype=np.float32,
                    )
                else:
                    tau_act = np.full(n_dofs, np.nan, dtype=np.float32)

                q_err = q_des - q_actual
                base_pos = self.robot.get_pos()
                base_height = float(base_pos[0, 2].detach().cpu().item())

                torque_limit = max_torques[test_joint_idx].item()

                # saturation flag based on commanded torque
                if np.all(np.isfinite(tau_ctrl)):
                    tau_sat = (np.abs(tau_ctrl) >= 0.98 * torque_limit).astype(np.float32)
                else:
                    tau_sat = np.zeros(n_dofs, dtype=np.float32)

                # --- store logs ---
                t_log.append(t)
                q_des_log.append(q_des.copy())
                q_act_log.append(q_actual.copy())
                q_err_log.append(q_err.copy())
                qd_act_log.append(qd_actual.copy())
                tau_ctrl_log.append(tau_ctrl.copy())
                tau_sat_log.append(tau_sat.copy())
                base_height_log.append(base_height)

                if manual_stepping:
                    print(
                        f"[step {step:4d}] "
                        f"joint={joint_name} | "
                        f"q_des={q_des[test_joint_idx]: .4f} | "
                        f"q_act={q_actual[test_joint_idx]: .4f} | "
                        f"err={q_err[test_joint_idx]: .4f} | "
                        f"qd={qd_actual[test_joint_idx]: .4f} | "
                        f"tau_ctrl={tau_ctrl[test_joint_idx]: .4f} | "
                        f"base_z={base_height: .4f}"
                    )

            # --- convert logs to arrays ---
            t_log = np.asarray(t_log, dtype=np.float32)
            q_des_log = np.asarray(q_des_log, dtype=np.float32)       # (T, n_dofs)
            q_act_log = np.asarray(q_act_log, dtype=np.float32)       # (T, n_dofs)
            q_err_log = np.asarray(q_err_log, dtype=np.float32)       # (T, n_dofs)
            qd_act_log = np.asarray(qd_act_log, dtype=np.float32)     # (T, n_dofs)
            tau_ctrl_log = np.asarray(tau_ctrl_log, dtype=np.float32) # (T, n_dofs)
            tau_sat_log = np.asarray(tau_sat_log, dtype=np.float32)   # (T, n_dofs)
            base_height_log = np.asarray(base_height_log, dtype=np.float32)

            # --- metrics ---
            mae = np.mean(np.abs(q_err_log), axis=0)
            rmse = np.sqrt(np.mean(q_err_log ** 2, axis=0))
            max_err = np.max(np.abs(q_err_log), axis=0)
            mean_speed = np.mean(np.abs(qd_act_log), axis=0)

            if np.all(np.isfinite(tau_ctrl_log)):
                mean_abs_tau = np.mean(np.abs(tau_ctrl_log), axis=0)
                sat_ratio = np.mean(tau_sat_log, axis=0)
            else:
                mean_abs_tau = np.full(n_dofs, np.nan, dtype=np.float32)
                sat_ratio = np.full(n_dofs, np.nan, dtype=np.float32)

            print("\n[PD TEST RESULTS]")
            for i, name in enumerate(self.joint_names):
                print(
                    f"{i:02d} | {name:20s} | "
                    f"MAE={mae[i]:.4f} rad | "
                    f"RMSE={rmse[i]:.4f} rad | "
                    f"MAX={max_err[i]:.4f} rad | "
                    f"|qd|={mean_speed[i]:.4f} rad/s | "
                    f"|tau|={mean_abs_tau[i]:.4f} Nm | "
                    f"sat={sat_ratio[i] * 100.0:.1f}%"
                )

            print(
                f"\n[Test joint summary] {joint_name} | "
                f"MAE={mae[test_joint_idx]:.4f} rad | "
                f"RMSE={rmse[test_joint_idx]:.4f} rad | "
                f"MAX={max_err[test_joint_idx]:.4f} rad | "
                f"sat={sat_ratio[test_joint_idx] * 100.0:.1f}%"
            )

        except gs.GenesisException as e:
            if "Viewer closed" in str(e):
                print("Viewer closed – simulation finished.")
                return None
            raise

    # -----------------------------------------------------------------------------
    def _terrain_cfg_from_dict(self, d: dict) -> TerrainCfg:
        uneven = UnevenTerrainCfg(**d["uneven"])
        return TerrainCfg(
            mode=d["mode"],
            options=d.get("options", []),
            probs=d.get("probs", []),
            uneven=uneven,
        )
    
    # -----------------------------------------------------------------------------
    def _joint_param_to_list(self, x):
        # Fall 1: Dataclass wie DodoJointParams
        if is_dataclass(x):
            return list(asdict(x).values())

        # Fall 2: Dict aus cfgs.pkl
        if isinstance(x, dict):
            return list(x.values())

        # Fall 3: einzelner Skalar
        if isinstance(x, (int, float)):
            return [float(x)] * self.num_actions

        # Fall 4: schon Liste/Tuple/Array
        return list(x)

    # -----------------------------------------------------------------------------
    def eval_trained_model(self, v_x: float = 0.5, v_y: float = 0.0, v_ang: float = 0.0, exp_name: str = "dodo-walking", model_name: str = "model_final.pt"):
        """
        Evaluates an already trained model by loading the saved configs and the model.pt file. The robot should then execute the learned policy in the Genesis scene. 
        You can specify the desired
        Commands v_x, v_y, v_ang.

        IMPORTANT:
        - This function will load the configs from the cfgs.pkl file, which is saved during training. Therefore, the environment will be initialized with the same hyperparameters as during training.
        - self.*_config_dataclass is NOT overwritten.
        """
        self.num_envs = 1

        self._init_buffers()

        self.commands[:] = gs_rand_float(
            self.command_config_dataclass.command_ranges.lin_vel_x[0],
            self.command_config_dataclass.command_ranges.lin_vel_x[1],
            (self.num_envs, self.num_commands),
            self.device,
        )

        self.disable_command_resampling = True # disable command resampling during eval

        # ------------------------------------------------------------------
        # 1. Load configs from pkl.
        # ------------------------------------------------------------------
        root_dir = str(self.dodo_path_helper.relevant_paths_dict["project_root"])
        log_dir = os.path.join(root_dir, "logs", exp_name)

        with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

        if "terrain_cfg" in env_cfg:
            terrain_config_dataclass = self._terrain_cfg_from_dict(env_cfg["terrain_cfg"])
        else:
            terrain_config_dataclass = self.env_config_dataclass.terrain_cfg
            print(
                "[WARN] No terrain_cfg found in saved config "
                "→ using current env_config_dataclass.terrain_cfg"
            )


        # ------------------------------------------------------------------
        # 2. Initialize Scene + Robot with values from env_cfg
        # ------------------------------------------------------------------
        # If a key is missing because of a new hyperparameter (e.g. spawn_position), we fall back to the value in the current env_config_dataclass (which should have a default value, so it won't crash).
        spawn_position = env_cfg.get("base_init_pos", getattr(self.env_config_dataclass, "base_init_pos", [0.0, 0.0, 0.38]))

        scene = self.create_genesis_scene(show_viewer=True, show_FPS=False)
        self._init_dodo_scene(scene=scene, spawn_position=spawn_position, terrain_cfg=terrain_config_dataclass)

        # PD-Gains from env_cfg-Dict (Fallback to Dataclass)
        kp_raw = env_cfg.get("kp", getattr(self.env_config_dataclass, "kp"))
        kd_raw = env_cfg.get("kd", getattr(self.env_config_dataclass, "kd"))
        self.kp = self._joint_param_to_list(kp_raw)
        self.kd = self._joint_param_to_list(kd_raw)

        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kd, self.motors_dof_idx)

        # Torque limits from env_cfg-Dict (Fallback to Dataclass)
        max_torques_raw = env_cfg.get("max_torques", getattr(self.env_config_dataclass, "max_torques"))
        max_torques = torch.tensor(self._joint_param_to_list(max_torques_raw), dtype=torch.float32, device=self.device)

        self.robot.set_dofs_force_range(
            lower=-max_torques,
            upper=max_torques,
            dofs_idx_local=self.motors_dof_idx,
        )

        # ------------------------------------------------------------------
        # 3. Load Link and Joint Information from env_cfg-Dict
        # ------------------------------------------------------------------
        # Foot links
        foot_link_names = env_cfg.get(
            "foot_link_names",
            getattr(self.env_config_dataclass, "foot_link_names", []),
        )
        self.ankle_links = [self.robot.get_link(name) for name in foot_link_names]

        # joint_names_mapped from env_cfg (Dict mit Keys wie "left_hip", "right_hip", ...)
        joint_names_mapped = env_cfg.get(
            "joint_names_mapped",
            {},  # Fallback: empty dict 
        )

        # Helper function to get joint indices based on the mapped names
        def _idx(name_key: str):
            joint_name = joint_names_mapped.get(name_key, None)
            if joint_name is None:
                raise KeyError(f"joint_names_mapped['{name_key}'] fehlt in env_cfg.")
            return self.joint_names.index(joint_name)

        # IMPORTANT: We assume the joint names below. Those are our genesis global joint names definitions (Not the one from the URDF)!
        self.hip_aa_indices = [_idx("left_hip"), _idx("right_hip")]
        self.hip_fe_indices = [_idx("left_thigh"), _idx("right_thigh")]
        self.knee_fe_indices = [_idx("left_knee"), _idx("right_knee")]

        # ------------------------------------------------------------------
        # 4. Initialize Buffers (Observations, Rewards, Dones, Infos, Commands)
        # ------------------------------------------------------------------
        self._init_buffers()

        # ------------------------------------------------------------------
        # 5. Set fixed Commands for evaluation (v_x, v_y, v_ang)
        # ------------------------------------------------------------------
        # Alle Envs bekommen dieselben Kommandos
        self.commands[:, 0] = v_x   # x
        self.commands[:, 1] = v_y   # y
        self.commands[:, 2] = v_ang # yaw

        # ------------------------------------------------------------------
        # 6) PPO-Runner + Load policy from checkpoint
        # ------------------------------------------------------------------
        ckpt = -1
        ckpt_name = f"model_{ckpt}.pt" if ckpt >= 0 else model_name

        runner = OnPolicyRunner(self, train_cfg, log_dir, device=gs.device)
        runner.load(os.path.join(log_dir, ckpt_name))
        policy = runner.get_inference_policy(device=gs.device)

        # ------------------------------------------------------------------
        # 7. Loop the policy and restart on "fall"... Some debugging info is printed, like the current commands, the base velocity and the applied torques. 
        # You can modify this part as you like to print out other information that you find useful for debugging.
        # ------------------------------------------------------------------
        
        obs, _ = self.reset()
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = self.step(actions)

                self.commands[:, 0] = v_x
                self.commands[:, 1] = v_y
                self.commands[:, 2] = v_ang

                # #print ankle heights and contact state for debugging
                # print(self.current_ankle_heights[0]) 
                # print((self.current_ankle_heights[0] < self.CONTACT_HEIGHT).float())

                if self.episode_length_buf[0] % 10 == 0:
                    # print(
                    #     f"Cmd: [{self.commands[0,0]:.2f}, "
                    #     f"{self.commands[0,1]:.2f}, {self.commands[0,2]:.2f}]"
                    # )
                    # print(
                    #     f"Vel: [{self.base_lin_vel[0,0]:.2f}, "
                    #     f"{self.base_lin_vel[0,1]:.2f}, {self.base_ang_vel[0,2]:.2f}]"
                    # )
                    # print(
                    #     f"Applied Torque: [{self.robot.get_dofs_control_force()}"
                    #     f"Actual Torque: [{self.robot.get_dofs_force()}"
                    # )
                    # print(self.base_pos[0, 2], reward_cfg.get("base_height_target", 0.55))
                    pass
    

    def export_checkpoint_to_jit(self, exp_name: str, model_name: str = "model_best.pt"):
        """
        Convert a training checkpoint (.pt) to a TorchScript JIT model for inference.

        The JIT model will be saved next to the checkpoint.

        This export uses the observation dimension stored in the checkpoint,
        so it also works for older policies that were trained with a different
        observation layout (e.g. without clock sin/cos).
        """
        root_dir = str(self.dodo_path_helper.relevant_paths_dict["project_root"])
        log_dir = os.path.join(root_dir, "logs", exp_name)
        checkpoint_path = os.path.join(log_dir, model_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")

        # load configs from pkl
        with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

        # Load the checkpoint directly to read the actual observation dimension from the weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        try:
            ckpt_num_obs = checkpoint["model_state_dict"]["actor.0.weight"].shape[1]
        except KeyError:
            raise KeyError(
                "Could not infer observation dimension from checkpoint. "
                "Expected key 'model_state_dict[\"actor.0.weight\"]'."
            )

        print(f"Checkpoint observation dimension: {ckpt_num_obs}")

        # ------------------------------------------------------------
        # Dummy-Env just for exporting the policy. The actual environment won't be used, but we need to create a compatible runner to load the weights and the normalizer.
        # ------------------------------------------------------------
        class DummyExportEnv:
            def __init__(self, num_obs, num_actions, device):
                self.num_obs = num_obs
                self.num_actions = num_actions
                self.num_envs = 1
                self.device = device

            def get_observations(self):
                obs = torch.zeros((1, self.num_obs), device=self.device)
                return obs, {"observations": {"critic": obs.clone()}}

            def get_privileged_observations(self):
                return None

            def reset(self):
                obs = torch.zeros((1, self.num_obs), device=self.device)
                return obs, {"observations": {"critic": obs.clone()}}

            def step(self, actions):
                obs = torch.zeros((1, self.num_obs), device=self.device)
                rew = torch.zeros((1,), device=self.device)
                done = torch.zeros((1,), dtype=torch.bool, device=self.device)
                infos = {"observations": {"critic": obs.clone()}}
                return obs, rew, done, infos

        dummy_env = DummyExportEnv(
            num_obs=ckpt_num_obs,
            num_actions=self.num_actions,
            device=self.device,
        )

        # Create runner
        runner = OnPolicyRunner(
            env=dummy_env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=self.device,
        )

        # Load weights and normalizer from checkpoint
        runner.load(checkpoint_path)

        policy = runner.alg.actor_critic.actor
        obs_normalizer = runner.obs_normalizer

        policy.eval()
        if obs_normalizer is not None:
            obs_normalizer.eval()

        class PolicyWrapper(torch.nn.Module):
            def __init__(self, policy, obs_norm):
                super().__init__()
                self.policy = policy
                self.obs_norm = obs_norm

            def forward(self, obs):
                if self.obs_norm is not None:
                    obs = self.obs_norm(obs)
                return self.policy(obs)

        model = PolicyWrapper(policy, obs_normalizer).to(self.device)
        model.eval()

        example_obs = torch.zeros(1, ckpt_num_obs, device=self.device)

        print("Tracing model...")
        with torch.no_grad():
            try:
                scripted_model = torch.jit.script(model)
            except Exception:
                scripted_model = torch.jit.trace(model, example_obs)

        jit_path = os.path.splitext(checkpoint_path)[0] + ".jit.pt"
        scripted_model.save(jit_path)

        print(f"JIT model saved to: {jit_path}")


    # -----------------------------------------------------------------------------
    # Logging and plotting of RL training (WANDB + local Matplotlib plots every 100 iterations)
    # -----------------------------------------------------------------------------
    def _wandb_log(self, step, stats):
        # log to the console and W&B
        print(f"[WandB] Iter {step} | reward={stats['episode_reward_mean']:.2f} | loss={stats['value_loss']:.4f}")
        wandb.log(stats, step=step)

    def log_and_plot(self, log_dir, it, stats):
        # 1) Attach data
        self.iters.append(it)
        self.val_loss.append(stats["value_loss"])
        self.surrogate_loss.append(stats["surrogate_loss"])
        self.noise_std.append(stats["action_noise_std"])
        self.total_reward.append(stats["episode_reward_mean"])
        self.ep_length.append(stats["episode_length_mean"])
        self.periodic_gait.append(stats.get("periodic_gait", 0.0))
        self.energy_penalty.append(stats.get("energy_penalty", 0.0))
        self.foot_swing_clearance.append(stats.get("foot_swing_clearance", 0.0))
        self.forward_torso_pitch.append(stats.get("forward_torso_pitch", 0.0))
        self.knee_extension_at_push.append(stats.get("knee_extension_at_push", 0.0))
        self.bird_hip_phase.append(stats.get("bird_hip_phase", 0.0))
        self.hip_abduction_penalty.append(stats.get("hip_abduction_penalty", 0.0))
        self.lateral_drift_penalty.append(stats.get("lateral_drift_penalty", 0.0))

        # 2) Logging to W&B
        self._wandb_log(it, stats)

        # 3) Plot locally every 100 iterations
        if it % 100 == 0:
            fig, axes = plt.subplots(3, 5, figsize=(24, 12))
            axes = axes.flatten()
            metrics = [
                self.val_loss, self.surrogate_loss, self.noise_std,
                self.total_reward, self.ep_length,
                self.periodic_gait, self.energy_penalty, self.foot_swing_clearance,
                self.forward_torso_pitch, self.knee_extension_at_push,
                self.bird_hip_phase, self.hip_abduction_penalty, self.lateral_drift_penalty,
            ]
            titles = [
                "Value Loss", "Surrogate Loss", "Action Noise Std",
                "Mean Total Reward", "Mean Episode Length",
                "Periodic Gait", "Energy Penalty", "Foot Swing Clearance",
                "Forward Torso Pitch", "Knee Ext. at Push",
                "Bird Hip Phase", "Hip Abduction Penalty", "Lateral Drift"
            ]
            for ax, metric, title in zip(axes, metrics, titles):
                ax.plot(self.iters, metric)
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            save_path = os.path.join(log_dir, "metrics.png")
            fig.savefig(save_path)
            wandb.log({"metrics_plot": wandb.Image(save_path)}, step=it)
            plt.close(fig)
            print(f"[Plot] saved to {save_path}")


    # -----------------------------------------------------------------------------
    # Actual Training Loop with PPO using OnPolicyRunner from the GenRL library.
    # -----------------------------------------------------------------------------
    def dodo_train(
            self, 
            resume_from: str | None = None,
            ):
        """
         Main training loop for PPO. 
         We use the OnPolicyRunner from the GenRL library, which handles most of the training logic for us (e.g. collecting experience, computing advantages, updating the policy, etc.). 
         We just need to provide the environment and specify how to log the results.
        """
        
        # Create Genesis Scene and initialize the robot.
        self.genesis_scene = self.create_genesis_scene(show_viewer=False, show_FPS=False)

        # Initialize the environment with the current env_config_dataclass defined in dodo_configs.py
        (env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        train_cfg,) = dataclass_to_dict(
            env_cfg = self.env_config_dataclass,
            obs_cfg = self.obs_config_dataclass,
            reward_cfg = self.reward_config_dataclass,
            command_cfg = self.command_config_dataclass,
            train_cfg = self.train_config_dataclass,
        )

        wandb.init(project="dodo-birdlike-gait", name=self.exp_name)

        wandb.config.update({
            "num_envs": self.num_envs,
            "max_iterations": self.max_iterations,
            "env_cfg": env_cfg,
            "reward_scales": reward_cfg["reward_scales"],
            "obs_cfg": obs_cfg,
            "command_cfg": command_cfg,
            "train_cfg": train_cfg,
        })

        log_dir = Path(self.dodo_path_helper.relevant_paths_dict["project_root"]) / "logs" / self.exp_name

        # Clean old directory
        if log_dir.exists():
            shutil.rmtree(log_dir)

        # Recreate folder
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(log_dir / "cfgs.pkl", "wb") as f:
            pickle.dump([
                env_cfg,
                obs_cfg,
                reward_cfg,
                command_cfg,
                train_cfg,
            ], f)

        # Create a custom runner that inherits from OnPolicyRunner and overrides the save and load functions to handle the normalizer states. 
        # We also add a reference to the outer DodoEnvironment class to access the log_dir for saving checkpoints.
        class CustomRunner(OnPolicyRunner):
            def __init__(self, env, train_cfg, log_dir, device, outer_class: DodoEnvironment):
                super().__init__(env, train_cfg, log_dir, device)
                self.log_dir = log_dir
                self.outer: DodoEnvironment = outer_class

            def save(self, path):
                checkpoint = {
                    "model_state_dict": self.alg.actor_critic.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "iter": self.current_learning_iteration,
                    "infos": getattr(self, "infos", {}),
                }
                if hasattr(self.alg, "lr_scheduler"):
                    checkpoint["scheduler_state_dict"] = self.alg.lr_scheduler.state_dict()

                # Normalizer-States
                if hasattr(self, "obs_normalizer") and self.obs_normalizer is not None:
                    state = self.obs_normalizer.state_dict()
                    checkpoint["obs_normalizer_state"] = state           # dein Name
                    checkpoint["obs_norm_state_dict"] = state            # Name von OnPolicyRunner

                if hasattr(self, "critic_obs_normalizer") and self.critic_obs_normalizer is not None:
                    state = self.critic_obs_normalizer.state_dict()
                    checkpoint["critic_obs_normalizer_state"] = state    # dein Name
                    checkpoint["critic_obs_norm_state_dict"] = state     # Name von OnPolicyRunner

                torch.save(checkpoint, path)
                print(f"[CustomRunner] ✅ Saved checkpoint to {path}")

            def load(self, path):
                checkpoint = torch.load(path, map_location=self.device)

                self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
                self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                if hasattr(self.alg, "lr_scheduler") and "scheduler_state_dict" in checkpoint:
                    self.alg.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                # Beide Varianten akzeptieren (neu bevorzugt)
                if hasattr(self, "obs_normalizer") and self.obs_normalizer is not None:
                    if "obs_norm_state_dict" in checkpoint:
                        self.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
                    elif "obs_normalizer_state" in checkpoint:
                        self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

                if hasattr(self, "critic_obs_normalizer") and self.critic_obs_normalizer is not None:
                    if "critic_obs_norm_state_dict" in checkpoint:
                        self.critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_norm_state_dict"])
                    elif "critic_obs_normalizer_state" in checkpoint:
                        self.critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_normalizer_state"])

                if "infos" in checkpoint:
                    self.infos = checkpoint["infos"]

                self.current_learning_iteration = checkpoint.get("iter", 0)

                print(
                    f"[CustomRunner] 🔁 Loaded checkpoint from {path}, "
                    f"starting at iteration {self.current_learning_iteration}"
                )


            def learn(self, num_learning_iterations, init_at_random_ep_len=False):
                # Reset env and get initial observations
                self.env.reset()
                obs, extras = self.env.get_observations()
                critic_obs = extras["observations"]["critic"].to(self.device)
                obs = obs.to(self.device)
                self.train_mode()

                # ---- Initialize "best model" tracking ----
                best_metric = -1e9
                best_model_path = os.path.join(self.log_dir, "model_best.pt")

                for it in range(self.current_learning_iteration, num_learning_iterations):
                    ep_infos, rewbuffer, lenbuffer = [], [], []

                    # neue Buffer für globale Statistiken
                    stat_buffers = {
                        "fallen_frac": [],
                        "timeout_frac": [],
                        "mean_vx": [],
                    }

                    # -----------------------
                    # Rollouts sammeln
                    # -----------------------
                    for _ in range(self.num_steps_per_env):
                        actions = self.alg.act(obs, critic_obs)

                        # Env step
                        obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                        obs = obs.to(self.device)
                        rewards = rewards.to(self.device)
                        dones = dones.to(self.device)

                        # Normalize
                        obs = self.obs_normalizer(obs)
                        critic_obs = infos["observations"]["critic"].to(self.device)
                        critic_obs = self.critic_obs_normalizer(critic_obs)

                        # PPO-Rollout update
                        self.alg.process_env_step(rewards, dones, infos)

                        # Logging-Collector
                        ep_infos.append(infos["episode"])
                        rewbuffer.append(rewards.mean().item())

                        # get global stats
                        if "stats" in infos and infos["stats"] is not None:
                            s = infos["stats"]
                            for key in stat_buffers.keys():
                                if key in s and s[key] is not None:
                                    val = s[key]
                                    # Tensor -> float
                                    if torch.is_tensor(val):
                                        val = val.item()
                                    stat_buffers[key].append(float(val))

                        # get real episode lengths from infos
                        if "episode_info" in infos and infos["episode_info"] is not None:
                            ep_info = infos["episode_info"]
                            lengths = ep_info.get("length", [])
                            for L in lengths:
                                lenbuffer.append(float(L))

                    # -----------------------
                    # PPO-Update
                    # -----------------------
                    self.alg.compute_returns(critic_obs)
                    mv, ms, *_ = self.alg.update()  # mv = value_loss, ms = surrogate_loss

                    # -----------------------
                    # Build stats dict for logging and plotting
                    # -----------------------
                    if len(lenbuffer) > 0:
                        ep_len_mean = float(np.mean(lenbuffer))
                    else:
                        ep_len_mean = 0.0

                    stats = {
                        "value_loss": mv,
                        "surrogate_loss": ms,
                        "action_noise_std": self.alg.actor_critic.action_std.mean().item(),
                        "episode_reward_mean": float(np.mean(rewbuffer)),
                        "episode_length_mean": ep_len_mean,
                    }

                    # Set all reward scales to 0.0 
                    for name in self.env.reward_scales.keys():
                        stats[name] = 0.0

                    # Calculate mean of each reward component across all episodes in the batch
                    mean_logs = {}
                    for ep in ep_infos:
                        for k, v in ep.items():
                            if k in stats:
                                mean_logs.setdefault(k, []).append(v.mean().cpu().item())
                    for k, v_list in mean_logs.items():
                        stats[k] = float(np.mean(v_list))

                    # global Stats (fallen_frac, timeout_frac, mean_vx)
                    for key, buf in stat_buffers.items():
                        if len(buf) > 0:
                            stats[key] = float(np.mean(buf))
                        else:
                            stats[key] = 0.0

                    # -----------------------
                    # Save best model based on cutom metric
                    # -----------------------
                    reward = stats["episode_reward_mean"]
                    length = stats["episode_length_mean"]
                    vx = stats.get("mean_vx", 0.0)
                    fallen_frac = stats.get("fallen_frac", 0.0)

                    metric = 500.0 * vx + 8.0 * reward + 0.6 * length - 180.0 * fallen_frac # Skalierungsfaktoren, weil vx=0.1 und reward ca. =10 und length ca. =1000 -> FOR WALKING
                    #metric = 10.0 * reward + 1.0 * length - 200.0 * stats.get("fallen_frac", 0.0)  # FOR STANDING


                    if metric > best_metric:
                        best_metric = metric
                        print(f"[CustomRunner] ⭐ New best model at iter {it}: "
                            f"metric={best_metric:.3f} → saving model_best.pt")
                        self.save(best_model_path)

                    # -----------------------
                    # Logging & Plot
                    # -----------------------
                    self.outer.log_and_plot(self.log_dir, it, stats)
                    self.current_learning_iteration = it

        # -----------------------------------------------------------------------------
        # Prepare Genesis Scene
        # -----------------------------------------------------------------------------

        self._add_ground(scene=self.genesis_scene, terrain_cfg=self.env_config_dataclass.terrain_cfg) # Add Ground, that is defined inside the dodo_env_config - either plane or uneven terrain for example

        self.base_init_pos = torch.tensor(self.env_config_dataclass.base_init_pos, device=self.device)
        self.base_init_quat = torch.tensor(self.env_config_dataclass.base_init_quat, device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        init_joint_angles = list(asdict(self.env_config_dataclass.default_joint_angles).values())
        euler_init_position = torch.tensor((0, 0, 0), device=self.device)

        #Add robot to scene
        if self.dodo_path_helper.robot_file_format == "urdf":
            self.robot = self.genesis_scene.add_entity(
            gs.morphs.URDF(      
                file  = str(os.path.join(self.dodo_path_helper.relevant_paths_dict['urdf'], self.dodo_path_helper.robot_file_name)),
                fixed = False,
                pos   = self.base_init_pos.cpu().numpy(),
                quat = self.base_init_quat.cpu().numpy(),
                #euler = euler_init_position.cpu().numpy(),
                )
            )
   
        elif self.dodo_path_helper.robot_file_format == "xml":
            self.robot = self.genesis_scene.add_entity(
                gs.morphs.MJCF(
                    file  = str(os.path.join(self.dodo_path_helper.relevant_paths_dict['dodo_robot'], self.dodo_path_helper.robot_file_name)),
                    pos   = self.base_init_pos.cpu().numpy(),
                    quat = self.base_init_quat.cpu().numpy(),
                    #euler = euler_init_position.cpu().numpy(),
                )
            )

        else:
            raise Exception("Neither 'URDF' nor 'XML' file was loaded. Therefore No robot is loaded into the simulation")
        
        self.genesis_scene.build(n_envs=self.num_envs)

        for name in ["foot_left", "foot_right", "foot_sole_left", "foot_sole_right"]:
            try:
                self.robot.get_link(name)
                print("FOUND:", name)
            except Exception:
                print("MISSING:", name)

        # Set joints and forces after scene.build(), because before build() the dof_indices are not defined yet.
        self.motors_dof_idx = [self.robot.get_joint(n).dof_start for n in self.joint_names]


        # init_joint_angles ist eine Liste mit 8 Werten (in Joint-Reihenfolge)
        single_pose = np.array(init_joint_angles, dtype=np.float32).reshape(1, -1)  # (1, 8)
        all_poses   = np.repeat(single_pose, self.num_envs, axis=0)                 # (num_envs, 8)
        # Set for all envs at the same time
        self.robot.set_dofs_position(
            position=all_poses,
            dofs_idx_local=self.motors_dof_idx,
            # envs_idx=None  -> bedeutet: Form (num_envs, n_dofs) wird für alle Envs benutzt
        )


        self.kp = list(asdict(self.env_config_dataclass.kp).values())
        self.kd = list(asdict(self.env_config_dataclass.kd).values())
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kd, self.motors_dof_idx)

        max_torques = torch.tensor(list(asdict(self.env_config_dataclass.max_torques).values()), dtype=torch.float32, device=self.device)
        self.robot.set_dofs_force_range(
            lower=-max_torques,
            upper=max_torques,
            dofs_idx_local=self.motors_dof_idx,
        )

        #Edited for the use of self.joint_indexes
        self.ankle_links = [self.robot.get_link(n) for n in self.env_config_dataclass.foot_link_names]
        self.hip_aa_indices = [self.idx_left_hip, self.idx_right_hip]
        self.hip_fe_indices = [self.idx_left_thigh, self.idx_right_thigh]
        self.knee_fe_indices = [self.idx_left_knee, self.idx_right_knee]

        # === Initialisiere Beobachtungs- und Aktionsspeicher ===
        self._init_buffers()

        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # Reset env and get initial observations
        self.reset()
        # Create the runner object
        runner = CustomRunner(
            env=self,
            train_cfg=copy.deepcopy(asdict(self.train_config_dataclass)),
            log_dir=log_dir,
            device=gs.device,
            outer_class=self,
        )

        # ========================
        #  Resume-Logic
        # ========================
        if resume_from is not None and os.path.isfile(resume_from):
            # Load pretrained model and normalizer states from checkpoint
            runner.load(resume_from)
            start_it = runner.current_learning_iteration
            # Wenn keine extra_iterations angegeben: einfach nochmal max_iterations drauf
            extra_iterations = self.max_iterations
            total_iters = start_it + extra_iterations
            init_random = False   # beim Fortsetzen nicht wieder mitten in der Episode starten
            print(f"[DodoEnvironment] 🔁 Continuing training from iter {start_it} "
                f"for {extra_iterations} more iterations (up to {total_iters}).")
        else:
            # Fresh training from scratch
            total_iters = self.max_iterations
            init_random = True
            print(f"[DodoEnvironment] 🚀 Fresh training for {total_iters} iterations.")

        # Lernen
        runner.learn(
            num_learning_iterations=total_iters,
            init_at_random_ep_len=init_random,
        )

        # Save final model after training is done
        final_path = os.path.join(log_dir, "model_final.pt")
        runner.save(final_path)
        print(f"=== Trained model saved at {final_path} ===")



    def _init_buffers(self):
        """ Initialize all necessary buffers for observations, rewards, dones, commands, etc."""
        N, A, C = self.num_envs, self.num_actions, self.num_commands
        self.base_lin_vel = torch.zeros((N, 3), device=self.device)
        self.base_ang_vel = torch.zeros((N, 3), device=self.device)
        self.projected_gravity = torch.zeros((N, 3), device=self.device)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N,1)
        self.obs_buf = torch.zeros((N, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros((N,), device=self.device)
        self.reset_buf = torch.zeros((N,), dtype=torch.int32, device=self.device)
        self.episode_length_buf = torch.zeros((N,), dtype=torch.int32, device=self.device)
        self.commands = torch.zeros((N, C), device=self.device)
        self.commands_scale = torch.tensor([
            self.obs_config_dataclass.obs_scales.lin_vel,
            self.obs_config_dataclass.obs_scales.lin_vel,
            self.obs_config_dataclass.obs_scales.ang_vel
        ], device=self.device)
        self.actions = torch.zeros((N, A), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((N,3), device=self.device)
        self.base_quat = torch.zeros((N,4), device=self.device)
        self.base_euler = torch.zeros((N,3), device=self.device)
        self.current_ankle_heights = torch.zeros((N, 2), device=self.device)
        self.prev_contact = torch.zeros((N, 2), device=self.device)
        
        angles = self.env_config_dataclass.default_joint_angles
        names  = self.env_config_dataclass.joint_names_mapped
        joint_to_default = {
            names.left_hip:        angles.left_hip,
            names.right_hip:       angles.right_hip,
            names.left_thigh:      angles.left_thigh,
            names.right_thigh:     angles.right_thigh,
            names.left_knee:       angles.left_knee,
            names.right_knee:      angles.right_knee,
            names.left_foot_ankle: angles.left_foot_ankle,
            names.right_foot_ankle:angles.right_foot_ankle,
        }
        self.default_dof_pos = torch.tensor(
            [joint_to_default[j] for j in self.joint_names],
            device=self.device,
            )
        
        self.extras = {"observations": {}}

    def _resample_commands(self, env_ids):
        """ Resample commands for the given env_ids. This is called during reset to assign new target velocities to the Envs that are being reset."""
        # env_ids: Tensor mit Indizes der Envs, die gerade resampled werden sollen
        low, high = self.command_config_dataclass.command_ranges.lin_vel_x
        self.commands[env_ids,0] = gs_rand_float(low, high, (len(env_ids),), self.device)
        low, high = self.command_config_dataclass.command_ranges.lin_vel_y
        self.commands[env_ids,1] = gs_rand_float(low, high, (len(env_ids),), self.device)
        low, high = self.command_config_dataclass.command_ranges.ang_vel_yaw
        self.commands[env_ids,2] = gs_rand_float(low, high, (len(env_ids),), self.device)

    def _update_robot_state(self):
        """
        GET OBSERVATIONS from simulation:
        Update the robot state from Genesis and store it in the corresponding buffers. 
        This is called after reset and every step to keep the state up to date for reward calculation and observation construction.
        """
        # 1) Torques (currently 0)
        self.last_torques = torch.zeros_like(self.dof_pos)

        # 2) Base-Pos & -Orientation (Quaternions)
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()

        # 3) Base velocities in body coordinates
        inv_q = inv_quat(self.base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_q)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_q)

        # 4) Euler-Winkel (rad)
        self.base_euler[:] = quat_to_xyz(self.base_quat)

        # 5) DOF-Pos & -Vel (only Motor-DOFs)
        self.dof_pos[:] = self.robot.get_dofs_position()[..., self.motors_dof_idx]
        self.dof_vel[:] = self.robot.get_dofs_velocity()[..., self.motors_dof_idx]

        # 6) Ankle Heights (für Floor-contact-Check)
        self.current_ankle_heights[:] = torch.stack(
            [link.get_pos()[:, 2] for link in self.ankle_links],
            dim=1
        )


    def reset_idx(self, env_ids):
        """ Reset function for a subset of environments specified by env_ids. This is called internally when an episode ends (done=True) to reset only the environments that need to be reset, while keeping the others running."""
        if isinstance(env_ids, torch.Tensor):
            env_ids_torch = env_ids.to(device=self.device, dtype=torch.long)
            env_ids_np = env_ids_torch.detach().cpu().numpy()
        else:
            env_ids_np = np.array(env_ids, dtype=np.int64)
            env_ids_torch = torch.from_numpy(env_ids_np).to(self.device)

        # Reset physics
        self.genesis_scene.reset(envs_idx=env_ids_np)

        # Reset Buffer
        self.episode_length_buf[env_ids_torch] = 0

        # New Commands
        self._resample_commands(env_ids_torch)

        # DOF-States in Buffern resetten
        noise = self.env_config_dataclass.init_pose_noise * torch.randn_like(self.dof_pos[env_ids_torch])
        self.dof_pos[env_ids_torch] = self.default_dof_pos.unsqueeze(0) + noise # add small noise to default pose for better exploration after reset
        self.dof_vel[env_ids_torch] = 0.0

        # Set DOF poses in Genesis
        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids_torch].detach().cpu().numpy(),
            dofs_idx_local=self.motors_dof_idx,
            envs_idx=env_ids_np,
            zero_velocity=True,
        )

        # Update Robot State and get observations
        self._update_robot_state()

        self.prev_contact[:] = (self.current_ankle_heights < self.env_config_dataclass.contact_height).float()

        obs, _ = self.get_observations()
        return obs

    def reset(self):
        """ Reset all environments. This is called at the beginning of training to initialize everything, and can also be called manually if needed."""
        self.reset_buf[:] = 0
        self.episode_length_buf[:] = 0
        for key in self.episode_sums:
            self.episode_sums[key].zero_()

        # Reset physics for all Envs
        self.genesis_scene.reset()

        # alle Env-IDs vorbereiten
        all_ids_torch = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        all_ids_np = all_ids_torch.cpu().numpy()

        # DOFs in den Buffern auf Default
        self.dof_pos[:] = self.default_dof_pos + self.env_config_dataclass.init_pose_noise * torch.randn_like(self.dof_pos) # add small noise to default pose for better exploration after reset
        self.dof_vel[:] = 0.0

        # Default-Winkel auch in Genesis setzen (alle Envs)
        self.robot.set_dofs_position(
            position=self.dof_pos.cpu().numpy(),   # Shape: (num_envs, n_dofs)
            dofs_idx_local=self.motors_dof_idx,
            envs_idx=all_ids_np,
            zero_velocity=True,
        )

        # Zustände aus Genesis holen
        self._update_robot_state()

        self.prev_contact[:] = (self.current_ankle_heights < self.env_config_dataclass.contact_height).float()

        # neue Commands ziehen
        self._resample_commands(all_ids_torch)

        # Beobachtungen zurückgeben
        obs, extras = self.get_observations()
        return obs, extras
    

    def step(self, actions):
        """
        This is the core of the environment interaction. 
        It takes a batch of actions for all environments, applies them, steps the simulation, computes rewards and done flags, and returns the new observations, rewards, done flags, and any extra info needed for training and logging.
        """
        # 1) Save last actions and apply new actions
        self.last_actions[:] = self.actions
        self.actions = torch.clip(
            actions,
            -self.env_config_dataclass.clip_actions,
            self.env_config_dataclass.clip_actions
        )
        target = self.actions * self.env_config_dataclass.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target, self.motors_dof_idx)

        # 2) Step the simulation forward
        self.genesis_scene.step()

        # 3) Update robot state from simulation
        self._update_robot_state()

        # 4) Check termination conditions: Timeout vs. Fallen
        timeout = self.episode_length_buf >= self.max_episode_length          # just timeout
        fallen_mask = self._compute_fallen_mask()                             # Real fallen
        done = timeout | fallen_mask                                          # Episode end = Either timeout OR fallen 

        # reset_buf: "Episode just ends"
        self.reset_buf = done

        # 5) Calculate rewards (Survive/Fall nutzen intern nur fallen_mask)
        self.rew_buf[:] = 0.0
        per_step = {}
        for name, fn in self.reward_functions.items():
            r = fn() * self.reward_scales[name]
            self.rew_buf += r
            self.episode_sums[name] += r
            per_step[name] = r

        # 6) Get observations
        obs_buf, obs_extras = self.get_observations()

        # 7) Increment episode length (important: VOR Reset!)
        self.episode_length_buf += 1

        # 8) Stats + Episoden-Infos für Logging
        with torch.no_grad():
            fallen_f = fallen_mask.float()
            timeout_f = timeout.float()

        # Infos zu wirklich beendeten Episoden (für episode_length_mean)
        terminated_ids = done.nonzero(as_tuple=False).flatten()
        episode_info = None
        if terminated_ids.numel() > 0:
            episode_info = {
                # echte Episodenlänge in Schritten
                "length": self.episode_length_buf[terminated_ids].clone().cpu().numpy(),
                # optional: warum beendet
                "timeout": timeout[terminated_ids].clone().cpu().numpy(),
                "fallen": fallen_mask[terminated_ids].clone().cpu().numpy(),
            }

        self.extras = {
            "observations": obs_extras["observations"],
            # per-step Reward-Terms (wie bisher)
            "episode": per_step,
            # globale Statistiken pro Schritt
            "stats": {
                "fallen_frac": fallen_f.mean(),
                "timeout_frac": timeout_f.mean(),
                "mean_vx": self.base_lin_vel[:, 0].mean(),   # <-- dazu
            },
            # neue Info: Episoden, die in DIESEM Schritt zu Ende gingen
            "episode_info": episode_info,
        }

        # 9) Resample commands if neccessary.
        if not self.disable_command_resampling:
            resample_every = int(self.command_config_dataclass.resampling_time_s / self.dt)
            mask = (self.episode_length_buf > 0) & (self.episode_length_buf % resample_every == 0)
            idx = mask.nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                self._resample_commands(idx)

        # 10) Reset Environments that are done
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.reset_idx(done_ids)

        # 11) Debug prints
        if self.episode_length_buf[0] == 1:
            print("[DEBUG] Observation (env 0):", self.obs_buf[0])
            print("[DEBUG] Action      (env 0):", actions[0])
            print("[DEBUG] Reward      (env 0):", self.rew_buf[0])

        # 12) Return results
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def get_observations(self):
        """ 
        Construct the observation vector from the current robot state and commands. 
        This is called after reset and every step to provide the new observations for the policy.
        The resulting vector is basically the input to the neural network!
        """
        base_lin_vel = self.base_lin_vel * self.obs_config_dataclass.obs_scales.lin_vel
        base_ang_vel = self.base_ang_vel * self.obs_config_dataclass.obs_scales.ang_vel
        base_height = self.base_pos[:, 2:3]  # (N,1)
        #dof_pos = self.dof_pos * self.obs_config_dataclass.obs_scales.dof_pos
        dof_pos = (self.dof_pos - self.default_dof_pos) * self.obs_config_dataclass.obs_scales.dof_pos
        dof_vel = self.dof_vel * self.obs_config_dataclass.obs_scales.dof_vel
        joint_torques = self.last_torques
        #last_actions = self.last_actions * self.obs_config_dataclass.obs_scales.dof_pos
        last_actions = self.last_actions
        commands = self.commands * self.commands_scale
        proj_grav = self.projected_gravity  # (N,3)

        # -----------------------------
        # Clock / Phase Observation (sin, cos)
        # -----------------------------
        # Phase aus Episodenzeit (mod period)
        period = float(self.reward_config_dataclass.period)
        t = self.episode_length_buf.float() * self.dt
        phase = torch.remainder(t, period) / period  # [0,1)

        clock_sin = torch.sin(2.0 * math.pi * phase).unsqueeze(-1)  # (N,1)
        clock_cos = torch.cos(2.0 * math.pi * phase).unsqueeze(-1)  # (N,1)
        clock = torch.cat([clock_sin, clock_cos], dim=-1)           # (N,2)

        obs = torch.cat(
            (base_lin_vel, base_ang_vel, proj_grav, dof_pos, dof_vel, last_actions, commands), dim=-1) # TODO add base_height and clock to the observations if you want to use them

        self.obs_buf[:] = obs
        return self.obs_buf, {"observations": {"critic": obs.clone()}}
    
    def get_privileged_observations(self):
        return None
    
    def _compute_fallen_mask(self) -> torch.Tensor:
        """
        Robust fallen detection for bipeds:

        An environment is considered fallen if:
        - The base height is below a certain threshold OR   
        - The roll OR pitch exceeds their respective thresholds.

        Returns:
            A boolean tensor of shape (num_envs,), where True indicates the robot has fallen.
        """

        # Height check
        height = self.base_pos[:, 2]
        too_low = height < self.reward_config_dataclass.base_height_threshold  # z.B. 0.33

        # Orientation check (Roll & Pitch)
        roll = self.base_euler[:, 0]
        pitch = self.base_euler[:, 1]
        roll_thresh = self.reward_config_dataclass.roll_threshold
        pitch_thresh = self.reward_config_dataclass.pitch_threshold
        bad_orientation = (roll.abs() > roll_thresh) | (pitch.abs() > pitch_thresh)

        # fallen if either condition is met
        fallen = too_low | bad_orientation
        return fallen
    

    def _gait_gate(self):
        """
        0 at standstill / very small commands,
        1 at "really walking".
        You can adapt the thresholds (vmin, vmax) to control how "demanding" the gait reward is.
        Thresholds are smart because the robot should not step at standstill but should get rewarded for a nice gait when moving fast enough.
        """
        v = torch.norm(self.commands[:, 0:2], dim=1)
        vmin = 0.05   # darunter: wie "stehen"
        vmax = 0.25   # darüber: voller gait reward
        return torch.clamp((v - vmin) / (vmax - vmin), 0.0, 1.0)
    
    def _abduction_gate(self):
        """
        Gate for abduction stabilization:
        - fully active at vy ~ 0
        - off at |vy| >= vy_max
        """
        vy = torch.abs(self.commands[:, 1])
        vy_max = 0.10   # ab hier: keine Abduction-Einschränkung mehr
        return torch.clamp(1.0 - vy / vy_max, 0.0, 1.0)


    # ---------------------------------------------------
    # Reward Funktionen (überarbeitet)
    # ---------------------------------------------------

    @register_reward()
    def _reward_periodic_gait(self):
        """
        Command conditioned periodicity reward (mismatch form):
        - at v_cmd ~ 0, should NOT walk periodically -> desired_match ~ 0
        - at high v_cmd, should walk periodically -> desired_match ~ 1
        Reward is high when actual_match ~ desired_match.
        """
        # 1) Actual "periodic match" 
        phase = (self.episode_length_buf.float() * self.dt) % self.reward_config_dataclass.period
        half = self.reward_config_dataclass.period * 0.5
        contact = (self.current_ankle_heights < self.env_config_dataclass.contact_height).float()  # (N,2)

        want_left_stance  = (phase < half).float()
        want_right_stance = (phase >= half).float()

        left_match  = want_left_stance  * contact[:, 0] + (1 - want_left_stance)  * (1 - contact[:, 0])
        right_match = want_right_stance * contact[:, 1] + (1 - want_right_stance) * (1 - contact[:, 1])
        match = 0.5 * (left_match + right_match)  # in [0,1]

        # 2) Desired match: 
        desired = self._gait_gate()  # 0..1 abhängig von ||cmd_xy||

        # 3) Mismatch-Penalty (Gauß-Form)
        # Sigma is a hyperparameter that controls how strictly the reward enforces the desired periodicity.
        sigma = 0.25
        err = (match - desired) ** 2
        return torch.exp(-err / (2 * sigma**2 + 1e-8))


    @register_reward()
    def _reward_energy_penalty(self):
        """
        Energy penalty as a Gaussian reward on the change in actions (encouraging smoothness and energy efficiency):
        """
        err = torch.sum((self.actions - self.last_actions)**2, dim=1)
        sigma = self.reward_config_dataclass.energy_sigma 
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_foot_swing_clearance(self):
        """
        Rewarding swing foot clearance relative to the ground level.
        clearance_target is the desired height ABOVE the contact level.
        """
        g = self._gait_gate()                                  # (N,)

        hs = self.current_ankle_heights                        # (N,2)
        contact = (hs < self.env_config_dataclass.contact_height).float()          # (N,2)
        swing_mask = 1.0 - contact                            # (N,2)

        # relative Clearance
        clearance = torch.clamp(hs - self.env_config_dataclass.contact_height, min=0.0)

        target = self.reward_config_dataclass.clearance_target 
        min_clearance = 0.005 
        desired = (min_clearance + g * (target - min_clearance)).unsqueeze(1)

        err = (clearance - desired) ** 2

        sigma = 0.007 # small sigma because we have a clear target and want to strongly encourage matching it. We also have small values
        per_foot = torch.exp(-err / (2 * sigma**2))
        per_foot = per_foot * swing_mask

        num_swing = swing_mask.sum(dim=1).clamp(min=1.0)
        rew = per_foot.sum(dim=1) / num_swing

        no_swing = (swing_mask.sum(dim=1) < 0.5).float()
        rew = rew * (1.0 - no_swing)

        return rew



    @register_reward()
    def _reward_forward_torso_pitch(self):
        """
        Gauß-reward for forward pitch close to a target value.
        """
        pitch = self.base_euler[:, 1]
        err = (pitch - self.reward_config_dataclass.pitch_target)**2  
        sigma = self.reward_config_dataclass.pitch_sigma
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_knee_extension_at_push(self):
        """
        Rewarding knee extension at push-off can encourage a more natural and efficient gait, as it promotes better force transmission and energy transfer during walking. 
        This reward can help the robot learn to extend its knees properly when pushing off the ground, which is crucial for achieving a stable and effective walking pattern.
        """
        gate = self._gait_gate()

        hs = self.current_ankle_heights
        # some foots are in contact if hs < contact_height, so we want to reward knee extension only when we are in the stance phase (not swinging)
        stance = (hs < self.env_config_dataclass.contact_height).any(dim=1).float()

        idx_l = self.idx_left_knee
        idx_r = self.idx_right_knee

        angle_l = self.dof_pos[:, idx_l]
        angle_r = self.dof_pos[:, idx_r]

        max_angle = 1.75  # z.B. ~40 Grad

        ext_l = 1.0 - torch.clamp(torch.abs(angle_l) / max_angle, max=1.0)
        ext_r = 1.0 - torch.clamp(torch.abs(angle_r) / max_angle, max=1.0)

        ext_mean = 0.5 * (ext_l + ext_r)

        return gate * stance * ext_mean


    @register_reward()
    def _reward_tracking_lin_vel(self):
        """
        Gauß-Reward for linear velocity tracking in x/y, based on the squared error between commanded and actual velocity.
        """
        # desired and actual linear velocity in BODY frame
        cmd_xy = self.commands[:, 0:2]        # (N,2)
        vel_xy = self.base_lin_vel[:, 0:2]    # (N,2)

        err = torch.sum((cmd_xy - vel_xy) ** 2, dim=1)
        sigma = self.reward_config_dataclass.tracking_sigma


        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_tracking_ang_vel(self):
        """
        Gauß Reward for angular velocity tracking in z, based on the squared error between commanded and actual yaw rate. 
        """
        err = (self.commands[:, 2] - self.base_ang_vel[:, 2])**2
        sigma = self.reward_config_dataclass.tracking_sigma * 1.5  # evtl. engeres Tracking für Rotation
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_orientation_stability(self):
        """
        Gauß-reward for small roll/pitch deviation, encouraging the robot to maintain a stable upright orientation.
        """
        roll = self.base_euler[:, 0] 
        pitch = self.base_euler[:, 1] 
        err = roll**2 + pitch**2
        sigma = self.reward_config_dataclass.orient_sigma
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_base_height(self):
        """
        Gauß reward for base height close to a target value. 
        This encourages the robot to maintain an optimal height, which can be important for stability and energy efficiency. 
        The reward is high when the base height is close to the target, and decreases as it deviates from the target.
        """
        err = (self.base_pos[:, 2] - self.reward_config_dataclass.base_height_target)**2 
        sigma = self.reward_config_dataclass.height_sigma
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_survive(self):
        """
        Survival reward that increases with episode progress, but only if not fallen.
        - Starts at 0 and goes up to 1 as the episode progresses, but resets to 0 if the robot falls.

        Timeouts are acceptable, falls are not. 
        This encourages the robot to survive as long as possible, but does not give any reward for surviving if it has already fallen. 
        """
        
        fallen = self._compute_fallen_mask().float()  # 1.0 bei Sturz
        alive  = 1.0 - fallen                         # 1.0 solange nicht gefallen

        # normierte Episodenlänge in [0, 1]
        prog = self.episode_length_buf.float() / float(self.max_episode_length)
        prog = torch.clamp(prog, 0.0, 1.0)

        # damit es nicht ganz bei 0 startet: z.B. 0.2..1.0
        per_step = 0.05 + 0.8 * prog

        return alive * per_step # better for walking
        #return 1.0 - fallen # better for standing

    
    @register_reward()
    def _reward_fall_penalty(self):
        """
        Big penalty only for actual falls.
        Timeouts are neutral.
        """
        fallen = self._compute_fallen_mask().float()  # 1.0 bei Sturz
        return -fallen   # mit Scale z.B. 100.0 in deiner Config



    @register_reward()
    def _reward_bird_hip_phase(self):
        """
        Bird-like hip flexion/extension phase driver as a Gaussian reward:
        """
        gate = self._gait_gate()

        idx_l = self.idx_left_thigh
        idx_r = self.idx_right_thigh
        phase = ((self.episode_length_buf.float() * self.dt) % self.reward_config_dataclass.period) / self.reward_config_dataclass.period
        omega = 2 * math.pi * phase

        tgt  = self.reward_config_dataclass.bird_hip_target
        amp  = self.reward_config_dataclass.bird_hip_amp
        desired_l = tgt + amp * torch.sin(omega)
        desired_r = tgt - amp * torch.sin(omega)

        a_l = self.dof_pos[:, idx_l]
        a_r = self.dof_pos[:, idx_r]
        err = (a_l - desired_l)**2 + (a_r - desired_r)**2

        sigma = self.reward_config_dataclass.bird_hip_sigma
        return gate * torch.exp(-err / (2 * sigma**2))

    @register_reward()
    def _reward_hip_abduction_penalty(self):
        """
        Command conditioned Abduction Reward (Mismatch Form):
        - at |vy_cmd| ~ 0: desired_abd ~ 0 (keeping straight)
        - at high |vy_cmd|: desired_abd ~ abd_max (allowing/supporting side-stepping)
        """
        idx_l = self.idx_left_hip
        idx_r = self.idx_right_hip
        abd_l = self.dof_pos[:, idx_l]
        abd_r = self.dof_pos[:, idx_r]

        # 1) Messgröße: "wie viel Abduction nutzt du?"
        abd = 0.5 * (abd_l.abs() + abd_r.abs())  # >=0

        # 2) Desired-Abduction aus |vy_cmd|
        vy = self.commands[:, 1].abs()

        # lineare Rampe 0..vy_max -> 0..abd_max
        vy_max = 0.20    # sollte ungefähr zu deinem command range passen :contentReference[oaicite:3]{index=3}
        abd_max = 0.25   # rad, grob: erlaubte/gewünschte Abduction bei starkem Side-step

        alpha = torch.clamp(vy / (vy_max + 1e-8), 0.0, 1.0)
        desired = alpha * abd_max

        # 3) Reward hoch, wenn abd nahe desired
        sigma = self.reward_config_dataclass.hip_abduction_sigma
        err = (abd - desired) ** 2
        return torch.exp(-err / (2 * sigma**2 + 1e-8))


    @register_reward()
    def _reward_lateral_drift_penalty(self):
        """
        Gauß reward for low lateral drift (y-velocity).
        """
        drift = self.base_lin_vel[:, 1].abs()
        sigma = self.reward_config_dataclass.drift_sigma
        return torch.exp(-drift**2 / (2 * sigma**2))
    
    @register_reward()
    def _reward_vertical_stability(self):
        """
        Gauß reward for low vertical velocity, encouraging stable foot contact and reduced hopping/bouncing.
        """
        v_z = self.base_lin_vel[:, 2]
        sigma = 0.2
        return torch.exp(- (v_z**2) / (2 * sigma**2))
    
    
    @register_reward()
    def _reward_action_rate(self):
        """
        Real penalty instead of Gaussian reward, because we want to strongly discourage big jumps in actions, and a Gaussian would still give a small reward even for large changes.
        Reduces high-frequency action changes that can lead to jerky motions and high energy consumption -> JITTER
        """
        diff = self.actions - self.last_actions
        err = torch.sum(diff**2, dim=1)
        return -err
    
    @register_reward()
    def _reward_step_events(self):
        """
        Command conditioned step event reward:
        - At low commanded velocity, we want few to no steps -> desired_events ~ 0
        - At high commanded velocity, we want regular steps -> desired_events ~ 2*dt/period (because we have 2 steps per period in a nice gait)

        Reward is high when the actual step event rate matches the desired one.
        """
        g = self._gait_gate()
        contact = (self.current_ankle_heights < self.env_config_dataclass.contact_height).float()
        events = torch.clamp(contact - self.prev_contact, min=0.0)
        self.prev_contact[:] = contact
        return g * events.sum(dim=1)   # 0..2