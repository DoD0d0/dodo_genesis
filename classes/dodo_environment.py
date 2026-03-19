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

import matplotlib
matplotlib.use("Agg")   # kein Tkinter, nur offscreen rendering
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
    """Dekorator für Reward-Methoden; der Key wird automatisch aus dem Methodennamen abgeleitet."""
    def wrap(fn):
        key = fn.__name__.removeprefix("_reward_")
        REWARD_REGISTRY[key] = fn
        return fn
    return wrap

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class DodoEnvironment:
    CONTACT_HEIGHT = 0.047 # result in debug for "foot on the ground" was (tensor([0.0426, 0.0425], device='cuda:0'))
    SWING_HEIGHT_THRESHOLD = 0.065 #

    def __init__(self, 
                 dodo_path_helper: FileFormatAndPaths,
                 exp_name: str = "dodo-walking",
                 num_envs: int = 4096,
                 max_iterations: int = 2500,
                 ):
        
        
        # -----------------------------------------------------------------------------
        # Public class variables
        # -----------------------------------------------------------------------------
        self.device = gs.device
        self.dodo_path_helper: FileFormatAndPaths = dodo_path_helper
        self.exp_name: str = exp_name
        self.num_envs: int = num_envs
        self.max_iterations: int = max_iterations

        self.joint_names_unmapped = dodo_path_helper.joint_names # unsorted list of joint names (Order is exactly like the one in the urdf (from top to bottom))
        self.foot_link_names = dodo_path_helper.foot_link_names

        self._base_components = 3 + 3 + 3    # lin_vel, ang_vel, proj_grav
        self._per_dof_components = 3 * len(self.joint_names_unmapped)  # pos, vel, last_action
        self._command_components = 3

        # +2: Clock/Phase (sin, cos) -> damit Periodic-Gait/Phase-Rewards lernbar werden
        self._clock_components = 0 # TODO set to 2 if you want to use clock-based rewards (like periodic gait reward, bird hip phase reward, etc.)

        self.num_obs = (
            self._base_components
            + self._per_dof_components
            + self._command_components
            + self._clock_components
        )

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

        self.joint_names = list(asdict(self.env_config_dataclass.joint_names_mapped).values()) # sorted list of joint names (order is "left ...", "right" from top to bottom)
        
        # Pre-compute joint indices (do this ONCE)
        self.idx_left_thigh = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.left_thigh)
        self.idx_right_thigh = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.right_thigh)
        self.idx_left_hip = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.left_hip)
        self.idx_right_hip = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.right_hip)
        self.idx_left_knee = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.left_knee)
        self.idx_right_knee = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.right_knee)

        self.num_actions = self.env_config_dataclass.num_actions
        self.num_commands = self.command_config_dataclass.num_commands
        self.simulate_action_latency = self.env_config_dataclass.simulate_action_latency
        self.dt = 0.01
        self.max_episode_length = math.ceil(self.env_config_dataclass.episode_length_s / self.dt)
        self.last_torques = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.obs_scales = self.obs_config_dataclass.obs_scales
        self.reward_scales = self.reward_config_dataclass.reward_scales

        self.genesis_scene = None
        self.robot = None
        self.motors_dof_idx = None
        self.default_joint_angles = None
        self.kp = None
        self.kd = None

        # -----------------------------------------------------------------------------
        # Global logs (alle wichtigen Reward‑Terme)
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

        # Dataclass -> Dict konvertieren, damit wir einfach drüber iterieren können
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


        # -----------------------------------------------------------------------------
        # Private class variables
        # -----------------------------------------------------------------------------




        # -----------------------------------------------------------------------------
        # Wandb (All relevant for logging and plotting the training results)
        # -----------------------------------------------------------------------------


    # -----------------------------------------------------------------------------
    # Everything that has to do with the Configs for the RL training
    # -----------------------------------------------------------------------------
    
    

    # -----------------------------------------------------------------------------
    # Create basic genesis scene for the robot
    # -----------------------------------------------------------------------------

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
        Create a new genesis scene and save it inside the object as self.genesis_scene
        """
        new_scene: Scene = Scene(
            show_viewer=show_viewer,
            sim_options=sim_options,
            viewer_options=viewer_options,
            rigid_options=rigid_options,
            vis_options=vis_options,
            show_FPS=show_FPS,
        )

        #self.genesis_scene = new_scene
        return new_scene
    
    # def create_and_add_uneven_terrain(self, scene):
    #     """
    #     Creates a terrain centered around world origin (0,0,0),
    #     with a flat "spawn island" around the center and uneven terrain further out.
    #     """

    #     # -------------------------
    #     # Größe (für ~10s laufen)
    #     # -------------------------
    #     n = 9                  # 9x9 Subterrains
    #     sub_size = 4.0         # 4m pro Subterrain -> 36m x 36m Gesamtfläche
    #     # 10s bei ~0.6 m/s ~ 6m; diagonal ~ 8.5m -> 36m ist sehr entspannt.

    #     horizontal_scale = 0.25  # Meter pro Gridzelle
    #     vertical_scale = 0.008   # Hügelhöhe (klein anfangen)

    #     # -------------------------
    #     # Flache Spawn-Zone
    #     # -------------------------
    #     # radius=1 -> 3x3 Subterrains flach (12m x 12m)
    #     spawn_flat_radius_sub = 0 # 0 is for just one flat subterrain in the center. 1 is for 3x3 flat subterrains etc.
    #     c = n // 2  # Zentrum-Index

    #     subterrain_types = []
    #     for i in range(n):
    #         row = []
    #         for j in range(n):
    #             di = abs(i - c)
    #             dj = abs(j - c)

    #             # Chebyshev-Distanz => quadratische flache Zone
    #             if max(di, dj) <= spawn_flat_radius_sub:
    #                 row.append("flat_terrain")
    #             else:
    #                 # "kleine Hügelchen" weiter draußen
    #                 row.append("random_uniform_terrain")
    #         subterrain_types.append(row)

    #     # Optional: äußersten Rand flach machen (Debug-freundlich; verhindert fiese Randkanten)
    #     for k in range(n):
    #         subterrain_types[0][k] = "flat_terrain"
    #         subterrain_types[n - 1][k] = "flat_terrain"
    #         subterrain_types[k][0] = "flat_terrain"
    #         subterrain_types[k][n - 1] = "flat_terrain"

    #     # -------------------------
    #     # Zentrierung um (0,0)
    #     # -------------------------
    #     total_size = n * sub_size
    #     terrain_pos = (-0.5 * total_size, -0.5 * total_size, 0.0)

    #     terrain = scene.add_entity(
    #         morph=gs.morphs.Terrain(
    #             pos=terrain_pos,  # <-- wichtig: zentriert das Terrain um den Ursprung
    #             n_subterrains=(n, n),
    #             subterrain_size=(sub_size, sub_size),
    #             horizontal_scale=horizontal_scale,
    #             vertical_scale=vertical_scale,
    #             subterrain_types=subterrain_types,
    #             randomize=True,   # optional: Variation (wenn du deterministisch willst -> False)
    #         ),
    #     )

    # def create_and_add_plane(self, scene: Scene):
    #     """
    #     Create a plane in the given genesis scene
    #     """
    #     scene.add_entity(
    #         morph=gs.morphs.Plane(),
    #     )

    #-------------------------------------------------------------------------------
    # Helper Funktion for adding different terrains
    #-------------------------------------------------------------------------------
    def _add_ground(self, scene: Scene, terrain_cfg):

        cfg = terrain_cfg

        # Terrain-Typ wählen
        if cfg.mode == "random":
            terrain_type = np.random.choice(cfg.options, p=cfg.probs)
        else:
            terrain_type = cfg.mode

        self.current_terrain_type = terrain_type  # optional fürs Logging

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
            #jnt_names = ["left_joint_1","right_joint_1","left_joint_2","right_joint_2", "left_joint_3","right_joint_3","left_joint_4","right_joint_4"]
        elif self.dodo_path_helper.robot_file_format == "xml":
            self.robot = self.genesis_scene.add_entity(
                gs.morphs.MJCF(
                    file  = str(os.path.join(self.dodo_path_helper.relevant_paths_dict['dodo_robot'], self.dodo_path_helper.robot_file_name)),
                    pos   = spawn_position,
                    #euler = (0, 0, 270),
                )
            )
            #jnt_names = ["Left_HIP_AA","Right_HIP_AA","Left_THIGH_FE","Right_THIGH_FE", "Left_KNEE_FE","Right_SHIN_FE","Left_FOOT_ANKLE","Right_FOOT_ANKLE"]
        else:
            raise Exception("Neither 'URDF' nor 'XML' file was loaded. Therefore No robot is loaded into the simulation")
        
        self.genesis_scene.build(n_envs=self.num_envs)

        self.motors_dof_idx  = [self.robot.get_joint(n).dof_start for n in self.joint_names]

        self.robot.set_dofs_position(np.array(self.default_joint_angles), self.motors_dof_idx) 

        self.kp = [self.env_config_dataclass.kp] * self.num_actions
        self.kd = [self.env_config_dataclass.kd] * self.num_actions
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kd, self.motors_dof_idx)

        max_force = 7.0 # Newtonmeter
        
        self.robot.set_dofs_force_range(
            lower=-max_force * torch.ones(self.num_actions, dtype=torch.float32),
            upper= max_force * torch.ones(self.num_actions, dtype=torch.float32),
            dofs_idx_local=self.motors_dof_idx,
        )


    def import_robot_sim(self, manual_stepping: bool = False, total_steps: int = 2000, spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.55)):
        self.num_envs = 1

        scene = self.create_genesis_scene(show_viewer=True, show_FPS=True)
        self._init_dodo_scene(scene = scene, spawn_position = spawn_position, terrain_cfg=self.env_config_dataclass.terrain_cfg)

        n_dofs    = len(self.motors_dof_idx)
        q_amp  = 0.8
        freq   = 1.3
        omega  = 2 * np.pi * freq
        kp     = 50.0  * np.ones(n_dofs, dtype=np.float32)
        kv     = 2.0 * np.sqrt(kp) 
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kv, self.motors_dof_idx)

        self.robot.set_dofs_force_range(
            lower = -3.0*np.ones(n_dofs, dtype=np.float32),
            upper =  3.0*np.ones(n_dofs, dtype=np.float32),
            dofs_idx_local = self.motors_dof_idx,
        )

        dt = self.genesis_scene.sim_options.dt

        try:
            for step in range(total_steps):
                t = step * dt
                q_des = q_amp * np.sin(omega * t) * np.ones(n_dofs, dtype=np.float32)

                # q_des[1::2] *= -1    # alle ungeraden Indizes negieren
                # q_des[0] = 0
                # q_des[1] = 0


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
    # Import the robot into the scene and hardcode the joint movements
    # -----------------------------------------------------------------------------

    def import_robot_standing(self, manual_stepping: bool = False, total_steps: int = 2000, spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.55)):
        self.num_envs = 1

        scene = self.create_genesis_scene(show_viewer=True, show_FPS=False)
        self._init_dodo_scene(scene = scene, spawn_position = spawn_position, terrain_cfg=self.env_config_dataclass.terrain_cfg)
        
        n_dofs    = len(self.motors_dof_idx)
        q_amp  = 0.8
        freq   = 1.3
        omega  = 2 * np.pi * freq
        kp     = 120.0  * np.ones(n_dofs, dtype=np.float32)
        kv     = 2.0*np.sqrt(kp) 
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kv, self.motors_dof_idx)

        self.robot.set_dofs_force_range(
            lower = -3.0*np.ones(n_dofs, dtype=np.float32),
            upper =  3.0*np.ones(n_dofs, dtype=np.float32),
            dofs_idx_local = self.motors_dof_idx,
        )
        

        try:
            for step in range(total_steps):
                q_des = self.default_joint_angles

                self.robot.control_dofs_position(q_des, self.motors_dof_idx)
                if manual_stepping:
                    input("enter to continue…")   # keep this to step manually
                base_pos = self.robot.get_pos()
                if manual_stepping:
                    print(f"[pos ctrl] step {step:4d} → base height = {base_pos[0,2]:.4f} m")
                self.genesis_scene.step()
                #print(self.robot.get_pos()[0,2])
        except gs.GenesisException as e:
            if "Viewer closed" in str(e):
                print("Viewer closed – simulation finished.")
            else:
                raise

    def test_robot_controller(
        self,
        manual_stepping: bool = False,
        total_steps: int = 1000,
        spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.55),
        kp_value: float = 120.0,
        kd_value: float | None = None,
        torque_limit: float = 5.0,
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

        Parameters
        ----------
        manual_stepping : bool
            If True, waits for Enter each step.
        total_steps : int
            Number of simulation steps.
        spawn_position : tuple
            Robot spawn position.
        kp_value : float
            Positional gain for all actuated DOFs.
        kd_value : float | None
            Velocity gain for all actuated DOFs. If None, uses 2*sqrt(kp_value).
        torque_limit : float
            Symmetric force / torque limit in Nm.
        q_amp : float
            Amplitude of the test motion in rad.
        freq : float
            Frequency for sine motion in Hz.
        test_joint_idx : int
            Local test joint index in self.joint_names.
        test_mode : str
            "sine" for sinusoidal target, "step" for step target.
        plot_results : bool
            If True, plots the logs at the end.
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
        self.robot.set_dofs_force_range(
            lower=-torque_limit * np.ones(n_dofs, dtype=np.float32),
            upper=torque_limit * np.ones(n_dofs, dtype=np.float32),
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

    def _terrain_cfg_from_dict(self, d: dict) -> TerrainCfg:
        uneven = UnevenTerrainCfg(**d["uneven"])
        return TerrainCfg(
            mode=d["mode"],
            options=d.get("options", []),
            probs=d.get("probs", []),
            uneven=uneven,
        )

    def eval_trained_model(self, v_x: float = 0.5, v_y: float = 0.0, v_ang: float = 0.0, exp_name: str = "dodo-walking", model_name: str = "model_final.pt"):
        """
        Evaluiert ein trainiertes Modell (Logs unter logs/exp_name) mit festen
        Kommandos v_x, v_y, v_ang.

        WICHTIG:
        - Es werden die in cfgs.pkl gespeicherten Configs als DICTIONARIES benutzt.
        - self.*_config_dataclass wird NICHT überschrieben.
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
        # 1) Config-Dicts aus Pickle laden
        # ------------------------------------------------------------------
        root_dir = str(self.dodo_path_helper.relevant_paths_dict["project_root"])
        log_dir = os.path.join(root_dir, "logs", exp_name)

        with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
            # Annahme: das sind bereits Dicts; falls nicht, kannst du hier noch
            # env_cfg = vars(env_cfg) o.Ä. machen.

        if "terrain_cfg" in env_cfg:
            terrain_config_dataclass = self._terrain_cfg_from_dict(env_cfg["terrain_cfg"])
        else:
            terrain_config_dataclass = self.env_config_dataclass.terrain_cfg
            print(
                "[WARN] No terrain_cfg found in saved config "
                "→ using current env_config_dataclass.terrain_cfg"
            )


        # ------------------------------------------------------------------
        # 2) Szene + Roboter mit Werten aus env_cfg initialisieren
        # ------------------------------------------------------------------
        # base_init_pos aus env_cfg-Dict (Fallback auf aktuelle Dataclass falls Key fehlt)
        spawn_position = env_cfg.get("base_init_pos", getattr(self.env_config_dataclass, "base_init_pos", [0.0, 0.0, 0.55]))

        scene = self.create_genesis_scene(show_viewer=True, show_FPS=False)
        self._init_dodo_scene(scene=scene, spawn_position=spawn_position, terrain_cfg=terrain_config_dataclass)

        # PD-Gains aus env_cfg-Dict (Fallback auf Dataclass)
        kp = env_cfg.get("kp", getattr(self.env_config_dataclass, "kp", 175.0))
        kd = env_cfg.get("kd", getattr(self.env_config_dataclass, "kd", 2.0 * np.sqrt(175.0)))

        self.kp = [kp] * self.num_actions
        self.kd = [kd] * self.num_actions
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kd, self.motors_dof_idx)

        # Kraftgrenzen / Torque-Limit aus env_cfg (z.B. clip_actions), sonst Fallback
        torque_limit = 7.0 #Newtonmeter
        self.robot.set_dofs_force_range(
            lower=- torque_limit * torch.ones(self.num_actions, dtype=torch.float32, device=self.device),
            upper=  torque_limit * torch.ones(self.num_actions, dtype=torch.float32, device=self.device),
            dofs_idx_local=self.motors_dof_idx,
        )

        # ------------------------------------------------------------------
        # 3) Link- und Joint-Informationen aus env_cfg-Dict holen
        # ------------------------------------------------------------------
        # Fuß-Links für Rewards (wenn vorhanden)
        foot_link_names = env_cfg.get(
            "foot_link_names",
            getattr(self.env_config_dataclass, "foot_link_names", []),
        )
        self.ankle_links = [self.robot.get_link(name) for name in foot_link_names]

        # joint_names_mapped aus env_cfg (Dict mit Keys wie "left_hip", "right_hip", ...)
        joint_names_mapped = env_cfg.get(
            "joint_names_mapped",
            {},  # Fallback: leeres Dict
        )

        # self.joint_names kommt bei dir aus der URDF-Parsing-Logik
        # (die lassen wir in Ruhe) – wir nutzen hier NUR die Mapping-Infos
        def _idx(name_key: str):
            joint_name = joint_names_mapped.get(name_key, None)
            if joint_name is None:
                raise KeyError(f"joint_names_mapped['{name_key}'] fehlt in env_cfg.")
            return self.joint_names.index(joint_name)

        # ACHTUNG: hier nehmen wir an, dass die Keys so heißen wie bisher:
        #   "left_hip", "right_hip", "left_thigh", "right_thigh", "left_knee", "right_knee"
        self.hip_aa_indices = [_idx("left_hip"), _idx("right_hip")]
        self.hip_fe_indices = [_idx("left_thigh"), _idx("right_thigh")]
        self.knee_fe_indices = [_idx("left_knee"), _idx("right_knee")]

        # ------------------------------------------------------------------
        # 4) Buffer initialisieren
        # ------------------------------------------------------------------
        self._init_buffers()

        # ------------------------------------------------------------------
        # 5) Kommandos setzen (hier verwenden wir NICHT die Dataclass,
        #    sondern direkt v_x, v_y, v_ang; command_cfg brauchen wir
        #    nur, wenn du später z.B. ranges für Noise etc. nutzen willst)
        # ------------------------------------------------------------------
        # Alle Envs bekommen dieselben Kommandos
        self.commands[:, 0] = v_x   # x
        self.commands[:, 1] = v_y   # y
        self.commands[:, 2] = v_ang # yaw

        # Wenn du trotzdem die geladenen Ranges aus command_cfg nutzen willst, z.B.:
        # cmd_ranges = command_cfg.get("command_ranges", {})
        # lin_vel_x_range = cmd_ranges.get("lin_vel_x", [v_x, v_x])
        # usw. – aktuell machen wir aber fixe Kommandos.

        # ------------------------------------------------------------------
        # 6) PPO-Runner + Policy aus Checkpoint laden
        # ------------------------------------------------------------------
        ckpt = -1
        ckpt_name = f"model_{ckpt}.pt" if ckpt >= 0 else model_name

        runner = OnPolicyRunner(self, train_cfg, log_dir, device=gs.device)
        runner.load(os.path.join(log_dir, ckpt_name))
        policy = runner.get_inference_policy(device=gs.device)

        # ------------------------------------------------------------------
        # 7) Schleife: Policy laufen lassen & ein bisschen Debug-Ausgabe
        # ------------------------------------------------------------------
        
        obs, _ = self.reset()
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = self.step(actions)

                # Wenn du sicherstellen willst, dass Commands NIE resampled werden
                # (falls deine step-Logik Commands updatet), kannst du hier
                # jedes Mal wieder setzen:
                self.commands[:, 0] = v_x
                self.commands[:, 1] = v_y
                self.commands[:, 2] = v_ang

                #print ankle heights and contact state for debugging
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
        #self.device = gs.cpu
        root_dir = str(self.dodo_path_helper.relevant_paths_dict["project_root"])
        log_dir = os.path.join(root_dir, "logs", exp_name)
        checkpoint_path = os.path.join(log_dir, model_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")

        # configs laden
        with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

        # Checkpoint direkt laden, um die echte Obs-Dimension aus den Gewichten zu lesen
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
        # Dummy-Env nur für den Runner-Aufbau
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

        # Runner auf Basis der Checkpoint-Architektur erzeugen
        runner = OnPolicyRunner(
            env=dummy_env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=self.device,
        )

        # Gewichte + Normalizer laden
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
    # Logging and plotting of RL training
    # -----------------------------------------------------------------------------

    def _wandb_log(self, step, stats):
        # log to the console and W&B
        print(f"[WandB] Iter {step} | reward={stats['episode_reward_mean']:.2f} | loss={stats['value_loss']:.4f}")
        wandb.log(stats, step=step)

    def log_and_plot(self, log_dir, it, stats):
        # 1) Daten anhängen
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

        # 2) Logging an W&B
        self._wandb_log(it, stats)

        # 3) Alle 100 Iterationen lokal plotten
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
    # Logging and plotting of RL training
    # -----------------------------------------------------------------------------
    def dodo_train(
            self, 
            resume_from: str | None = None,
            ):
        

        self.genesis_scene = self.create_genesis_scene(show_viewer=False, show_FPS=False)

        # env_cfg = self.dataclass_to_dict(dataclass_object=self.env_config_dataclass)
        # train_cfg = self.dataclass_to_dict(dataclass_object=self.train_config_dataclass)
        # reward_cfg = self.dataclass_to_dict(dataclass_object=self.reward_config_dataclass)
        # obs_cfg = self.dataclass_to_dict(dataclass_object=self.obs_config_dataclass)
        # command_cfg = self.dataclass_to_dict(dataclass_object=self.command_config_dataclass)

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

                # Normalizer-States: beide Key-Varianten speichern
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
                # Env einmal resetten und erste Beobachtungen holen
                self.env.reset()
                obs, extras = self.env.get_observations()
                critic_obs = extras["observations"]["critic"].to(self.device)
                obs = obs.to(self.device)
                self.train_mode()

                # ---- Best-Model-Tracking initialisieren ----
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

                        # Schritt in der Env
                        obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                        obs = obs.to(self.device)
                        rewards = rewards.to(self.device)
                        dones = dones.to(self.device)

                        # Normalisierung
                        obs = self.obs_normalizer(obs)
                        critic_obs = infos["observations"]["critic"].to(self.device)
                        critic_obs = self.critic_obs_normalizer(critic_obs)

                        # PPO-Rollout updaten
                        self.alg.process_env_step(rewards, dones, infos)

                        # Logging-Sammler
                        ep_infos.append(infos["episode"])
                        rewbuffer.append(rewards.mean().item())

                        # globale Statistiken einsammeln
                        if "stats" in infos and infos["stats"] is not None:
                            s = infos["stats"]
                            for key in stat_buffers.keys():
                                if key in s and s[key] is not None:
                                    val = s[key]
                                    # Tensor -> float
                                    if torch.is_tensor(val):
                                        val = val.item()
                                    stat_buffers[key].append(float(val))

                        # Echte Episodenlängen aus den Infos ziehen
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
                    # Statistiken bauen
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

                    # Alle Reward-Terms erstmal auf 0 setzen
                    for name in self.env.reward_scales.keys():
                        stats[name] = 0.0

                    # Mittlere Rewards über alle Episoden berechnen
                    mean_logs = {}
                    for ep in ep_infos:
                        for k, v in ep.items():
                            if k in stats:
                                mean_logs.setdefault(k, []).append(v.mean().cpu().item())
                    for k, v_list in mean_logs.items():
                        stats[k] = float(np.mean(v_list))

                    # globale Statistiken (fallen_frac, timeout_frac, mean_vx)
                    for key, buf in stat_buffers.items():
                        if len(buf) > 0:
                            stats[key] = float(np.mean(buf))
                        else:
                            stats[key] = 0.0

                    # -----------------------
                    # Bestes Modell speichern
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
        #self.genesis_scene.add_entity(gs.morphs.Plane(fixed=True))

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

        # === Nach scene.build(): Gelenke und Kräfte setzen ===
        self.motors_dof_idx = [self.robot.get_joint(n).dof_start for n in self.joint_names]


        # init_joint_angles ist eine Liste mit 8 Werten (in Joint-Reihenfolge)
        single_pose = np.array(init_joint_angles, dtype=np.float32).reshape(1, -1)  # (1, 8)
        all_poses   = np.repeat(single_pose, self.num_envs, axis=0)                 # (num_envs, 8)
        # Jetzt für alle Envs gleichzeitig setzen
        self.robot.set_dofs_position(
            position=all_poses,
            dofs_idx_local=self.motors_dof_idx,
            # envs_idx=None  -> bedeutet: Form (num_envs, n_dofs) wird für alle Envs benutzt
        )


        kp = [self.env_config_dataclass.kp] * self.num_actions
        kd = [self.env_config_dataclass.kd] * self.num_actions
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd, self.motors_dof_idx)

        torque_limit = 7.0  # oder was in deiner Hand-Demo gut funktioniert
        self.robot.set_dofs_force_range(
            lower=- torque_limit * torch.ones(self.num_actions, dtype=torch.float32),
            upper= torque_limit * torch.ones(self.num_actions, dtype=torch.float32),
            dofs_idx_local=self.motors_dof_idx,
        )

        #Edited for the use of self.joint_indexes
        self.ankle_links = [self.robot.get_link(n) for n in self.env_config_dataclass.foot_link_names]
        self.hip_aa_indices = [self.idx_left_hip, self.idx_right_hip]
        self.hip_fe_indices = [self.idx_left_thigh, self.idx_right_thigh]
        self.knee_fe_indices = [self.idx_left_knee, self.idx_right_knee]

        # === Initialisiere Beobachtungs- und Aktionsspeicher ===
        self._init_buffers()

        # self.commands[:] = gs_rand_float(
        #     self.command_config_dataclass.command_ranges.lin_vel_x[0],
        #     self.command_config_dataclass.command_ranges.lin_vel_x[1],
        #     (self.num_envs, self.num_commands),
        #     self.device,
        # )

        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        
        # #######################################
        # # Train loop -> Outdated version with the 4 different goal speeds.
        # # Stage‑Loop: nur Commands anpassen, Env wiederverwenden
        # cumulative_iter = 0
        # for i, v in enumerate([0.1, 0.3, 0.4, 0.5], start=1):
        #     iters_stage = int(self.max_iterations * (0.2 if i < 4 else 0.4))
        #     print(f"=== Stage {i}: Zielgeschwindigkeit {v:.1f} m/s ===")
        #     self.command_config_dataclass.command_ranges.lin_vel_x = [v, v] 
        #     self.reset()
        #     runner = CustomRunner(
        #         env=self,
        #         train_cfg=copy.deepcopy(asdict(self.train_config_dataclass)),
        #         log_dir=log_dir,
        #         device=gs.device,
        #         outer_class=self,
        #     )
        #     runner.current_learning_iteration = cumulative_iter
        #     runner.learn(
        #         num_learning_iterations=cumulative_iter + iters_stage,
        #         init_at_random_ep_len=(i == 1)
        #     )
        #     fname = f"model_stage{i}.pt" if i < 4 else "model_final.pt"
        #     runner.save(os.path.join(log_dir, fname))
        #     cumulative_iter += iters_stage

        # print(f"=== Trained model saved at {log_dir}/model_final.pt ===")

        #######################################
        # Train loop – einfache Version mit nur einer Zielgeschwindigkeit
        # fixe Zielgeschwindigkeit für geradeaus laufen

        # self.command_config_dataclass.command_ranges.lin_vel_x = [0.1, 0.5]
        # self.command_config_dataclass.command_ranges.lin_vel_y = [0.0, 0.0]
        # self.command_config_dataclass.command_ranges.ang_vel_yaw = [0.0, 0.0]

        # Env einmal sauber resetten (inkl. neuem Command)
        self.reset()
        # Ein Runner für das ganze Training
        runner = CustomRunner(
            env=self,
            train_cfg=copy.deepcopy(asdict(self.train_config_dataclass)),
            log_dir=log_dir,
            device=gs.device,
            outer_class=self,
        )

        # ========================
        #  Resume-Logik
        # ========================
        if resume_from is not None and os.path.isfile(resume_from):
            # bereits trainiertes Modell/Optimizer laden
            runner.load(resume_from)
            start_it = runner.current_learning_iteration
            # Wenn keine extra_iterations angegeben: einfach nochmal max_iterations drauf
            extra_iterations = self.max_iterations
            total_iters = start_it + extra_iterations
            init_random = False   # beim Fortsetzen nicht wieder mitten in der Episode starten
            print(f"[DodoEnvironment] 🔁 Continuing training from iter {start_it} "
                f"for {extra_iterations} more iterations (up to {total_iters}).")
        else:
            # Frisches Training
            total_iters = self.max_iterations
            init_random = True
            print(f"[DodoEnvironment] 🚀 Fresh training for {total_iters} iterations.")

        # Lernen
        runner.learn(
            num_learning_iterations=total_iters,
            init_at_random_ep_len=init_random,
        )

        # Finale Gewichte abspeichern (neuer Name, damit du sie unterscheiden kannst)
        final_path = os.path.join(log_dir, "model_final.pt")
        runner.save(final_path)
        print(f"=== Trained model saved at {final_path} ===")



    def _init_buffers(self):
        #env_cfg = self.dataclass_to_dict(self.env_config_dataclass)
        #env_cfg = dataclass_to_dict(env_cfg = self.env_config_dataclass)
        N, A, C = self.num_envs, self.num_actions, self.num_commands
        self.base_lin_vel = torch.zeros((N, 3), device=self.device)
        self.base_ang_vel = torch.zeros((N, 3), device=self.device)
        self.projected_gravity = torch.zeros((N, 3), device=self.device)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N,1)
        self.obs_buf = torch.zeros((N, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros((N,), device=self.device)
        #self.reset_buf = torch.ones((N,), dtype=torch.int32, device=self.device)
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
        # env_ids: Tensor mit Indizes der Envs, die gerade resampled werden sollen
        low, high = self.command_config_dataclass.command_ranges.lin_vel_x
        self.commands[env_ids,0] = gs_rand_float(low, high, (len(env_ids),), self.device)
        low, high = self.command_config_dataclass.command_ranges.lin_vel_y
        self.commands[env_ids,1] = gs_rand_float(low, high, (len(env_ids),), self.device)
        low, high = self.command_config_dataclass.command_ranges.ang_vel_yaw
        self.commands[env_ids,2] = gs_rand_float(low, high, (len(env_ids),), self.device)

    def _update_robot_state(self):
        # 1) Torques (aktuell noch 0)
        self.last_torques = torch.zeros_like(self.dof_pos)

        # 2) Basis-Pos & -Orientierung
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()

        # 3) Basis-Geschwindigkeiten in Körperkoordinaten
        inv_q = inv_quat(self.base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_q)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_q)

        # 4) Euler-Winkel (rad)
        self.base_euler[:] = quat_to_xyz(self.base_quat)

        # 5) DOF-Pos & -Vel (nur Motor-DOFs)
        self.dof_pos[:] = self.robot.get_dofs_position()[..., self.motors_dof_idx]
        self.dof_vel[:] = self.robot.get_dofs_velocity()[..., self.motors_dof_idx]

        # 6) Knöchel-Höhen
        self.current_ankle_heights[:] = torch.stack(
            [link.get_pos()[:, 2] for link in self.ankle_links],
            dim=1
        )


    def reset_idx(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            env_ids_torch = env_ids.to(device=self.device, dtype=torch.long)
            env_ids_np = env_ids_torch.detach().cpu().numpy()
        else:
            env_ids_np = np.array(env_ids, dtype=np.int64)
            env_ids_torch = torch.from_numpy(env_ids_np).to(self.device)

        # Physik resetten
        self.genesis_scene.reset(envs_idx=env_ids_np)

        # Buffer zurücksetzen
        self.episode_length_buf[env_ids_torch] = 0
        # self.reset_buf[env_ids_torch] = 0  # <- nicht mehr machen!

        # neue Kommandos
        self._resample_commands(env_ids_torch)

        # DOF-States in Buffern resetten
        noise = 0.02 * torch.randn_like(self.dof_pos[env_ids_torch])
        #noise = torch.clamp(noise, -0.03, 0.03)
        self.dof_pos[env_ids_torch] = self.default_dof_pos.unsqueeze(0) + noise # add small noise to default pose for better exploration after reset
        self.dof_vel[env_ids_torch] = 0.0

        # DOF-Posen in Genesis setzen
        #poses = self.default_dof_pos.detach().cpu().numpy().reshape(1, -1)
        #poses = np.repeat(poses, len(env_ids_np), axis=0)
        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids_torch].detach().cpu().numpy(),
            dofs_idx_local=self.motors_dof_idx,
            envs_idx=env_ids_np,
            zero_velocity=True,
        )

        # State aktualisieren & neue Obs holen
        self._update_robot_state()

        self.prev_contact[:] = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()

        obs, _ = self.get_observations()
        return obs

    def reset(self):
        self.reset_buf[:] = 0
        self.episode_length_buf[:] = 0
        for key in self.episode_sums:
            self.episode_sums[key].zero_()

        # Physik für alle Envs resetten
        self.genesis_scene.reset()

        # alle Env-IDs vorbereiten
        all_ids_torch = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        all_ids_np = all_ids_torch.cpu().numpy()

        # DOFs in den Buffern auf Default
        self.dof_pos[:] = self.default_dof_pos + 0.02 * torch.randn_like(self.dof_pos) # add small noise to default pose for better exploration after reset
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

        self.prev_contact[:] = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()

        # neue Commands ziehen
        self._resample_commands(all_ids_torch)

        # Beobachtungen zurückgeben
        obs, extras = self.get_observations()
        return obs, extras
    

    def step(self, actions):
        # 1) Actions speichern und anwenden
        self.last_actions[:] = self.actions
        self.actions = torch.clip(
            actions,
            -self.env_config_dataclass.clip_actions,
            self.env_config_dataclass.clip_actions
        )
        target = self.actions * self.env_config_dataclass.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target, self.motors_dof_idx)

        # 2) Simulationsschritt
        self.genesis_scene.step()

        # 3) Zustände updaten
        self._update_robot_state()

        # 4) Abbruchkriterien: Timeout vs. echter Sturz
        timeout = self.episode_length_buf >= self.max_episode_length          # nur Zeit
        fallen_mask = self._compute_fallen_mask()                             # echte Stürze
        done = timeout | fallen_mask                                          # Episodenende = eins von beidem

        # reset_buf: "in diesem Schritt ist die Episode zu Ende"
        self.reset_buf = done

        # 5) Rewards berechnen (Survive/Fall nutzen intern nur fallen_mask)
        self.rew_buf[:] = 0.0
        per_step = {}
        for name, fn in self.reward_functions.items():
            r = fn() * self.reward_scales[name]
            self.rew_buf += r
            self.episode_sums[name] += r
            per_step[name] = r

        # 6) Beobachtungen holen
        obs_buf, obs_extras = self.get_observations()

        # 7) Episodenlänge inkrementieren (wichtig: VOR Reset!)
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

        # 9) Kommandos bei Bedarf neu sampeln (optional deaktivierbar)
        if not self.disable_command_resampling:
            resample_every = int(self.command_config_dataclass.resampling_time_s / self.dt)
            mask = (self.episode_length_buf > 0) & (self.episode_length_buf % resample_every == 0)
            idx = mask.nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                self._resample_commands(idx)

        # 10) Envs mit done == True zurücksetzen
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.reset_idx(done_ids)

        # 11) (Optional) Debug
        if self.episode_length_buf[0] == 1:
            print("[DEBUG] Observation (env 0):", self.obs_buf[0])
            print("[DEBUG] Action      (env 0):", actions[0])
            print("[DEBUG] Reward      (env 0):", self.rew_buf[0])

        # 12) Ergebnis zurückgeben – done/reset_buf ist hier das Episodenende
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def get_observations(self):
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
        Robuste Fall-Erkennung für bipede Roboter.

        Gefallen gilt eine Env dann, wenn:
        - die Base-Height unter `height_thresh` liegt ODER
        - Roll ODER Pitch den jeweiligen Threshold überschreiten.

        Rückgabe:
            Bool-Tensor der Form (num_envs,), True = Roboter ist gefallen.
        """

        # Höhe
        height = self.base_pos[:, 2]
        too_low = height < self.reward_config_dataclass.base_height_threshold  # z.B. 0.33

        # Orientierung
        roll = self.base_euler[:, 0]
        pitch = self.base_euler[:, 1]
        roll_thresh = self.reward_config_dataclass.roll_threshold
        pitch_thresh = self.reward_config_dataclass.pitch_threshold
        bad_orientation = (roll.abs() > roll_thresh) | (pitch.abs() > pitch_thresh)

        # gefallen = zu niedrig ODER stark gekippt
        fallen = too_low | bad_orientation
        return fallen
    

    def _gait_gate(self):
        """
        0 bei Stand / sehr kleinen Commands,
        1 bei "wirklich laufen".
        Schwellen kannst du später in configs auslagern.
        """
        v = torch.norm(self.commands[:, 0:2], dim=1)
        vmin = 0.03   # darunter: wie "stehen"
        vmax = 0.17   # darüber: voller gait reward
        return torch.clamp((v - vmin) / (vmax - vmin), 0.0, 1.0)
    
    def _abduction_gate(self):
        """
        Gate für Abduction-Stabilisierung:
        - voll aktiv bei vy ~ 0
        - aus bei |vy| >= vy_max
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
        Phasenorientierte Gait‑Shaping‑Belohnung:
        Linke Stance‑Hälfte, rechte Swing‑Hälfte, dann umgekehrt.
        """
        # phase = (self.episode_length_buf.float() * self.dt) % self.reward_config_dataclass.period
        # half = self.reward_config_dataclass.period * 0.5
        # contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()
        # desired_left = (phase < half).float()
        # desired_right = (phase >= half).float()
        # # positiv im Bereich [0,1]
        # return desired_left * contact[:, 0] + desired_right * contact[:, 1]


        # gate = self._gait_gate()

        # phase = (self.episode_length_buf.float() * self.dt) % self.reward_config_dataclass.period
        # half = self.reward_config_dataclass.period * 0.5
        # contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()  # (N,2)

        # want_left_stance  = (phase < half).float()
        # want_right_stance = (phase >= half).float()

        # left_match  = want_left_stance  * contact[:,0] + (1-want_left_stance)  * (1-contact[:,0])
        # right_match = want_right_stance * contact[:,1] + (1-want_right_stance) * (1-contact[:,1])

        # return gate * 0.5 * (left_match + right_match)


        """
        Command-konditionierter Periodik-Reward (Mismatch-Form):
        - bei v_cmd ~ 0 soll NICHT periodisch gelaufen werden -> desired_match ~ 0
        - bei v_cmd groß soll periodisch gelaufen werden -> desired_match ~ 1
        Reward ist hoch, wenn actual_match ~ desired_match.
        """
        # 1) Actual "periodic match" aus deinem bestehenden Kontakt-Phasen-Match
        phase = (self.episode_length_buf.float() * self.dt) % self.reward_config_dataclass.period
        half = self.reward_config_dataclass.period * 0.5
        contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()  # (N,2)

        want_left_stance  = (phase < half).float()
        want_right_stance = (phase >= half).float()

        left_match  = want_left_stance  * contact[:, 0] + (1 - want_left_stance)  * (1 - contact[:, 0])
        right_match = want_right_stance * contact[:, 1] + (1 - want_right_stance) * (1 - contact[:, 1])
        match = 0.5 * (left_match + right_match)  # in [0,1]

        # 2) Desired match: wächst mit Command-Speed (dein existing gait_gate ist schon so eine Rampe)
        desired = self._gait_gate()  # 0..1 abhängig von ||cmd_xy||

        # 3) Mismatch-Penalty als Gauß-Reward um desired
        # sigma bestimmt, wie "hart" du das Match erzwingst
        sigma = 0.25
        err = (match - desired) ** 2
        return torch.exp(-err / (2 * sigma**2 + 1e-8))


    @register_reward()
    def _reward_energy_penalty(self):
        """
        Energie‑Effizienz als Gauß‑Reward statt -Summe.
        Minimierung der Aktionsänderungen.
        """
        err = torch.sum((self.actions - self.last_actions)**2, dim=1)
        sigma = self.reward_config_dataclass.energy_sigma 
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_foot_swing_clearance(self):
        """
        Command-konditionierter Swing-Clearance Reward.
        - bei v_cmd klein: kleine gewünschte Clearance
        - bei v_cmd groß: gewünschte Clearance nahe clearance_target
        Reward ist hoch, wenn die Swing-Höhe der Swing-Füße zur gewünschten Höhe passt.
        """
        g = self._gait_gate()                                  # (N,)

        hs = self.current_ankle_heights                        # (N,2)
        contact = (hs < self.CONTACT_HEIGHT).float()          # (N,2)
        swing_mask = 1.0 - contact                            # (N,2)

        target = self.reward_config_dataclass.clearance_target

        # nicht bis exakt 0 runter, damit "gar kein Swing" nicht optimal wird
        min_clearance = 0.025
        desired = (min_clearance + g * (target - min_clearance)).unsqueeze(1)

        err = (hs - desired) ** 2

        sigma = 0.02   # oder 0.02
        per_foot = torch.exp(-err / (2 * sigma**2))
        per_foot = per_foot * swing_mask

        num_swing = swing_mask.sum(dim=1).clamp(min=1.0)
        rew = per_foot.sum(dim=1) / num_swing

        # wenn kein Fuß swingt -> kein Reward
        no_swing = (swing_mask.sum(dim=1) < 0.5).float()
        rew = rew * (1.0 - no_swing)

        return rew



    @register_reward()
    def _reward_forward_torso_pitch(self):
        """
        Gauß‑Reward für Vorwärts‑Pitch nahe einem Sollwert.
        """
        pitch = self.base_euler[:, 1]
        err = (pitch - self.reward_config_dataclass.pitch_target)**2  
        sigma = self.reward_config_dataclass.pitch_sigma
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_knee_extension_at_push(self):
        # """
        # Belohnt gestrecktes Knie in Standphase (Kontakt).
        # """
        # hs = self.current_ankle_heights
        # stance = (hs < self.CONTACT_HEIGHT).any(dim=1).float()
        # idx_l = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.left_knee)
        # idx_r = self.joint_names.index(self.env_config_dataclass.joint_names_mapped.right_knee)
        # ext_l = 1.0 - torch.relu(-self.dof_pos[:, idx_l])
        # ext_r = 1.0 - torch.relu(-self.dof_pos[:, idx_r])
        # return stance * ((ext_l + ext_r) * 0.5)
        """
        Belohnt (nahezu) gestrecktes Knie in Standphase (Kontakt).
        """
        gate = self._gait_gate()

        hs = self.current_ankle_heights
        # irgendein Fuß im Kontakt → Standphase
        stance = (hs < self.CONTACT_HEIGHT).any(dim=1).float()

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
        Gauß‑Reward für lin. Geschw‑Tracking in x/y.
        """
        """ Alte Version
        v_cmd = self.commands[:, 0]
        v_x   = self.base_lin_vel[:, 0]
        sigma = self.reward_config_dataclass.tracking_sigma
        err   = (v_cmd - v_x)**2
        """
        # desired and actual linear velocity in BODY frame (bei dir ist base_lin_vel im Körperframe)
        cmd_xy = self.commands[:, 0:2]        # (N,2)
        vel_xy = self.base_lin_vel[:, 0:2]    # (N,2)

        err = torch.sum((cmd_xy - vel_xy) ** 2, dim=1)
        sigma = self.reward_config_dataclass.tracking_sigma


        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_tracking_ang_vel(self):
        """
        Gauß‑Reward für ang. Geschw‑Tracking in z.
        """
        err = (self.commands[:, 2] - self.base_ang_vel[:, 2])**2
        sigma = self.reward_config_dataclass.tracking_sigma * 2.0  # evtl. engeres Tracking für Rotation
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_orientation_stability(self):
        """
        Gauß‑Reward für kleine Roll‑/Pitch‑Abweichung.
        """
        roll = self.base_euler[:, 0] 
        pitch = self.base_euler[:, 1] 
        err = roll**2 + pitch**2
        sigma = self.reward_config_dataclass.orient_sigma
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_base_height(self):
        """
        Gauß‑Reward für Hüfthöhe nahe Ziel.
        """
        err = (self.base_pos[:, 2] - self.reward_config_dataclass.base_height_target)**2 
        sigma = self.reward_config_dataclass.height_sigma
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_survive(self):
        """
        Überlebens-Reward, der mit Episodenfortschritt wächst.
        - 0 am Anfang der Episode
        - 1 kurz vor Timeout (wenn nicht gefallen)
        Timeouts sind okay, Stürze nicht.
        """
        
        fallen = self._compute_fallen_mask().float()  # 1.0 bei Sturz
        alive  = 1.0 - fallen                          # 1.0 solange nicht gefallen

        # normierte Episodenlänge in [0, 1]
        prog = self.episode_length_buf.float() / float(self.max_episode_length)
        prog = torch.clamp(prog, 0.0, 1.0)

        # damit es nicht ganz bei 0 startet: z.B. 0.2..1.0
        per_step = 0.05 + 0.8 * prog

        return alive * per_step # better for walking
        #return 1.0 - fallen # better for standing

    
    @register_reward()
    def _reward_fall_penalty(self):
        # """
        # Bestrafe Umfallen: sobald Roll oder Pitch über den Schwellwert gehen.
        # Gibt –1.0 pro Step zurück, wenn überschritten.
        # """
        # # Roll und Pitch in Radiant
        # roll  = self.base_euler[:, 1] 
        # pitch = self.base_euler[:, 0] 

        # # Thresholds aus reward_cfg (z.B. 30° in Radiant)
        # thr_r = self.reward_config_dataclass.roll_threshold
        # thr_p = self.reward_config_dataclass.pitch_threshold

        # # Maske, wo einer der Winkel überschritten ist
        # mask = ((roll.abs() > thr_r) | (pitch.abs() > thr_p)).float()

        # # Als Penalty skaliert –1 pro Step
        # return -mask

        """
        Große Strafe nur bei echtem Sturz.
        Timeouts sind neutral.
        """
        fallen = self._compute_fallen_mask().float()  # 1.0 bei Sturz
        return -fallen   # mit Scale z.B. 100.0 in deiner Config



    @register_reward()
    def _reward_bird_hip_phase(self):
        """
        Vogel‑typischer Hüft‑FE‑Zyklustreiber als Gauß‑Reward.
        """
        # idx_l = self.idx_left_thigh
        # idx_r = self.idx_right_thigh
        # phase = ((self.episode_length_buf.float() * self.dt) % self.reward_config_dataclass.period) / self.reward_config_dataclass.period
        # omega = 2 * math.pi * phase
        # tgt  = self.reward_config_dataclass.bird_hip_target
        # amp  = self.reward_config_dataclass.bird_hip_amp
        # desired_l = tgt + amp * torch.sin(omega)
        # desired_r = tgt - amp * torch.sin(omega)
        # a_l = self.dof_pos[:, idx_l]
        # a_r = self.dof_pos[:, idx_r]
        # err = (a_l - desired_l)**2 + (a_r - desired_r)**2
        # sigma = self.reward_config_dataclass.bird_hip_sigma
        # return torch.exp(-err / (2 * sigma**2))
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
        # """
        # Gauß‑Strafe für Hüft‑AA Abduktion/Adduktion.
        # """
        # idx_l = self.idx_left_hip
        # idx_r = self.idx_right_hip
        # abd_l = self.dof_pos[:, idx_l]
        # abd_r = self.dof_pos[:, idx_r]
        # err = abd_l**2 + abd_r**2
        # sigma = self.reward_config_dataclass.hip_abduction_sigma
        # return torch.exp(-err / (2 * sigma**2))

        # """
        # Gauß-Strafe für Hüft-AA Abduktion/Adduktion.
        # Nur aktiv wenn |v_y| klein ist (sonst braucht er Abspreizen).
        # """
        # gate = self._abduction_gate()  # <- deine neue Funktion

        # idx_l = self.idx_left_hip
        # idx_r = self.idx_right_hip
        # abd_l = self.dof_pos[:, idx_l]
        # abd_r = self.dof_pos[:, idx_r]

        # err = abd_l**2 + abd_r**2
        # sigma = self.reward_config_dataclass.hip_abduction_sigma

        # return gate * torch.exp(-err / (2 * sigma**2))

        """
        Command-konditionierter Abduction-Reward (Mismatch-Form):
        - bei |vy_cmd| ~ 0: desired_abd ~ 0  (Spur halten)
        - bei |vy_cmd| groß: desired_abd ~ abd_max (Seitwärtsgehen erlauben/unterstützen)
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


        # """
        # Echte Penalty (negativ) für Hüft-Abduktion/Adduktion.
        # """
        # idx_l = self.idx_left_hip
        # idx_r = self.idx_right_hip
        # abd_l = self.dof_pos[:, idx_l]
        # abd_r = self.dof_pos[:, idx_r]

        # # Quadratische Strafe
        # err = abd_l**2 + abd_r**2
        # return -err


    @register_reward()
    def _reward_lateral_drift_penalty(self):
        """
        Gauß‑Reward für geringe seitliche Drift (y‑Geschw.).
        """
        drift = self.base_lin_vel[:, 1].abs()
        sigma = self.reward_config_dataclass.drift_sigma
        return torch.exp(-drift**2 / (2 * sigma**2))
    
    @register_reward()
    def _reward_vertical_stability(self):
        v_z = self.base_lin_vel[:, 2]
        sigma = 0.2
        return torch.exp(- (v_z**2) / (2 * sigma**2))
    
    
    @register_reward()
    def _reward_action_rate(self):
        # """
        # Bestraft schnelle Änderungen in den Aktionen (hohe Aktionsrate).
        # Größere Änderungen -> stärkerer negativer Reward.
        # """
        
        # diff = self.actions - self.last_actions
        # err = torch.sum(diff**2, dim=1)
        # sigma = 0.2 * self.env_config_dataclass.action_scale  # relative to action scale
        # return torch.exp(-err / (2 * sigma**2))
        """
        Echte Penalty (negativ) für schnelle Aktionsänderungen -> reduziert Hochfrequenz-Jitter.
        """
        diff = self.actions - self.last_actions
        err = torch.sum(diff**2, dim=1)
        return -err
    
    @register_reward()
    def _reward_step_events(self):
        # gate = self._gait_gate()

        # contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()  # (N,2)
        # events = torch.clamp(contact - self.prev_contact, min=0.0)
        # self.prev_contact[:] = contact

        # return gate * events.sum(dim=1)

        """
        Command-konditionierter Step-Event Reward (Mismatch-Form):
        - bei v_cmd ~ 0: desired_events ~ 0 (keine Schritte)
        - bei v_cmd groß: desired_events ~ 2*dt/period (regelmäßige Schritte)
        Reward hoch, wenn Event-Rate zur gewünschten passt.
        """
        g = self._gait_gate()
        contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()
        events = torch.clamp(contact - self.prev_contact, min=0.0)
        self.prev_contact[:] = contact
        return g * events.sum(dim=1)   # 0..2