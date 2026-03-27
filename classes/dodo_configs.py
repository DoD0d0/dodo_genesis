from typing import TypedDict, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, is_dataclass
from numpy import pi
import numpy as np


# -----------------------------------------------------------------------------
# Use Dataclasses for Programming 
# They can be converted into normal dicts if necessary.
# This was done to avoid the long dicts and avoid spelling mistakes in the keys. You can access the parameters with dot notation, which is more convenient and less error-prone.
# -----------------------------------------------------------------------------

@dataclass
class DodoJointParams:
    """
    Default joint parametersof the Dodo robot for each leg joint.
    """
    left_hip: float
    right_hip: float
    left_thigh: float
    right_thigh: float
    left_knee: float
    right_knee: float
    left_foot_ankle: float
    right_foot_ankle: float

@dataclass
class DodoJointNames:
    """
    Joint name mapping used by the simulator/URDF/MJCF for each Dodo leg joint.
    The KEYS are the naming convention, this programm is using.
    The corresponding VALUES should be the joint names used in the URDF or the XML file your are currently using!

    -> The class "FileFormatAndPaths" can be used to extract the joint names from those files! You just need to do the remapping.
    """
    left_hip: str
    right_hip: str
    left_thigh: str
    right_thigh: str
    left_knee: str
    right_knee: str
    left_foot_ankle: str
    right_foot_ankle: str

@dataclass
class DodoObservations:
    """
    Scaling factors applied to the different observation components.
    """
    lin_vel: float
    ang_vel: float
    dof_pos: float
    dof_vel: float

@dataclass
class RewardScales:
    """
    Per-term scaling factors that weight the individual reward components.
    If you add a new reward term, you should add a new entry here and also in the reward function in dodo_environment.py
    """
    tracking_lin_vel: float
    tracking_ang_vel: float
    orientation_stability: float
    base_height: float
    survive: float
    fall_penalty: float
    vertical_stability: float
    periodic_gait: float
    foot_swing_clearance: float
    knee_extension_at_push: float
    bird_hip_phase: float
    forward_torso_pitch: float
    hip_abduction_penalty: float
    lateral_drift_penalty: float
    energy_penalty: float
    action_rate: float
    step_events: float

@dataclass
class CommandRanges:
    """
    Sampling ranges [min, max] for the commanded base velocities.
    """
    lin_vel_x: List[float]
    lin_vel_y: List[float]
    ang_vel_yaw: List[float]

@dataclass
class TrainAlgorithm:
    """
    Defining the RL training algorithm and its hyperparameters.
    """
    class_name: str
    clip_param: float
    desired_kl: float
    entropy_coef: float
    gamma: float
    lam: float
    learning_rate: float
    max_grad_norm: float
    num_learning_epochs: int
    num_mini_batches: int
    schedule: str
    use_clipped_value_loss: bool
    value_loss_coef: float

@dataclass
class TrainPolicy:
    """
    Defining the RL training policy
    """
    activation: str
    actor_hidden_dims: List
    critic_hidden_dims: List
    init_noise_std: float
    class_name: str

@dataclass
class TrainRunner:
    """
    Defining the RL runner parameters
    """
    checkpoint: int
    experiment_name: str
    load_run: int
    log_interval: int
    max_iterations: int
    record_interval: int
    resume: bool
    resume_path: Any
    run_name: str


# -----------------------------------------------------------------------------
# Terrain Configs
# -----------------------------------------------------------------------------

@dataclass
class UnevenTerrainCfg:
    """
    Configuration for uneven terrain generation.
    """
    n_subterrains: Tuple[int, int]
    subterrain_size: Tuple[float, float]
    horizontal_scale: float
    vertical_scale: float
    spawn_flat_radius_sub: int
    border_flat: bool
    randomize: bool


@dataclass
class TerrainCfg:
    """
    Configuration for terrain generation.
    """
    # "plane" | "uneven" | "random"
    mode: str

    # Für mode="random" (leicht erweiterbar später)
    options: List[str]
    probs:   List[float]

    uneven: UnevenTerrainCfg

# -----------------------------------------------------------------------------
# Config-Classes 
# -----------------------------------------------------------------------------

@dataclass
class EnvCfg:
    """
    Configuration of the Dodo environment: robot model, init state and
    general simulation and control parameters used during training/evaluation.
    """
    num_actions: int
    default_joint_angles: DodoJointParams
    joint_names_mapped: DodoJointNames
    kp: DodoJointParams
    kd: DodoJointParams
    max_torques: DodoJointParams
    max_velocities: DodoJointParams
    termination_if_roll_greater_than: float
    termination_if_pitch_greater_than: float
    base_init_pos: List[float]
    base_init_quat: List[float]
    episode_length_s: float
    resampling_time_s: float
    action_scale: float
    simulate_action_latency: bool
    clip_actions: float
    robot_file_path: str
    foot_link_names: List[str]
    robot_file_format: str
    terrain_cfg: TerrainCfg
    contact_height: float
    init_pose_noise: float


@dataclass
class ObsCfg:
    """
    Observation configuration: total observation size and scaling per component.
    """
    num_obs: int
    obs_scales: DodoObservations

@dataclass
class RewardCfg:
    """
    Reward configuration: scales for each term and shaping hyperparameters
    (targets and sigmas) used in the reward functions.
    """
    reward_scales: RewardScales
    tracking_sigma: float
    base_height_target: float
    height_sigma: float
    orient_sigma: float
    energy_sigma: float
    period: float
    clearance_target: float
    pitch_target: float
    pitch_sigma: float
    bird_hip_target: float
    bird_hip_amp: float
    bird_hip_sigma: float
    hip_abduction_sigma: float
    drift_sigma: float
    pitch_threshold: float
    roll_threshold: float
    base_height_threshold: float

@dataclass
class CommandCfg:
    """
    Command configuration: number of commands and how often as well as from
    which ranges they are (re-)sampled.
    """
    num_commands: int
    resampling_time_s: float
    command_ranges: CommandRanges


@dataclass
class TrainCfg:
    """
    Full training configuration used by the PPO trainer (rsl-rl).
    
    This config defines:
      • PPO algorithm hyperparameters (learning rates, clipping, KL targets)
      • Policy network architecture and initialization settings
      • Runner behavior (logging, checkpoints, iteration limits)
      • General training settings such as random seeds and normalization

    These parameters collectively determine how the agent is optimized, how
    rollouts are collected, and how model checkpoints and logs are saved.
    """
    algorithm: TrainAlgorithm
    init_member_classes: dict
    policy: TrainPolicy
    runner: TrainRunner
    runner_class_name: str
    num_steps_per_env: int
    save_interval: int
    empirical_normalization: Any
    seed: int
    logger: str
    tensorboard_subdir: str




# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def init_dodo_configs(
        exp_name: str, 
        max_iterations: int, 
        joint_names: list[str], 
        robot_file_path_relative: str,
        foot_link_names: list[str],
        robot_file_format: str,
        num_obs: int
        ) -> tuple[EnvCfg, ObsCfg, RewardCfg, CommandCfg, TrainCfg] :
    """
    This function can be used to directly create and return all configs that are relevant for the DODO training as dataclasses.
    You can easily create a dataclass object on your own and use that.
    This function can be used it should have good values for training walking using the dodo.

    JUST CHANGE THE VALUES IN THIS FUNCTION IF YOU WANT TO CHANGE THE CONFIGS FOR THE TRAINING.
    
    :param exp_name: Name of the experiment
    :type exp_name: str
    :param max_iterations: Max iterations for the RL training
    :type max_iterations: int
    :param joint_names: List of the joint names of the robot
    :type joint_names: list[str]
    :param robot_file_path_relative: Realtive path to your dodo robot urdf/xml file (starting from project root)
    :type robot_file_path_relative: str
    :param foot_link_names: List of the foot links of the robot
    :type foot_link_names: list[str]
    :param robot_file_format: Format of the robot while (usually "xml" or "urdf")
    :type robot_file_format: str
    :param num_obs: Number of observations that are made for the RL training.
    :type num_obs: int
    """
    
    train_config_dataclass: TrainCfg = TrainCfg(
        algorithm=TrainAlgorithm(
            class_name =                  "PPO",
            clip_param =                  0.2,        # PPO clipping parameter epsilon, which limits how much the policy can change in one update.
            desired_kl =                  0.021,      # KL divergence target for adaptive KL penalty. If the actual KL divergence after an update is greater than this value, the update will be penalized to keep the policy changes within a reasonable range.
            entropy_coef =                0.01,       # Coefficient for the entropy bonus, which encourages exploration by adding a term to the loss function that rewards higher entropy (more randomness) in the action distribution.
            gamma =                       0.99,       # Discount factor for future rewards, which determines how much the agent values future rewards compared to immediate rewards. A value close to 1 means the agent will consider long-term rewards more strongly.
            lam =                         0.95,       # GAE lambda parameter for advantage estimation, which controls the bias-variance tradeoff in the Generalized Advantage Estimation (GAE) method. A value close to 1 will use more of the future rewards for advantage estimation, while a value close to 0 will rely more on the immediate reward.
            learning_rate =               1.5e-4,     # Learning rate for the optimizer
            max_grad_norm =               1.0,        # Maximum norm for gradient clipping, which helps to prevent exploding gradients during training by capping the maximum value of the gradients.
            num_learning_epochs =         4,          # Number of epochs to perform for each update, which determines how many times the entire batch of collected data will be used to update the policy in one iteration.
            num_mini_batches =            8,          # Number of mini-batches to split the collected data into for each update. This is used in conjunction with num_learning_epochs to determine how the data is shuffled and fed into the optimizer. More batches === smaller batch size per batch === more updates per iteration but also more noise in the updates (can be beneficial for exploration and avoid local optima).
            schedule =                    "adaptive", # Learning rate schedule, which can be "constant", "linear", "adaptive", etc. "adaptive" means the learning rate will be adjusted based on the KL divergence between the old and new policies to keep updates stable.
            use_clipped_value_loss =      True,       # Whether to use a clipped value loss, which is a technique to stabilize training by clipping the value function updates in a similar way to how the policy updates are clipped in PPO.
            value_loss_coef =             1.0         # Coefficient for the value loss term in the overall loss function, which determines how much the value function error contributes to the total loss during training.
        ),
        init_member_classes={},
        policy=TrainPolicy(
            activation=                   "elu",
            actor_hidden_dims=            [512, 256, 128],
            critic_hidden_dims=           [512, 256, 128],
            init_noise_std=               0.20,     
            class_name=                   "ActorCritic"
        ),
        runner=TrainRunner(
        checkpoint=                     -1,
        experiment_name=                exp_name,
        load_run=                       -1,
        log_interval=                    5,
        max_iterations=                 max_iterations,
        record_interval=                -1,
        resume=                         False,
        resume_path=                    None,
        run_name=                       ""
        ),
        runner_class_name=                "OnPolicyRunner",
        # collect at least one gait cycle per env: e.g. 1.0s / dt(0.01) = 100 steps 
        num_steps_per_env=                160, # previous 192 -> longer rollouts === bigger batch size === higher time consumption per iteration.
        save_interval=                    50,
        empirical_normalization=          True,
        seed=                             1,
        logger=                           "wandb",
        tensorboard_subdir=               "tb"
    )


    terrain_config_dataclass: TerrainCfg = TerrainCfg(
        mode="plane",
        options=["plane", "uneven"],
        probs=[0.5, 0.5],
        uneven=UnevenTerrainCfg(
            n_subterrains=(16, 16),
            subterrain_size=(2.0, 2.0),
            horizontal_scale=0.25, # Add curriculum later
            vertical_scale=0.003,
            spawn_flat_radius_sub=0, # 0 is for just one flat subterrain in the center. 1 is for 3x3 flat subterrains etc.
            border_flat=True,
            randomize=True
        )
    )
    
    env_config_dataclass: EnvCfg = EnvCfg(
        num_actions= len(joint_names),
        # default_joint_angles=DodoJointAngles( # old dodo standing configuration
        #     left_hip=                     0.0,
        #     right_hip=                    0.0,
        #     left_thigh=                   0.4,
        #     right_thigh=                  0.4,
        #     left_knee=                    -0.7,
        #     right_knee=                   -0.7,
        #     left_foot_ankle=              0.3,
        #     right_foot_ankle=             0.3
        # ),
        default_joint_angles=DodoJointParams( # new dodo daimao configuration in rad
            left_hip=                     0.0, # standing = 0.0, lying flat = 0.0
            right_hip=                    0.0, # standing = 0.0, lying flat = 0.0
            left_thigh=                   0.6, # standing = 0.6, lying flat = 1.57
            right_thigh=                  0.6, # standing = 0.6, lying flat = 1.57
            left_knee=                    -1.4, # standing = -1.4, lying flat = -3.14
            right_knee=                   -1.4, # standing = -1.4, lying flat = -3.14
            left_foot_ankle=              0.8, # standing = 0.8, lying flat = 1.57
            right_foot_ankle=             0.8 # standing = 0.8, lying flat = 1.57
        ),
        joint_names_mapped=DodoJointNames( # The values should be the joint_names from the robot file (URDF or XML) you can hardcode it here if you want to. I am using the joint names that are automatically extracted by the helper function. The keys should stay the same because they are used in the rest of the code to access the joints in a unified way independent of the actual joint names in the robot file.
            left_hip=                     joint_names[4],
            right_hip=                    joint_names[0],
            left_thigh=                   joint_names[5],
            right_thigh=                  joint_names[1],
            left_knee=                    joint_names[6],
            right_knee=                   joint_names[2],
            left_foot_ankle=              joint_names[7],
            right_foot_ankle=             joint_names[3],
        ),
        max_torques=DodoJointParams( # max torques for each joint in Nm
            left_hip=                     27.0,
            right_hip=                    27.0,
            left_thigh=                   27.0,
            right_thigh=                  27.0,
            left_knee=                    9.0,
            right_knee=                   9.0,
            left_foot_ankle=              9.0,
            right_foot_ankle=             9.0
        ),
        max_velocities=DodoJointParams( # max velocities for each joint in rad/s
            left_hip=                     6.0,     
            right_hip=                    6.0,
            left_thigh=                   6.0,
            right_thigh=                  6.0,
            left_knee=                    6.0,
            right_knee=                   6.0,
            left_foot_ankle=              6.0,
            right_foot_ankle=             6.0
        ),
        kp=DodoJointParams( 
            left_hip=70.0,
            right_hip=70.0,
            left_thigh=110.0,
            right_thigh=110.0,
            left_knee=55.0,
            right_knee=55.0,
            left_foot_ankle=35.0,
            right_foot_ankle=35.0,
        ),
        kd=DodoJointParams(
            left_hip=1.4 * np.sqrt(70.0),
            right_hip=1.4 * np.sqrt(70.0),
            left_thigh=1.4 * np.sqrt(110.0),
            right_thigh=1.4 * np.sqrt(110.0),
            left_knee=1.5 * np.sqrt(55.0),
            right_knee=1.5 * np.sqrt(55.0),
            left_foot_ankle=1.6 * np.sqrt(35.0),
            right_foot_ankle=1.6 * np.sqrt(35.0),
        ),
        termination_if_roll_greater_than= 30.0,
        termination_if_pitch_greater_than=30.0,
        base_init_pos=                    [0.0, 0.0, 0.38],
        base_init_quat=                   [1.0, 0.0, 0.0, 0.0],
        episode_length_s=                 10.0,
        resampling_time_s=                4.0,
        action_scale=                     0.7, 
        simulate_action_latency=          False, 
        clip_actions=                     1.3,                          # clamp actions to this value (after scaling), to avoid exploding actions. Actions are offsets to the init pose and in rad, most joints have limits of around 1.5 rad, it does not make sense to have actions that are greater than that, it can also destabilize the training if the actions are too big in the beginning.
        robot_file_path=                  robot_file_path_relative,     # for example: "robot_mjcf": dodo_robot\dodo.xml
        foot_link_names=                  foot_link_names,              # for example: ['Left_FOOT_FE', 'Right_FOOT_FE']
        robot_file_format=                robot_file_format,            # for example "xml" or "urdf"
        terrain_cfg=                      terrain_config_dataclass, 
        contact_height=                   0.05,                         # m, if the foot is lower than that it is considered in contact with the ground.
        init_pose_noise=                  0.02                          # std of the noise added to the initial joint angles after reset for better exploration and more robust policies (tested values up to 0.04, but it can cause convergence issues if it is too high
    )    
    
    obs_config_dataclass: ObsCfg = ObsCfg(
        num_obs=                          num_obs,
        obs_scales=DodoObservations(
            lin_vel=                      2.0,
            ang_vel=                      0.25, 
            dof_pos=                      1.0,
            dof_vel=                      0.05
        )
    )

    reward_config_dataclass: RewardCfg = RewardCfg(
        # All reward functions are defined at the bottom of the dodo_environment.py file!
        reward_scales=RewardScales(
            #velocity tracking
            tracking_lin_vel=             5.0,    # reward: track commanded linear velocity
            tracking_ang_vel=             5.0,    # reward: track commanded angular velocity
            #stability and posture   
            orientation_stability=        0.4,    # reward: keep stable orientation (not rolling or pitching)
            base_height=                  0.5,    # reward: keep the base at a desired height
            survive=                      0.2,    # reward: survive many steps
            fall_penalty=                 60.0,   # penalty: avoid falling
            vertical_stability=           0.06,   # penalty: avoid jumping
            #gait-shaping (bird style)
            periodic_gait=                0.0,    # reward: follow a periodic gait pattern (defined in dodo_environment.py)
            foot_swing_clearance=         2.0,    # reward: keep enough foot clearance during swing phase to avoid tripping
            knee_extension_at_push=       0.0,    # reward: extend the knee during the push-off phase for more propulsion
            bird_hip_phase=               0.0,    # reward: move the hip in a bird-like manner (defined in dodo_environment.py)
            forward_torso_pitch=          0.0,    # reward: lean the torso forward during walking
            #Joint penalties
            hip_abduction_penalty=        0.05,   # penalty: avoid hip abduction from the initial pose
            #drift and efficiency
            lateral_drift_penalty=        0.0,    # penalty: avoid lateral drift -> only usefull if cmd_vel_y is zero
            action_rate=                  0.000,  # penalty: avoid changing actions too quickly, which can be a sign of an unstable policy and can also damage the motors in real life -> It can reduce jitter
            energy_penalty=               0.05,   # penalty: similar to action_rate penalty -> Only use one of them.
            step_events=                  1.0,    # reward: reward for each step event
        ),
        # Hyperparameters for the shaping functions in the reward terms
        base_height_target=               0.37,           # m, Hip height target
        base_height_threshold=            0.23,           # m, if the base height gets lower than that the robot is considered fallen.
        clearance_target=                 0.09,           # m, target foot clearance during swing phase
        period=                           1.0,            # s, Cycle period for the periodic gait reward in seconds
        bird_hip_target=                 -0.7,            # rad, Hip target for the bird style gait
        bird_hip_amp=                     0.35,           # rad, cyclic amplitude for the bird hip movement
        pitch_target=                     0.0,            # rad, target torso pitch angle (leaning forward/backward)
        pitch_threshold=                  45 * pi/180,    # rad, if the torso pitch angle gets greater than that the robot is considered fallen.
        roll_threshold=                   45 * pi/180,    # rad, if the torso roll angle gets greater than that the robot is considered fallen.
        orient_sigma=                     0.08,           # sigma, Roll/Pitch sigma
        height_sigma=                     0.07,           # sigma, Hip height sigma
        pitch_sigma=                      0.07,           # sigma, Pitch reward
        energy_sigma=                     0.15,           # sigma, changing actions
        tracking_sigma=                   0.12,           # sigma, Velocity tracking
        bird_hip_sigma=                   0.12,           # sigma, bird hip reward
        hip_abduction_sigma=              0.13,           # sigma, Hip abduction from initial pose
        drift_sigma=                      0.08,           # sigma,  lateral drift
    )   

    command_config_dataclass: CommandCfg = CommandCfg(
        num_commands= 3,
        resampling_time_s= 4.0,
        command_ranges=CommandRanges(
            lin_vel_x=[-0.2, 0.5],
            lin_vel_y=[-0.3, 0.3], 
            ang_vel_yaw=[-1.0, 1.0] # for example [-1.0, 1.0]
        )
    )


    return env_config_dataclass, obs_config_dataclass, reward_config_dataclass, command_config_dataclass, train_config_dataclass


def dataclass_to_dict(**kwargs) -> Dict[str, Any]:
    """
    Convert multiple dataclass instances into dicts and return them
    in the same order as provided in the function call.

    Usage:
        env_cfg, obs_cfg, reward_cfg = dataclass_to_dict(
            env_cfg=env,
            obs_cfg=obs,
            reward_cfg=reward,
        )
    """
    result = []

    for name, obj in kwargs.items():
        if not is_dataclass(obj):
            raise TypeError(f"Argument '{name}' is not a dataclass instance")
        result.append(asdict(obj))

    # return as tuple so it can be unpacked
    return tuple(result)