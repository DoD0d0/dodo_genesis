import numpy as np
import genesis as gs
from genesis import Scene

# Init scene and define simulation options 
gs.init(backend=gs.cuda)

sim_options = gs.options.SimOptions(
    dt=0.01, 
    substeps=2,
    gravity=(0, 0, -9.81)
    )

viewer_options = gs.options.ViewerOptions(
    max_FPS=100,
    camera_pos=(2.0, 0.0, 2.5),
    camera_lookat=(0.0, 0.0, 0.5),
    camera_fov=40
    )

rigid_options= gs.options.RigidOptions(
    dt=0.01,
    constraint_solver=gs.constraint_solver.Newton,
    enable_collision=True,
    enable_joint_limit=True,
    )

genesis_scene = Scene(
    show_viewer=True,
    sim_options=sim_options,
    viewer_options=viewer_options,
    rigid_options=rigid_options,
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    show_FPS=True,
    )

####################################################################################

# add entities to the scene
plane = genesis_scene.add_entity(
    gs.morphs.Plane(),
)

robot = genesis_scene.add_entity(
    gs.morphs.URDF(      
        file  = str("assets/dodobot.urdf"),
        fixed = False,
        pos   = np.array([0.0, 0.0, 0.5]),
        )
    )

####################################################################################

# set joint names and indices
jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

####################################################################################

# set positional and velocity gains
robot.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)


robot.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)

####################################################################################

# control loop
for i in range(150):
    if i < 50:
        robot.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i < 100:
        robot.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i <= 150:
        robot.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )

    genesis_scene.step()