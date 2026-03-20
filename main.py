import numpy as np
from classes.dodo_environment import DodoEnvironment
from classes.file_format_and_paths import FileFormatAndPaths
import genesis as gs
import argparse


def main():
    # -----------------------------------------------------------------------------
    # Initialize Arguments that can be given in the CLI
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=1) #4096 oder 8192
    parser.add_argument("--max_iterations", type=int, default=2500)
    args = parser.parse_args()

    exp_name = args.exp_name
    num_envs = args.num_envs
    max_iterations = args.max_iterations


    # -----------------------------------------------------------------------------
    # Initialize Genesis
    # -----------------------------------------------------------------------------
    gs.init(backend=gs.cuda) #gs.cuda or gs.cpu
    

    # -----------------------------------------------------------------------------
    # Initialize relevant classes and the Dodo Environment
    # -----------------------------------------------------------------------------
    #dodo_path_helper: FileFormatAndPaths = FileFormatAndPaths(robot_file_name="dodobot_v3_simple.urdf") #"dodobot_v3.urdf" or "dodo.xml"
    dodo_path_helper: FileFormatAndPaths = FileFormatAndPaths(robot_file_name="dodo_daimao_simple.urdf") #"dodobot_v3.urdf" or "dodo.xml"
    print("robot joint names: ",dodo_path_helper.joint_names) # You can use those joint order to do the correct remapping of the joints in dodo_configs.py at "joint_names_mapped"
    print("robot foot links: ", dodo_path_helper.foot_link_names)

    # Create an instance of the DodoEnvironment class, which will handle the simulation, training and evaluation of the Dodo robot.
    dodo_env: DodoEnvironment = DodoEnvironment(
        dodo_path_helper=dodo_path_helper, 
        exp_name=exp_name, 
        num_envs=num_envs, 
        max_iterations=max_iterations
        )
    
    # -----------------------------------------------------------------------------
    # Main functions for testing, training and evaluating the Dodo robot. You can uncomment the desired function and run the script to execute it.
    # -----------------------------------------------------------------------------

    """
    The following function can be used to import the robot in a simple simulation configuration, which is useful for debugging and testing the 
    Genesis API. You can see if the robot spawns correctly, if the simulation is running without errors and if the joints are moving as expected (here the joints should move in a sinusoidal pattern, which is defined in the dodo_environment class in the function import_robot_sim).
    """
    #dodo_env.import_robot_sim(manual_stepping=False, total_steps=2000, spawn_position=(0.0, 0.0, 0.55))

    """
    The following function can be used to test the robot controller with a simple PD control pattern. 
    This is useful for debugging and testing the controller implementation, 
    and also for tuning the PD gains to achieve a stable standing configuration of the robot.
    """
    #dodo_env.test_robot_controller()

    """
    The following function can be used to import the robot in a standing configuration, which is useful for debugging and testing a new URDF file
    or initial parameters like choosing the right spawn position, etc...
    """
    #dodo_env.import_robot_standing(manual_stepping=False, total_steps=1000, spawn_position=(0.0, 0.0, 0.38)) # old dodobot_v3 was (0.0, 0.0, 0.55) new dodo_daimao standing is (0.0, 0.0, 0.38), new dodo_daimao lying is (0.0, 0.0, 0.095)

    """
    The following function can be used to train a new model.
    First make sure to set the desired hyperparameters in the dodo_environment class (like the reward functions, observations, ppo hyperparameters, and also the hyperparameters in dodo_configs.py, etc...)

    Example function call:
    python main.py --num_envs 4096 --max_iterations 150 --exp_name dodo-standing
    """
    #dodo_env.dodo_train() #Training from scratch (random weights initialization)

    """
    The following function can be used to evaluate a trained model.
    The function opens a window with the simulation and visualizes the robot's behavior using the trained model and the given velocity commands.
    """
    dodo_env.eval_trained_model(exp_name="daimao_walking_002", v_x=0.4, v_y=0.0, v_ang=0.0, model_name="model_best.pt")

    """
    The following function can be used to export an already trained model to a JIT format.
    This JIT is neccessary for sim2sim and sim2real transfer, as it can be loaded in C++ and is not dependent on the Python environment.

    -> gs.init(backend=gs.cpu) should be used before loading the JIT model in C++ for sim2sim or sim2real transfer, as the JIT model is exported in a CPU compatible format. 
    """
    #dodo_env.export_checkpoint_to_jit(exp_name="dodo-walking-new-009", model_name="model_final.pt")

    """
    The following function can be used to resume training from a previously trained checkpoint. 
    This is useful if you want to continue training a model that has already been trained for some iterations, 
    or if you want to fine-tune a pre-trained model on a new task or with different hyperparameters.
    """
    # checkpoint_path = "C:\\Users\\Liamb\\SynologyDrive\\TUM\\3_Semester\\dodo_alive\\dodo_genesis\\logs\\dodo-standing_007-curr002\\model_final.pt"
    # dodo_env.dodo_train(
    #     resume_from=checkpoint_path,
    # )

if __name__ == "__main__":
    main()