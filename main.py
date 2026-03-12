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
    gs.init(backend=gs.cpu) #gs.cuda or gs.cpu
    

    # -----------------------------------------------------------------------------
    # Initialize relevant classes and the Dodo Environment
    # -----------------------------------------------------------------------------
    dodo_path_helper: FileFormatAndPaths = FileFormatAndPaths(robot_file_name="dodobot_v3_simple.urdf") #"dodobot_v3.urdf" or "dodo.xml"
    
    dodo_env: DodoEnvironment = DodoEnvironment(
        dodo_path_helper=dodo_path_helper, 
        exp_name=exp_name, 
        num_envs=num_envs, 
        max_iterations=max_iterations
        )

    #dodo_env.import_robot_sim(manual_stepping=False, total_steps=2000, spawn_position=(0.0, 0.0, 0.55))
    
    #dodo_env.import_robot_standing(manual_stepping=False, total_steps=1000, spawn_position=(0.0, 0.0, 0.55))

    #dodo_env.dodo_train_walking() #Training from scratch (random weights initialization)

    #dodo_env.eval_trained_model(exp_name="dodo-walking-new-005-curr003", v_x=0.8, v_y=0.0, v_ang=0.0, model_name="model_final.pt")

    dodo_env.export_checkpoint_to_jit(exp_name="dodo-standing", model_name="model_final.pt")

    # checkpoint_path = "C:\\Users\\Liamb\\SynologyDrive\\TUM\\3_Semester\\dodo_alive\\dodo_genesis\\logs\\dodo-walking-new-005-curr002\\model_final.pt"
    # dodo_env.dodo_train_walking(
    #     resume_from=checkpoint_path,
    # )

if __name__ == "__main__":
    main()