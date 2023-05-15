import numpy as np
import torch
import os
import pandas as pd

from torch.utils.data import Dataset
from tqdm import tqdm

from typing import List
from training.network import CustomNet
from training.dataset import CustomDataset
# from modeling import VRNet, DataLoader

############# TASK 1 #############
# import gym
# DATA_DIR = "data/cartpole-samples"  
# EPISODE_LIST = list(range(7))  
# MODEL_PATH = "models/task1/"
# MODEL_NAME = "task1_model_ep-100_lr-0.0005_bs-32_opt-adam_loss-bce"
# BATCH_SIZE = 32

# NET_CONFIG = {
#         "fc4_out_dim": 1, 
#         "spatialSoftmax_dim": (193, 293)
#     }
# CLASSIFICATION_TYPE = "binary"
# LOSS_FUNCTION = "bce"

# TASK = 1
# TASK_ENVIRONMENT = 'CartPole-v1'

############# TASK 2 #############
import gymnasium as gym
DATA_DIR = "data/battlezone-samples"
EPISODE_LIST = [0,1]
MODEL_PATH = "models/task2/"
MODEL_NAME = "task2_model_ep-150_lr-0.001_bs-64_opt-adam_loss-lg"
BATCH_SIZE = "64"
NET_CONFIG = {
    "fc4_out_dim": 9,
    "spatialSoftmax_dim": (98, 73)
}
CLASSIFICATION_TYPE = "multi-class"
LOSS_FUNCTION = "lg"

# PARAM_GRID = {
#     "num_epochs": [100],
#     "learning_rate": [1e-03, 5e-04, 1e-04, 5e-05, 1e-05],
#     "batch_size": [16, 32, 64, 128], 
#     "optimizer": ["adam"],
# }

TASK = 2
TASK_ENVIRONMENT = 'ALE/BattleZone-v5'

##############################################

def run_model_on_environment(model_path, model_name, rgb_std, rgb_mean):
    model = CustomNet(**NET_CONFIG)

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda:0')
    else:
        print("Using CPU")
        device = torch.device('cpu')
    
    # device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path + model_name, map_location=device))

    model.to(device) # move model to GPU
    if TASK == 1:
        recording_env = gym.wrappers.Monitor(gym.make(TASK_ENVIRONMENT), MODEL_NAME, video_callable=lambda episode_id: True, force=True)
    else:
        recording_env = gym.wrappers.RecordVideo(gym.make(TASK_ENVIRONMENT, render_mode = "rgb_array"), MODEL_NAME, episode_trigger = lambda x:True)

    total_rewards_list = []
    counter = 0
    for _ in tqdm(range(10)):
        recording_env.reset()
        recording_env.metadata['render_fps'] = 25
        if TASK == 1:
            # Shape of obs is (H,W,C)
            obs = recording_env.render(mode = "rgb_array")
        else:
            obs = recording_env.render()
        done = False
        total_rewards = 0
        while not done:

            # Change the order of dimensions from (H, W, C) to (C, H, W)
            # this is done to match the format of tensor returned by torch.io.read_image
            rgb_tensor = torch.from_numpy(np.transpose(obs, (2, 0, 1))).float()

            image_tensors = []

            image_tensors.append(rgb_tensor)
            preprocessed_image = torch.stack(image_tensors).float() / 255
            preprocessed_image[:,0,:,:] = (preprocessed_image[:,0,:,:] - rgb_mean[0]) / rgb_std[0]
            preprocessed_image[:,1,:,:] = (preprocessed_image[:,1,:,:] - rgb_mean[1]) / rgb_std[1]
            preprocessed_image[:,2,:,:] = (preprocessed_image[:,2,:,:] - rgb_mean[2]) / rgb_std[2]
            
            preprocessed_image = preprocessed_image.to(device).float()
            
            output = model(preprocessed_image)
            
            if TASK == 1:
                predicted_probs = torch.sigmoid(output)
                predicted_action = int((predicted_probs > 0.5).float())
                obs, rewards, done, info = recording_env.step(predicted_action)
            # print("Predicted Action: ", predicted_action)
            else:
                predicted_probs = torch.softmax(output, dim = 1)
                predicted_action = torch.argmax(predicted_probs, dim = 1)
                obs, rewards, terminated, truncated, info = recording_env.step(predicted_action)

                if terminated or truncated:
                    done = True

            total_rewards += rewards
            if TASK == 1:
                obs = recording_env.render(mode = "rgb_array")
            else:
                obs = recording_env.render()
            
        print("Rewards for test " + str(counter) +": ", total_rewards)
        total_rewards_list.append(int(total_rewards))
        counter += 1

    if TASK == 2:
        recording_env.close()

    return total_rewards_list



def iterate_files(directory: str) -> List:
    """Iterates over the files in the given directory and returns a list of 
    found files."""
    files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fullpath = os.path.join(directory, filename)
        if os.path.isdir(fullpath):
            files += iterate_files(fullpath)
        else:
            files.append(fullpath)
    return files


def get_model_history_files(directory: str) -> List:
    """Returns a list of pkl model history files in the given directory. 
    Checks to make sure that a model file of the same name exists."""
    files = iterate_files(directory)
    # Sort files by name
    files.sort()
    
    model_ext = ".pth"
    history_ext = ".pkl"
    
    # Get the history files if a model file of the same name exists
    model_files = []
    model_names = []
    for file in files:
        if file.endswith(model_ext):
            history_file = file.replace(model_ext, history_ext)
            if history_file in files:
                model_files.append(file)
                model_name = file.split("/")[-1].replace(model_ext, "")
                model_names.append(model_name)
    return model_files, model_names


if __name__ == "__main__":
    transformed_dataset = CustomDataset(DATA_DIR, EPISODE_LIST)
    rgb_mean, rgb_std = transformed_dataset.rgb_mean, transformed_dataset.rgb_std
    
    _, model_names = get_model_history_files(MODEL_PATH)
    
    df = pd.DataFrame(columns = ["Model_Name", "Total Rewards", "Average_Rewards", "Max Rewards"])
    
    for model in model_names:
        total_rewards_list = run_model_on_environment(MODEL_PATH, MODEL_NAME + ".pth", rgb_std, rgb_mean)
        print("Total_Rewards_List: ", total_rewards_list)
        print("Max Reward: ", max(total_rewards_list))
        rewards_sum = 0
        for i in total_rewards_list:
            rewards_sum += i
        
        average_rewards = rewards_sum/10
        print("Average Rewards achieved by model: ", average_rewards)
        
        # Add row to df
        df = df.append({
            "Model_Name": model,
            "Total Rewards": total_rewards_list,
            "Average_Rewards": average_rewards,
            "Max Rewards": max(total_rewards_list)
        }, ignore_index = True)
        
        df.to_csv(f"Results_Task_{TASK}.csv", index = False)
