
import torch
import torchvision
from torch.utils.data import Dataset
import os
import gym
from tqdm import tqdm
import numpy as np
import pandas as pd
# import tensorflow as tf
import torch

from typing import List

from modeling import VRNet

def run_model_on_environment(model_name, rgb_std, rgb_mean):
    model = VRNet()
    model.load_state_dict(torch.load(model_name,map_location=torch.device('cuda:0')))
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    recording_env = gym.wrappers.Monitor(gym.make('CartPole-v1'), 'model_1_video', video_callable=lambda episode_id: True, force=True)
    
    obs = recording_env.reset()
    
    total_rewards = 0
    for _ in tqdm([0]):
        # obs = recording_env.reset()
        img_array = recording_env.render(mode='rgb_array')
        print(img_array.shape)
        dones = False
        tensor_lst = []
        while not dones:
            # image_tensor = torch.tensor(img_array)
            image_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device) 
            preprocessed_image = image_tensor / 255
            # tensor_lst.append(image_tensor)
            # preprocessed_image = torch.stack(image_tensor).float() / 255

            epsilon = 1e-6
            
            preprocessed_image[:,0,:,:] = (preprocessed_image[:,0,:,:] - rgb_mean[0]) / rgb_std[0]
            preprocessed_image[:,1,:,:] = (preprocessed_image[:,1,:,:] - rgb_mean[0]) / rgb_std[1]
            preprocessed_image[:,2,:,:] = (preprocessed_image[:,2,:,:] - rgb_mean[0]) / rgb_std[2]

            output = model(preprocessed_image)

            predicted_probs = torch.sigmoid(output)
            # predicted_action = (predicted_probs > 0.5).float()
            predicted_action = (predicted_probs > 0.5).float().cpu().numpy().squeeze().astype(int)  # Convert to integer action
            
            _, reward, done, _ = recording_env.step(predicted_action)
            total_rewards += reward

            img_array = recording_env.render(mode='rgb_array')  # Update the image array
            # obs, rewards, dones, info = recording_env.step(predicted_action)
    recording_env.close()

    return total_rewards
    # return rewards


class DataLoader(Dataset):

    def __init__(self, data_dir: str, episode_list: List, samples: int, batch_size=1, indices=None):
        self.data_dir = data_dir
        self.episode_list = episode_list # number of episodes
        self.samples = samples # number of samples in each episode
        self.batch_size = batch_size
        self.rgb_std, self.rgb_mean = self.load_data()

    def load_data(self):
        """Loads data from the data directory and returns a list of rgb images and a list of actions"""
        rgbs = []
        actions = []
        
        for i in self.episode_list:
            episode_number = str(i)
            print(f"Loading episode {episode_number}")

            episode_dir = os.path.join(self.data_dir, "episode-" + episode_number)
            episode_info_df = pd.read_csv(episode_dir + "/episode-" + episode_number +".csv")
            episode_actions = episode_info_df["Action"].iloc[0:self.samples].tolist() # Changed - by Sandra 10/04 - 7:13pm
            
            for j in range(self.samples):
                sample_number = str(j + 1)
                # if int(sample_number) < 10:
                #     zero_prefix = "00"
                # elif int(sample_number) < 100:
                #     zero_prefix = "0"
                # else:
                #     zero_prefix = ""
                
                rgb_path = os.path.join(episode_dir, "episode-" + episode_number + "-" + f"{j+1:03d}" + ".png")
                # rgb_path = os.path.join(episode_dir, "episode-" + episode_number + "-" + zero_prefix + sample_number+ ".png")

                rgb_image = torchvision.io.read_image(rgb_path)
                rgbs.append(rgb_image)
            actions = actions + episode_actions # Changed - by Sandra 10/04 - 7:13pm
        
        # Preprocessing rgbs
        rgbs = torch.stack(rgbs).float() / 255
        rgb_mean = torch.mean(rgbs, dim=(0, 2, 3))

        # Modified by Sandra - 20230413 - to ensure data does not contain nan or infinite values
        epsilon = 1e-6
        rgb_std = torch.std(rgbs, dim=(0, 2, 3)) + epsilon
        # # rgb_std = torch.std(rgbs, dim=(0, 2, 3))
        # rgbs[:,0,:,:] = (rgbs[:,0,:,:] - rgb_mean[0]) / rgb_std[0]
        # rgbs[:,1,:,:] = (rgbs[:,1,:,:] - rgb_mean[0]) / rgb_std[1]
        # rgbs[:,2,:,:] = (rgbs[:,2,:,:] - rgb_mean[0]) / rgb_std[2]
        #-------
        
        return rgb_std, rgb_mean


if __name__ == "__main__":
    # rgb_std = [142, 156, 78]
    # rgb_mean = [145, 153, 80]

    data_dir = "data\simulated-samples"
    data_loader = DataLoader(data_dir=data_dir, episode_list=list(range(7)), samples=499)
    rgb_std, rgb_mean = data_loader.rgb_std, data_loader.rgb_mean


    model_name = "model_ep-50_lr-1e-05_bs-4"
    rewards = run_model_on_environment(model_name + '.pth', rgb_std, rgb_mean)