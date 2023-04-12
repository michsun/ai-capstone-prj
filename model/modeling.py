import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib as plt
import pandas as pd


## SpatialSoftmax implementation taken from https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = torch.nn.Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)
        
        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

#Implementation of Network from Figure 3 (on pg 4) of paper
class VRNet(nn.Module):
    def __init__(self):
        super(VRNet, self).__init__()
        # Convolution 1 160x120x3 -> 77x57x64
        self.conv1_rgb = nn.Conv2d(3, 64, 7, padding='valid', stride=2)

        # Convolution 2 77x57x64 -> 77x57x32
        self.conv2 = nn.Conv2d(64, 32, 1, padding='same')
        self.conv2_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        # Convolution 3 77x57x32 -> 75x55x32
        self.conv3 = nn.Conv2d(32, 32, 3, padding='valid')
        self.conv3_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        # Convolution 4 75x55x32 -> 73x53x32
        self.conv4 = nn.Conv2d(32, 32, 3, padding='valid')
        self.conv4_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)

        # spatial softmax
        # self.spatialSoftmax = SpatialSoftmax(53, 73, 32, temperature=1, data_format='NCHW')
        self.spatialSoftmax = SpatialSoftmax(193, 293, 32, temperature=1, data_format='NCHW')

        self.flatten = nn.Flatten()

        #fully connected layers
        self.fc1 = nn.Linear(64, 50)
        self.fc1_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc3 = nn.Linear(50, 50)
        self.fc3_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        # self.fc4 = nn.Linear(50, 7) # Vx, Vy, Vz, Wx, Wy, Wz, grabber open
        self.fc4 = nn.Linear(50, 1) 

        #set conv1_rgb weights to be first layer from pretrained model
        googlenet = torchvision.models.googlenet(pretrained=True)
        self.conv1_rgb.weight.data = googlenet.conv1.conv.weight.data

        #weights should be uniformly sampled from [-0.1, 0.1]
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.conv4.weight.data.uniform_(-0.1, 0.1)

        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4.weight.data.uniform_(-0.1, 0.1)

    def forward(self, rgbImg):
        #conv layers
        x = F.relu(self.conv1_rgb(rgbImg))

        #implement convulutional layers with batch normalization
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  

        x = self.spatialSoftmax(x)
        x = self.flatten(x)

        #fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DataLoader(Dataset):

    def __init__(self, data_dir: str, episodes: int, samples: int, batch_size=1):
        self.data_dir = data_dir
        # self.startRun = startRun
        # self.lastRun = lastRun
        self.episodes = episodes # number of episodes
        self.samples = samples # number of samples in each episode
        self.batch_size = batch_size
        self.rgb_images, self.actions = self.load_data() # Changed - Sandra 10/04 - 7:50pm states to actions
        self.arrayIndicies = list([i for i in range(len(self.rgb_images))])
        print(len(self.rgb_images), len(self.actions))
        assert(len(self.rgb_images) == len(self.actions))
    
    # train_data_loader function - by Sandra
    # Output: rgbs list and action list

    def load_data(self):
        """Loads data from the data directory and returns a list of rgb images and a list of actions"""
        rgbs = []
        actions = []
        
        for i in range(self.episodes):
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
        
        #compute std
        rgb_std = torch.std(rgbs, dim=(0, 2, 3))
        #normalize images
        rgbs[:,0,:,:] = (rgbs[:,0,:,:] - rgb_mean[0]) / rgb_std[0]
        rgbs[:,1,:,:] = (rgbs[:,1,:,:] - rgb_mean[0]) / rgb_std[1]
        rgbs[:,2,:,:] = (rgbs[:,2,:,:] - rgb_mean[0]) / rgb_std[2]

        return rgbs, actions
    
    def __len__(self):
        return len(self.actions) // self.batch_size # changed - Sandra - states to actions

    def __getitem__(self, idx):
        #shuffle array index mapping
        # if idx == 0:
        np.random.shuffle(self.arrayIndicies)
            
        idx = idx * self.batch_size
        desiredIndexes = self.arrayIndicies[idx:idx+self.batch_size]

        rgb_img = []
        action = []
        
        for i in desiredIndexes:
            rgb_img.append(self.rgb_images[i])
            action.append(self.actions[i])

        rgb_img = torch.stack(rgb_img)
        # action = torch.stack(action)
        action = torch.FloatTensor(action)
        # action = torch.tensor(action)  
        action = action.view(1, -1) 

        return rgb_img, action
    
class DataPreprocessor():
    def __init__(self, rgb_mean, rgb_std):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def normalizeRgb(self, rgb):
        rgb[:,0,:,:] = (rgb[:,0,:,:] - self.rgb_mean[0]) / self.rgb_std[0]
        rgb[:,1,:,:] = (rgb[:,1,:,:] - self.rgb_mean[1]) / self.rgb_std[1]
        rgb[:,2,:,:] = (rgb[:,2,:,:] - self.rgb_mean[2]) / self.rgb_std[2]
        return rgb

if __name__ == "__main__":
    # Test Data Loader
    data_dir = "data\simulated-samples"
    data_loader = DataLoader(data_dir = data_dir, 
                             episodes = 5, 
                             samples = 499
    )
    
