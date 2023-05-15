import torch
import torchvision
from torch.utils.data import Dataset
import os
import pandas as pd

from typing import List


class CustomDataset(Dataset):

    def __init__(self, data_dir: str, episode_list: List, samples: int or List = None, indices=None):
        self.data_dir = data_dir
        self.episode_list = episode_list
    
        self.samples = samples
        if type(samples) == int:
            self.samples = [samples] * len(episode_list)
        elif type(samples) == list:
            assert len(samples) == len(episode_list), \
                "The number of samples must be equal to the number of episodes"
            
        self.rgb_images, self.actions, self.rgb_std, self.rgb_mean = self.load_data()
        assert(len(self.rgb_images) == len(self.actions))

        self.indices = indices if indices is not None else list(range(len(self.rgb_images)))
    

    def load_data(self):
        """Loads data from the data directory and returns a list of rgb images and a list of actions"""
        rgbs = []
        actions = []
        
        print(f"Loading datataset {self.data_dir}...")
        for i in self.episode_list:
            episode_number = str(i)

            episode_dir = os.path.join(self.data_dir, "episode-" + episode_number)
            episode_info_df = pd.read_csv(episode_dir + "/episode-" + episode_number +".csv")

            if self.samples is None:
                self.samples = [ len(episode_info_df)]
            elif len(self.samples) < len(self.episode_list):
                self.samples = self.samples + [ len(episode_info_df)]
            print(f"\tepisode {episode_number} - samples {self.samples[i]}")
            episode_actions = episode_info_df["Action"].iloc[0:self.samples[i]].tolist() # Changed - by Sandra 10/04 - 7:13pm
            
            for j in range(self.samples[i]):
                # sample_number = str(j + 1)
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
        
        # #compute std
        # rgb_std = torch.std(rgbs, dim=(0, 2, 3))
        # #normalize images
        # rgbs[:,0,:,:] = (rgbs[:,0,:,:] - rgb_mean[0]) / rgb_std[0]
        # rgbs[:,1,:,:] = (rgbs[:,1,:,:] - rgb_mean[0]) / rgb_std[1]
        # rgbs[:,2,:,:] = (rgbs[:,2,:,:] - rgb_mean[0]) / rgb_std[2]

        # Modified by Sandra - 20230413 - to ensure data does not contain nan or infinite values
        epsilon = 1e-6
        rgb_std = torch.std(rgbs, dim=(0, 2, 3)) + epsilon
        # rgb_std = torch.std(rgbs, dim=(0, 2, 3))
        rgbs[:,0,:,:] = (rgbs[:,0,:,:] - rgb_mean[0]) / rgb_std[0]
        rgbs[:,1,:,:] = (rgbs[:,1,:,:] - rgb_mean[1]) / rgb_std[1]
        rgbs[:,2,:,:] = (rgbs[:,2,:,:] - rgb_mean[2]) / rgb_std[2]
        #-------
        
        print("Dataset loaded.")
        
        return rgbs, actions, rgb_std, rgb_mean
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
            
        # idx = idx * self.batch_size
        # desiredIndexes = self.indices[idx:idx+self.batch_size]

        # rgb_img = []
        # action = []
        # 
        # for i in desiredIndexes:
            # rgb_img.append(self.rgb_images[i])
            # action.append(self.actions[i])

        # rgb_img = torch.stack(rgb_img)
        # action = torch.stack(action)
        # action = torch.FloatTensor(action)
        # action = torch.tensor(action)  
        # action = action.view(1, -1) 

        desiredIndex = self.indices[idx]
        
        rgb_img = self.rgb_images[desiredIndex]
        action = self.actions[desiredIndex]

        return rgb_img, action

    
# def train_val_split(dataset: CustomDataset, val_split: float):
#     """Splits the CustomDataset into a training set and a validation set"""
#     train_split = int((1 - val_split) * len(data_loader))
#     indices = np.arange(len(data_loader))
#     np.random.seed(SEED)
#     np.random.shuffle(indices)
#     train_indices = indices[:train_split]
#     val_indices = indices[train_split:]
#     print("Split data loader:", len(train_indices), len(val_indices))

#     # Deep copy data loader
#     train_data_loader = CustomDataset(dataset.data_dir, dataset.episode_list, dataset.samples, indices=train_indices)
    
#     # print("Object address:", id(data_loader), id(train_data_loader))
#     val_data_loader.arrayIndices = val_indices
    
#     return train_data_loader, val_data_loader


def print_output(string):
    with open("output.txt", "a") as f:
        f.write(string + '\n')


if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split

    data_dir = r"data\battlezone-sample"
    episode_list = [0, 1]

    data_dir = r"data\simulated-samples"
    episode_list = [0, 1, 2, 3, 4, 5, 6]
    
    ##############################################

    # Load dataset
    transformed_dataset = CustomDataset(data_dir, episode_list)
    print("Dataset size:", len(transformed_dataset))

    # Split dataset
    train_size = int(0.8 * len(transformed_dataset))
    val_size = len(transformed_dataset) - train_size
    print("Train size:", train_size)
    print("Val size  :", val_size)
    train_dataset, val_dataset = random_split(transformed_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print("Train loader:", len(train_loader))
    print("Val loader  :", len(val_loader))
    
    # Iterate through the dataset
    for i, (rgb_img, action) in enumerate(train_loader):
        print(f"Batch {i}:", rgb_img.shape, action.shape)


