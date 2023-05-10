import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models.googlenet import GoogLeNet_Weights


from typing import List

SEED = 101


def conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """Calculates the output size of a convolutional layer"""
    output_size = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return output_size


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


# Modified from VRNet implementation from https://github.com/NathanMalta/VRImitationLearning/blob/main/VRNet.py
class CustomNet(nn.Module):
    def __init__(self, fc4_out_dim: int = 1, spatialSoftmax_dim: tuple = (193, 293)):
        # Task 1: fc4_out_dim = 1, spatialSoftmax_dim = (193, 293)
        # Task 2: fc4_out_dim = 9, spatialSoftmax_dim = (98, 73)
        super(CustomNet, self).__init__()
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
        # self.spatialSoftmax = SpatialSoftmax(193, 293, 32, temperature=1, data_format='NCHW')  # task 1
        # self.spatialSoftmax = SpatialSoftmax(98, 73, 32, temperature=1, data_format='NCHW')  # task 2
        self.spatialSoftmax = SpatialSoftmax(spatialSoftmax_dim[0], spatialSoftmax_dim[1], 32, temperature=1, data_format='NCHW')

        self.flatten = nn.Flatten()

        #fully connected layers
        self.fc1 = nn.Linear(64, 50)
        self.fc1_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc3 = nn.Linear(50, 50)
        self.fc3_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        # self.fc4 = nn.Linear(50, 7) # Vx, Vy, Vz, Wx, Wy, Wz, grabber open
        self.fc4 = nn.Linear(50, fc4_out_dim)  

        #set conv1_rgb weights to be first layer from pretrained model
        # googlenet = torchvision.models.googlenet(pretrained=True)
        googlenet = torchvision.models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        # googlenet = torchvision.models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
        self.conv1_rgb.weight.data = googlenet.conv1.conv.weight.data

        #weights should be uniformly sampled from [-0.1, 0.1]
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.conv4.weight.data.uniform_(-0.1, 0.1)

        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4.weight.data.uniform_(-0.1, 0.1)


    # ADDED by Sandra # 20230314 - uses the batch normalization:
    def forward(self, rgbImg):

        # Convolution 1 160x120x3 -> 77x57x64
        x = self.conv1_rgb(rgbImg)
        x = F.relu(x)

        # print_output("conv1_rgb_shape: " + str(x.shape))
        # print("conv1_rgb_shape:", x.shape)

        # Convolution 2 77x57x64 -> 77x57x32
        # x = self.conv2_bn(self.conv2(x))
        # x = F.relu(self.conv2_bn(x))
        x = self.conv2(x)
        # print_output("conv2_shape: " + str(x.shape))
        x = F.relu(self.conv2_bn(x))
        # print_output("conv2_bn_shape: " + str(x.shape))

        # Convolution 3 77x57x32 -> 75x55x32
        x = self.conv3(x)
        # print_output("conv3_shape: " + str(x.shape))
        x = F.relu(self.conv3_bn(x))
        # print_output("conv3_bn_shape: " + str(x.shape))

        # Convolution 4 75x55x32 -> 73x53x32
        # x = self.conv4(x)
        # x = F.relu(self.conv4_bn(x))
        x = self.conv4(x)
        # print_output("conv4_shape: " + str(x.shape))
        x = F.relu(self.conv4_bn(x))
        # print_output("conv4_bn_shape:"+ str(x.shape))

        # spatial softmax
        x = self.spatialSoftmax(x)
        # print_output("spatialSoftmax_shape:"+ str(x.shape))
        

        x = self.flatten(x)
        # print_output("flatten_shape:" + str(x.shape))

        # Fully connected layers
        x = F.relu(self.fc1(x))
        # print_output("fc1_shape:" + str(x.shape))
        # x = F.relu(self.fc1_bn(x))
        # print_output("fc1_bn_shape:" + x.shape)
        x = F.relu(self.fc2(x))
        # print_output("fc2_shape:" + str(x.shape))
        # x = F.relu(self.fc2_bn(x))
        # print_output("fc2_bn_shape:" + str(x.shape))
        x = F.relu(self.fc3(x))
        # print_output("fc3_shape:" + str(x.shape))
        # x = F.relu(self.fc3_bn(x))
        # print_output("fc3_bn_shape:" + str(x.shape))

        # Output layer
        x = self.fc4(x)
        # print_output("fc4_shape:" + str(x.shape))

        return x


def print_output(string):
    with open("output.txt", "a") as f:
        f.write(string + '\n')






