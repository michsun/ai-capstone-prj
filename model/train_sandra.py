from model_sandra import VRNet, VRDataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Sandra - 10/04 7:45pm

# create custom Lc loss function
class LcLoss(nn.Module):
    def __init__(self):
        super(LcLoss, self).__init__()

    def forward(self, pred, target):
        loss = torch.zeros(pred.shape[0])
        for i in range(pred.shape[0]):
            p = pred[i]
            t = target[i]
            loss[i] = torch.arccos(torch.dot(t, p) / (torch.norm(t) * torch.norm(p)))
        
        return torch.sum(loss)

def data_preprocessing():
    return

def train(data_loader, num_epochs, learning_rate):
    
    model = VRNet()
    model.train()
    # TODO: Add train details here
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    L1_loss = nn.L1Loss()
    L2_loss = nn.MSELoss()
    # Lc loss encourages directional alignment between output and target
    # should be used if the direction is more important than the magnitude
    L_c_loss = LcLoss()
    # Sigmoid cross entropy loss that can be used to train for outputs like {0,1}
    # was known as Lg loss function in code and paper
    L_g_loss = nn.CrossEntropyLoss() 
    for epoch in range(num_epochs):
        for i in range(len(data_loader)):
            rgb_img, action = data_loader[i]
            # rgb_img = rgb_img


            #get rgb and depth images
            rgb_img = rgb_img.to(device).float()
            # depth_img = depth_img.to(device).float()

            
            #apply data augmentation
            # rgb_img, depth_img = applyTransforms(rgb_img, depth_img)
            
            #add batch dimension
            target_action = action.to(device).float()

            optimizer.zero_grad()
            # predicted_action = model(rgb_img, depth_img)
            predicted_action = model(rgb_img)
            
            #calculate combined loss
            # loss = L1_loss(output[0:3], state[0:3]) * loss_weights[0]

            #combine 0:3 and 6
            # important_output = torch.cat((output[:, 0:3], output[:, 6].unsqueeze(1)), dim=1)
            # important_state = torch.cat((state[:, 0:3], state[:, 6].unsqueeze(1)), dim=1)
            

            loss = L1_loss(predicted_action, target_action)
            loss += L2_loss(predicted_action, target_action) # comment if you don't want this.


            # loss = L1_loss(predicted_action, target_action) # * loss_weights[1]
            # loss += L_c_loss(output[:, 0:6], state[:, 0:6]) * loss_weights[2]
            # loss += L_g_loss(output[:, 6], state[:, 6]) * loss_weights[3]
            

            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')

# END - Sandra - 10/04 7:45pm

if __name__ == "__main__":
    data_dir = "../data/simulated-examples"
    rgbs, actions = VRDataLoader(data_dir=data_dir).load_data()
    print(actions)
    # train()