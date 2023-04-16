import datetime
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from modeling import VRNet, DataLoader

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


def train(data_loader, num_epochs:int, learning_rate:float, batch_size:int, model_name:str, optimizer:str="adam", loss_function:str="bce"):
    OPTIMIZERS = ["adam", "sgd"]
    LOSS_FUNCTIONS = [ "bce", "l1", "l2", "lc", "lg" ]
    assert(optimizer in OPTIMIZERS)
    assert(loss_function in LOSS_FUNCTIONS)
    
    # Initialize history dictionary
    history = {}
    history["config"] = {
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer, 
        "batch_size": batch_size
    }
    
    # Initialise model
    model = VRNet()

    # Set GPU if available
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda:0")
        model.to(device)
    else:
        print("Using CPU: WARNING - this will take a long time!")
        device = torch.device("cpu")

    model.train() 
    
    # Sets optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Sets loss function
    if loss_function == "bce":
        loss_function = nn.BCEWithLogitsLoss()
    elif loss_function == "l1":
        loss_function = nn.L1Loss()
    elif loss_function == "l2":
        loss_function = nn.MSELoss()
    elif loss_function == "lc":
        # Lc loss encourages directional alignment between output and target
        # should be used if the direction is more important than the magnitude
        loss_function = LcLoss()
    elif loss_function == "lg":
        # Sigmoid cross entropy loss that can be used to train for outputs like {0,1}
        # was known as Lg loss function in code and paper
        # L_g_loss = nn.CrossEntropyLoss() 
        loss_function = nn.CrossEntropyLoss()

    history["loss"] = []
    for epoch in tqdm(range(num_epochs)):
        for i in range(len(data_loader)):
            rgb_img, action = data_loader[i]
            
            # get rgb images
            rgb_img = rgb_img.to(device).float()
            
            #apply data augmentation
            # rgb_img, depth_img = applyTransforms(rgb_img, depth_img)
            
            #add batch dimension
            target_action = action.to(device).float()

            optimizer.zero_grad()
            
            predicted_action = model(rgb_img)
            
            # print(f"predicted_action shape: {predicted_action.shape}")
            # print(f"target_action shape: {target_action.shape}")

            #calculate combined loss
            # loss = L1_loss(output[0:3], state[0:3]) * loss_weights[0]

            #combine 0:3 and 6
            # important_output = torch.cat((output[:, 0:3], output[:, 6].unsqueeze(1)), dim=1)
            # important_state = torch.cat((state[:, 0:3], state[:, 6].unsqueeze(1)), dim=1)

            # loss = L1_loss(predicted_action, target_action)
            # loss += L2_loss(predicted_action, target_action) # comment if you don't want this.

            # loss = L1_loss(predicted_action, target_action) # * loss_weights[1]
            # loss += L_c_loss(output[:, 0:6], state[:, 0:6]) * loss_weights[2]
            # loss += L_g_loss(output[:, 6], state[:, 6]) * loss_weights[3]
            loss = loss_function(predicted_action, target_action)
            
            loss.backward()
            optimizer.step()
        
        history['loss'].append(loss.item())
        print(f'Epoch: {epoch+1}/{num_epochs}, Iteration: {i}, Loss: {loss.item()}')
    
    # Save model
    torch.save(model.state_dict(), model_name+'.pth')

    return history

# END - Sandra - 10/04 7:45pm

if __name__ == "__main__":

    data_dir = "data\simulated-samples"
    data_loader = DataLoader(data_dir=data_dir, episode_list=list(range(5)), samples=499)
    
    model_name = "model_ep-50_lr-1e-05_bs-4"
    # Train the model and save the training history (loss info)
    model_history = train(data_loader, num_epochs=50, learning_rate=1e-05, batch_size=4, model_name=model_name)
    
    # Pickle model history
    with open(model_name + '.pkl', 'wb') as f:
        pickle.dump(model_history, f)
    
    