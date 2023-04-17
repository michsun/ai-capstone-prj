import datetime
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from modeling import VRNet, DataLoader, split_data_loader

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


def get_accuracy(model, data_loader, device):

    num_of_predictions = 0
    correct_number_of_predictions = 0

    for i in range(len(data_loader)):
        rgb_img, action = data_loader[i]

        rgb_img = rgb_img.to(device).float()
        target_action = action.to(device).float()
        
        output = model(rgb_img)
        # Apply sigmoid activation
        predicted_probs = torch.sigmoid(output)
        predicted_action = (predicted_probs > 0.5).float()

        if torch.equal(predicted_action, target_action):
            correct_number_of_predictions += 1
    
        num_of_predictions += 1

    accuracy = correct_number_of_predictions / num_of_predictions
    return accuracy


def train(train_data_loader, val_data_loader, num_epochs:int, learning_rate:float, batch_size:int, model_name:str, 
          optimizer:str="adam", loss_function:str="bce"):
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
    
    history["train_acc"], history["train_loss"], history["val_acc"], history["val_loss"] = [], [], [], []
    for epoch in tqdm(range(num_epochs)):
        for i in range(len(train_data_loader)):
            rgb_img, action = train_data_loader[i]
            
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
            train_loss = loss_function(predicted_action, target_action)
            
            train_loss.backward()
            optimizer.step()
        
        train_acc = get_accuracy(model, train_data_loader, device)

        # val_acc = get_accuracy(model, val_data_loader, device)
        
        # Calculate validation loss
        # val_loss = 0
        # for i in range(len(val_data_loader)):
        #     rgb_img, action = val_data_loader[i]
        #     rgb_img = rgb_img.to(device).float()
        #     target_action = action.to(device).float()
        #     output = model(rgb_img)
        #     val_loss += loss_function(output, target_action)
        #     del rgb_img, output, target_action
        # val_loss /= len(val_data_loader)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss.item())
        # history['val_acc'].append(val_acc)
        # history['val_loss'].append(val_loss.item())

        print(f'Epoch: {epoch+1}/{num_epochs}, train_acc: {train_acc}, train_loss: {train_loss.item()}')

    # Save model
    torch.save(model.state_dict(), model_name+'.pth')

    return history

# END - Sandra - 10/04 7:45pm

if __name__ == "__main__":

    
    EPOCHS = 50
    LEARNING_RATE = 5e-06
    BATCH_SIZE = 4
    OPTIMIZER = "adam"
    LOSS_FUNCTION = "bce"
    
    model_name = f"model_ep-{EPOCHS}_lr-{str(LEARNING_RATE)}_bs-{BATCH_SIZE}_opt-{OPTIMIZER}_loss-{LOSS_FUNCTION}"
    
    data_dir = "data\simulated-samples"
    # data_loader = DataLoader(data_dir=data_dir, episodes=7, samples=499)
    train_data_loader = DataLoader(data_dir=data_dir, episode_list=list(range(7)), samples=499)
    val_data_loader = DataLoader(data_dir=data_dir, episode_list=[5,6], samples=499)

    # train_data_loader, val_data_loader = split_data_loader(data_loader, val_split=0.2)
    
    # Delete data loader to free up memory
    # del data_loader
    
    print(len(train_data_loader), len(val_data_loader))

    # Train the model and save the training history (loss info)
    model_history = train(train_data_loader, val_data_loader,
                          num_epochs=EPOCHS, 
                          learning_rate=LEARNING_RATE, 
                          batch_size=BATCH_SIZE,
                          optimizer=OPTIMIZER,
                          loss_function=LOSS_FUNCTION,
                          model_name=model_name)
    
    # Pickle model history
    with open(model_name + '.pkl', 'wb') as f:
        pickle.dump(model_history, f)
    
    