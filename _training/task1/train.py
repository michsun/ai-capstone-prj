import datetime
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
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


def train(train_data_loader, val_data_loader, num_epochs:int, learning_rate:float, batch_size:int,
          optimizer:str="adam", loss_function:str="bce",
          save_path:str="models/", model_name:str="model"):
    """
    Trains the model for num_epochs epochs, using the given learning rate, batch size, optimizer and loss function.
    Saves the model after each epoch, and saves the training and validation loss and accuracy after each epoch.

    Args:
        train_data_loader: DataLoader object for the training data
        val_data_loader: DataLoader object for the validation data
        num_epochs: number of epochs to train for
        learning_rate: learning rate to use for training
        batch_size: batch size to use for training
        model_name: name of the model to save
        optimizer: optimizer to use for training
        loss_function: loss function to use for training
    
    Returns:
        history: dictionary containing the training and validation loss and accuracy for each epoch
    """
    OPTIMIZERS = ["adam", "sgd"]
    LOSS_FUNCTIONS = [ "bce", "l1", "l2", "lc", "lg" ]
    assert(optimizer in OPTIMIZERS)
    assert(loss_function in LOSS_FUNCTIONS)
    
    model_name = f"{model_name}_ep-{num_epochs}_lr-{learning_rate}_bs-{batch_size}_opt-{optimizer}_loss-{loss_function}"
    
    # Initialize history dictionary
    history = {}
    history["config"] = {
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer, 
        "batch_size": batch_size,
        "loss_function": loss_function
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
    
    # Reset the data loader
    train_data_loader.reset()
    val_data_loader.reset()

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
            
            train_loss = loss_function(predicted_action, target_action)
            
            train_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        
        train_acc = get_accuracy(model, train_data_loader, device)
        
        # Validation loop
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            val_acc = get_accuracy(model, val_data_loader, device)
        
            # Calculate validation loss
            val_loss = 0
            for i in range(len(val_data_loader)):
                rgb_img, action = val_data_loader[i]
                rgb_img = rgb_img.to(device).float()
                target_action = action.to(device).float()
                output = model(rgb_img)
                val_loss += loss_function(output, target_action)
                del rgb_img, output, target_action
            val_loss /= len(val_data_loader)

        model.train()  # Set model back to training mode
        
        # Save history
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss.item())
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss.item())
        
        # Print progress
        output = f'Epoch: {epoch+1}/{num_epochs}, train_acc: {train_acc:4f}, train_loss: {train_loss.item():4f}'
        output += f', val_acc: {val_acc:4f}, val_loss: {val_loss.item():4f}'
        print(output)

    # Save the model
    save_loc = os.path.join(save_path, model_name+'.pth')
    torch.save(model.state_dict(), save_loc)

    # Save the history
    save_loc = os.path.join(save_path, model_name+'.pkl')
    with open(save_loc, 'wb') as f: 
        pickle.dump(history, f)

    return history

# END - Sandra - 10/04 7:45pm

def train_individual_model():
    EPOCHS = 50
    LEARNING_RATE = 5e-06
    BATCH_SIZE = 4
    OPTIMIZER = "adam"
    LOSS_FUNCTION = "bce"
    
    data_dir = "data\simulated-samples"  # task 1
    train_data_loader = DataLoader(data_dir=data_dir, episode_list=list(range(5)), samples=499)
    val_data_loader = DataLoader(data_dir=data_dir, episode_list=[5,6], samples=499)

    # train_data_loader, val_data_loader = split_data_loader(data_loader, val_split=0.2)
    
    print(len(train_data_loader), len(val_data_loader))

    # Train the model and save the training history (loss info)
    model_history = train(train_data_loader, val_data_loader,
                          num_epochs=EPOCHS, 
                          learning_rate=LEARNING_RATE, 
                          batch_size=BATCH_SIZE,
                          optimizer=OPTIMIZER,
                          loss_function=LOSS_FUNCTION,
                          save_path="task1_models",
                          model_name="task1_model"
    )


if __name__ == "__main__":
    
    # Load the data
    data_dir = "data\simulated-samples"  # task 1
    
    # train_data_loader = DataLoader(data_dir=data_dir, episode_list=list(range(5)), samples=499)
    # val_data_loader = DataLoader(data_dir=data_dir, episode_list=[5,6], samples=499)
    
    data_loader = DataLoader(data_dir=data_dir, episode_list=[0,1,2,3,4,5,6], samples=499)
    print("Total images: ", len(data_loader))
    
    train_data_loader, val_data_loader = split_data_loader(data_loader, val_split=0.2)
    print("Train images: ", len(train_data_loader))
    print("Val images  : ", len(val_data_loader))

    # Set parameters
    save_path = "task1/models"
    model_name = "task1"

    param_grid = {
        "num_epochs": [50, 100],
        "learning_rate": [1e-05, 5e-05, 1e-04, 5e-04, 1e-03],
        "batch_size": [4, 8, 16, 32, 64],
        "optimizer": ["adam", "sgd"],
        "loss_function": ["bce"]
    }

    # Create a ParameterGrid object
    grid = ParameterGrid(param_grid)

    best_val_loss = np.inf
    best_params = None

    # Iterate over all possible hyperparameter combinations
    for params in grid:
        epochs = params["num_epochs"]
        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]
        optimizer = params["optimizer"]
        loss_function = params["loss_function"]
        
        history = train(train_data_loader, val_data_loader, 
                        num_epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        save_path=save_path,
                        model_name=model_name
                        )
        
        min_val_loss = np.min(history["val_loss"])

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_params = params
    
    # Print results
    print("Best validation loss:", best_val_loss)
    print("Best parameters:", best_params)
    
    best_model_name = f"{model_name}_best_model_{best_params['num_epochs']}_{best_params['learning_rate']}_{best_params['batch_size']}_{best_params['optimizer']}_{best_params['loss_function']}"
    best_model_path = os.path.join(save_path, best_model_name+".pth")
    
    print("Best model path:", best_model_path)
    
    