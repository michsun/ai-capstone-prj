import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from timeit import default_timer as timer
from datetime import timedelta
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CustomDataset
from network import CustomNet

# Set seed for reproducibility - for DataLoader shuffle
torch.manual_seed(0)

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


def save_best_model(metric: str, previous_val: float, previous_model_path: str, curr_vals: tuple[float, float, float, float], 
                    curr_model: CustomNet, epoch:int, delete_previous_best:bool = False, **kwargs):
    
    metric_map = { "train_acc": 0, "train_loss": 1, "val_acc": 2, "val_loss": 3 }
    curr_val = curr_vals[metric_map[metric]]
    
    if (metric == "val_loss" or metric == "train_loss") and curr_val > previous_val:
        return previous_val, previous_model_path
    elif (metric == "train_acc" or metric == "val_acc") and curr_val < previous_val:
        return previous_val, previous_model_path

    if delete_previous_best and previous_model_path is not None:
        os.remove(previous_model_path)
    new_best_path = os.path.join(kwargs.get('save_path'), 
                                 f"{kwargs.get('model_name')}_best-{epoch}_{metric}-{curr_val:.4f}.pth")
    print(f"Saving new best model to {new_best_path}")
    torch.save(curr_model.state_dict(), new_best_path)

    return curr_val, new_best_path


def get_accuracy(model: CustomNet, data_loader: DataLoader, device: torch.device, type: str):

    def binary_calc(pred_output):
        # Apply sigmoid activation
        pred_output = torch.sigmoid(pred_output)
        pred_action = (pred_output > 0.5).float()
        return pred_action
    
    def multi_calc(pred_output):
        # Apply softmax activation
        pred_output = torch.softmax(pred_output, dim = 1)
        pred_action = torch.argmax(pred_output, dim = 1)
        return pred_action
    
    calc_prediction_map = {
        "binary": binary_calc, 
        "multi-class": multi_calc
    }
    assert type in calc_prediction_map.keys(), f"Invalid type {type} - must be one of {calc_prediction_map.keys()}"
    
    num_of_predictions = 0
    correct_number_of_predictions = 0
    
    model.eval()
    with torch.no_grad():
        for rgb_img, action in data_loader:

            batch_size = rgb_img.size(0)
            rgb_img = rgb_img.to(device).float()

            if type == "binary":
                target_action = action.to(device).view(-1,1).float()
            elif type == "multi-class":
                target_action = action.to(device).view(batch_size).long()

            output = model(rgb_img)

            predicted_action = calc_prediction_map[type](output)
            
            correct_number_of_predictions += (predicted_action == target_action).sum().item()
            
            num_of_predictions += batch_size
            

    accuracy = correct_number_of_predictions / num_of_predictions

    model.train()
    return accuracy


def train(train_dataset: CustomDataset, val_dataset: CustomDataset,
          num_epochs:int, learning_rate:float, batch_size:int,
          classification_type:str, 
          optimizer:str="adam", loss_function:str="bce",
          save_best:str=None, 
          save_path:str="models/", model_name:str="model",
          CustomNet_kwargs:dict={}):
    """
    Trains the model for num_epochs epochs, using the given learning rate, batch size, optimizer and loss function.
    Saves the model after each epoch, and saves the training and validation loss and accuracy after each epoch.

    Args:
        train_dataset (CustomDataset): training dataset
        val_dataset (CustomDataset): validation dataset
        num_epochs (int): number of epochs to train for
        learning_rate (float): learning rate for optimizer
        batch_size (int): batch size for training
        classification_type (str): classification type - either "binary" or "multi-class"
        optimizer (str, optional): optimizer to use - either "adam" or "sgd". Defaults to "adam".
        loss_function (str, optional): loss function to use - either "bce", "l1", "l2", "lc" or "lg". Defaults to "bce".
        save_best (str, optional): whether to save the best model based on training or validation loss. Defaults to "val_loss".
        save_path (str, optional): path to save models to. Defaults to "models/".
        model_name (str, optional): name of model. Defaults to "model".
        CustomNet_kwargs (dict, optional): keyword arguments for CustomNet. Defaults to {}.
    
    Returns:
        history: dictionary containing the training and validation loss and accuracy for each epoch
    """
    # Check valid inputs
    OPTIMIZERS = ["adam", "sgd"]
    LOSS_FUNCTIONS = [ "bce", "l1", "l2", "lc", "lg" ]
    CLASSIFICATION_TYPES = ["binary", "multi-class"]
    SAVE_BEST = ["val_loss", "train_loss", "val_acc", "train_acc", None]
    assert optimizer in OPTIMIZERS, f"Invalid optimizer {optimizer} - must be one of {OPTIMIZERS}"
    assert loss_function in LOSS_FUNCTIONS, f"Invalid loss function {loss_function} - must be one of {LOSS_FUNCTIONS}"
    assert classification_type in CLASSIFICATION_TYPES, f"Invalid classification type {classification_type} - must be one of {CLASSIFICATION_TYPES}"
    assert save_best in SAVE_BEST, f"Invalid save_best {save_best} - must be one of {SAVE_BEST}"

    # Check if save_loc exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set model name
    model_name = f"{model_name}_ep-{num_epochs}_lr-{learning_rate}_bs-{batch_size}_opt-{optimizer}_loss-{loss_function}"

    # Initialize history dictionary
    history = {}
    history["config"] = {
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer, 
        "batch_size": batch_size,
        "loss_function": loss_function,
        "classification_type": classification_type,
        "save_path": save_path,
        "save_best": save_best,
        "model_name": model_name,
        "CustomNet_kwargs": CustomNet_kwargs
    }
    
    # Print
    print("\n=====================Starting training=====================")
    timer_start = timer()
    print(f"Train config: {history['config']}")
    
    # Initialise model
    model = CustomNet(**CustomNet_kwargs)

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

    # Initialise the data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Reset shuffle seed
    torch.manual_seed(torch.initial_seed())

    history["train_acc"], history["train_loss"], history["val_acc"], history["val_loss"] = [], [], [], []
    
    # Initialise best model
    if save_best:
        best_model_path = None
        best_model_val = float("inf") if save_best == "val_loss" or save_best == "val_acc" else 0        

    print("Starting training...")

    for epoch in tqdm(range(num_epochs)):
        
        for rgb_img, action in train_data_loader:
            
            # Set curr batch size
            curr_batch_size = rgb_img.size(0)
            
            # Move data to GPU if available
            rgb_img = rgb_img.to(device).float()

            if classification_type == "binary":
                target_action = action.to(device).view(-1,1).float()
            elif classification_type == "multi-class":
                target_action = action.to(device).view(curr_batch_size).long()

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_action = model(rgb_img)

            # Calculate loss
            train_loss = loss_function(predicted_action, target_action)

            # Backward pass
            train_loss.backward()
            # Update weights
            optimizer.step()
            # Clear cache
            torch.cuda.empty_cache()
        
        train_acc = get_accuracy(model, train_data_loader, device, type=classification_type)
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Get validation accuracy
            val_acc = get_accuracy(model, val_data_loader, device, type=classification_type)

            # Calculate validation loss
            val_loss = 0
            for rgb_img, action in val_data_loader:
                curr_batch_size = rgb_img.size(0)
                rgb_img = rgb_img.to(device).float()
                if classification_type == "binary":
                    action = action.to(device).view(-1,1).float()
                elif classification_type == "multi-class":
                    action = action.to(device).view(curr_batch_size).long()
                output = model(rgb_img)
                val_loss += loss_function(output, action)

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

        # Saves best model
        if save_best:
            best_model_val, best_model_path = save_best_model(metric=save_best,
                                                            previous_val=best_model_val,
                                                            previous_model_path=best_model_path,
                                                            curr_vals=(train_acc, train_loss, val_acc, val_loss),
                                                            curr_model=model,
                                                            epoch=epoch,
                                                            delete_previous_best=True,
                                                            **history["config"]
                                                            )

    # Save elapsed time
    timer_end = timer()
    elapsed_time = timer_end - timer_start
    history["elapsed_time"] = elapsed_time
    
    print(f"\nTraining complete. Elapsed time: {timedelta(seconds=elapsed_time)}")

    # Save the model
    save_loc = os.path.join(save_path, model_name+'.pth')
    torch.save(model.state_dict(), save_loc)
    print(f"Model saved to {save_loc}")

    # Save the history
    save_loc = os.path.join(save_path, model_name+'.pkl')
    with open(save_loc, 'wb') as f: 
        pickle.dump(history, f)
        print(f"History saved to {save_loc}")

    print("===========================================================")

    return history


if __name__ == "__main__":
    
    ####### TRAINING PARAMETERS #######

    EPOCHS = 100
    LEARNING_RATE = 5e-04
    BATCH_SIZE = 32
    OPTIMIZER = "adam"
    
    ############# TASK 1 #############

    DATA_DIR = "data\cartpole-samples"  
    EPISODE_LIST = list(range(7))  
    OUTPUT_DIR = r"models\task1"
    MODEL_NAME = "task1_model"
    NET_CONFIG = {
        "fc4_out_dim": 1, 
        "spatialSoftmax_dim": (193, 293)
    }
    CLASSIFICATION_TYPE = "binary"
    LOSS_FUNCTION = "bce"

    ############# TASK 2 #############

    # DATA_DIR = r"data\battlezone-samples"
    # EPISODE_LIST = [0,1]
    # OUTPUT_DIR = r"models\task2"
    # MODEL_NAME = "task2_model"
    # NET_CONFIG = {
    #     "fc4_out_dim": 9,
    #     "spatialSoftmax_dim": (98, 73)
    # }
    # CLASSIFICATION_TYPE = "multi-class"
    # LOSS_FUNCTION = "lg"
    
    ##############################################

    # Define the dataset
    transformed_dataset = CustomDataset(DATA_DIR, EPISODE_LIST)
    print("\nDataset samples   :", len(transformed_dataset))

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(transformed_dataset))
    val_size = len(transformed_dataset) - train_size
    print(f"Train, val samples : ({train_size}, {val_size})")
    train_dataset, val_dataset = random_split(transformed_dataset, [train_size, val_size])

    # Train the model and save the training history (loss info)
    model_history = train(train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        num_epochs=EPOCHS, 
                        learning_rate=LEARNING_RATE, 
                        batch_size=BATCH_SIZE,
                        classification_type=CLASSIFICATION_TYPE,
                        optimizer=OPTIMIZER,
                        loss_function=LOSS_FUNCTION,
                        save_best="val_loss",
                        save_path=OUTPUT_DIR,
                        model_name=MODEL_NAME,
                        CustomNet_kwargs=NET_CONFIG
    )
