import numpy as np
import os
import torch

from sklearn.model_selection import ParameterGrid
from torch.cuda import OutOfMemoryError
from torch.utils.data import random_split

from dataset import CustomDataset
from train import train

if __name__ == "__main__":
        
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

    PARAM_GRID = {
        "num_epochs": [100],
        "learning_rate": [1e-03, 5e-04, 1e-04, 5e-05, 1e-05],
        "batch_size": [8, 16, 32],
        "optimizer": ["adam"],
    }

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

    # PARAM_GRID = {
    #     "num_epochs": [100],
    #     "learning_rate": [1e-03, 5e-04, 1e-04, 5e-05, 1e-05],
    #     "batch_size": [16, 32, 64, 128], 
    #     "optimizer": ["adam"],
    # }
    
    ##############################################

    # Define the dataset
    transformed_dataset = CustomDataset(DATA_DIR, EPISODE_LIST)
    print("\nDataset samples    :", len(transformed_dataset))

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(transformed_dataset))
    val_size = len(transformed_dataset) - train_size
    print(f"Train, val samples : ({train_size}, {val_size})")
    train_dataset, val_dataset = random_split(transformed_dataset, [train_size, val_size])
    
    ###### GRID SEARCH ######

    # Create a ParameterGrid object
    grid = ParameterGrid(PARAM_GRID)

    best_val_loss = np.inf
    best_params = None

    oom_config = []

    print(f"Grid search on {len(grid)} hyperparameter combinations\n")
    # Iterate over all possible hyperparameter combinations
    for params in grid:
        epochs = params["num_epochs"]
        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]
        optimizer = params["optimizer"]
        
        try:
            history = train(train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            num_epochs=epochs,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            optimizer=optimizer,
                            classification_type=CLASSIFICATION_TYPE,
                            loss_function=LOSS_FUNCTION,
                            save_path=OUTPUT_DIR,
                            save_best="val_loss",
                            model_name=MODEL_NAME,
                            CustomNet_kwargs=NET_CONFIG,
                            )
        
        except OutOfMemoryError as e:  # type: ignore
            print(f"OOM error with {params}")
            oom_config.append(params)
            continue

        min_val_loss = np.min(history["val_loss"])

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_params = params
    
    # Print results
    print("Best validation loss :", best_val_loss)
    print("Best parameters      :", best_params)
    
    best_model_name = f"{MODEL_NAME}_best_model_{best_params['num_epochs']}_{best_params['learning_rate']}_{best_params['batch_size']}_{best_params['optimizer']}_{best_params['loss_function']}"
    best_model_path = os.path.join(OUTPUT_DIR, best_model_name+".pth")
    print("Best model path      :", best_model_path)

    if len(oom_config) > 0:
        print("OOM errors           :", len(oom_config))
        for config in oom_config:
            print(config)
    