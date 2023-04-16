import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tqdm import tqdm

from modeling import DataLoader, VRNet

# FOR REFERENCE - we would need to edit this further later on....


# VIDEO_RECORD_TRY = 5

# recording_env = gym.wrappers.Monitor(gym.make('CartPole-v1'), 'sample-video', video_callable=lambda episode_id: True, force=True)
# for _ in tqdm(range(VIDEO_RECORD_TRY)):
#     obs = recording_env.reset()
#     dones = False
#     while not dones:
#         action, _states = ai_expert.predict(obs)
#         obs, rewards, dones, info = recording_env.step(action)

def get_accuracy(model_path):
    data_dir = "data/simulated-samples"
    data_loader = DataLoader(data_dir=data_dir, episode_list=[5,6], samples=499)
    
    loaded_model = VRNet()
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda:0')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    loaded_model.load_state_dict(torch.load(model_path, map_location=device))

    loaded_model.to(device)

    if(loaded_model == None):
        raise("Loaded model is empty")

    num_of_predictions = 0
    correct_number_of_predictions = 0

    for i in tqdm(range(len(data_loader))):
        rgb_img, action = data_loader[i]

        rgb_img = rgb_img.to(device).float()
        target_action = action.to(device).float()
        
        output = loaded_model(rgb_img)

        # Apply sigmoid activation
        predicted_probs = torch.sigmoid(output)
        predicted_action = (predicted_probs > 0.5).float()

        if torch.equal(predicted_action, target_action):
            correct_number_of_predictions += 1
    
        num_of_predictions += 1

    accuracy = (correct_number_of_predictions/num_of_predictions) * 100 
    return accuracy

def loss_graph(history, figsize=(8,6)):
    loss = history['loss']

    plt.figure(figsize=figsize)
    plt.plot(loss)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(loss)), np.arange(1, len(loss)+1))
    plt.ylim(ymin=0, ymax=max(1, max(loss)))
    plt.show()

if __name__ == "__main__":
    
    # Load pickle file
    model_name = "model-23-04-16-v4"
    
    # Load history from file_name
    with open(model_name + ".pkl", 'rb') as f:
        history = pickle.load(f)
    
    print("Settings:", history["config"])
    accuracy = get_accuracy(model_path=(model_name + ".pth"))
    print("Accuracy:", accuracy)
    # Plot loss graph
    loss_graph(history)
    