import gym
from tqdm import tqdm

env=gym.make('CartPole-v1')
recording_env = gym.wrappers.Monitor(env, '/Users/sandrabenetho/Desktop/sample video', video_callable=lambda episode_id: True, force=True)
VIDEO_RECORD_TRY = 100
counter = 1
for _ in tqdm(range(VIDEO_RECORD_TRY)):
    observation = recording_env.reset()
    done = False
    episode_observation_action_pairs = []
    while not (done):
        action = env.action_space.sample()
        observation, reward, done, info = recording_env.step(action)
        episode_observation_action_pairs.append([observation, action])
    
    with open('/Users/sandrabenetho/Desktop/observation_action_pairs/episode_' + str(counter) + '.txt', 'w') as f:
        for observation_action_pair in episode_observation_action_pairs:
            f.write("Observation: " + str(observation_action_pair[0]) + " Action: " + str(action))
            f.write("\n")
    counter = counter + 1
