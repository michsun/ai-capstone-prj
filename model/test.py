import gymnasium as gym

from tqdm import tqdm

# FOR REFERENCE - we would need to edit this further later on....


VIDEO_RECORD_TRY = 5

recording_env = gym.wrappers.Monitor(gym.make('CartPole-v1'), 'sample-video', video_callable=lambda episode_id: True, force=True)
for _ in tqdm(range(VIDEO_RECORD_TRY)):
    obs = recording_env.reset()
    dones = False
    while not dones:
        action, _states = ai_expert.predict(obs)
        obs, rewards, dones, info = recording_env.step(action)