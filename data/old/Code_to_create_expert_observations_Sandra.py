import pandas as pd
import gym
from gym.envs.classic_control.cartpole import *
from pyglet.window import key
import matplotlib.pyplot as plt
import time

bool_do_not_quit = True  # Boolean to quit pyglet
scores = []  # Your gaming score
a = 0  # Action

def key_press(k, mod):
    global bool_do_not_quit, a, restart
    if k==0xff0d: restart = True
    if k==key.ESCAPE: bool_do_not_quit=False  # Added to Quit
    if k==key.Q: bool_do_not_quit=False  # Added to Quit
    if k==key.LEFT:  a = 0  # 0	Push cart to the left
    if k==key.RIGHT: a = 1  # 1	Push cart to the right

def run_cartPole_asHuman(policy=None, record_video=True, counter = 1):
    env = CartPoleEnv()

    env.reset()
    env.render()
    env = gym.wrappers.Monitor(env, '/Users/sandrabenetho/Desktop/videos/video-'+str(counter))
    # env.monitor.start('/Users/sandrabenetho/Desktop/videos/video-'+str(counter), force=True)
    env.viewer.window.on_key_press = key_press

    while bool_do_not_quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        t1 = time.time()  # Trial timer
        frame_numbers = []
        observations = []
        actions = []
        while bool_do_not_quit:
            s, r, done, info = env.step(a)
            frame_numbers.append(steps)
            observations.append(s)
            actions.append(a)
            time.sleep(1/10)  # 10fps: Super slow for us poor little human!
            total_reward += r
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart:
                t1 = time.time()-t1
                scores.append(total_reward)
                print("Trial", len(scores), "| Score:", total_reward, '|', steps, "steps | %0.2fs."% t1)
                break
        data = {'Frame No.': frame_numbers,
                'Observation': observations,
                'Action': actions}
        data_df = pd.DataFrame(data)
        data_df.to_excel('/Users/sandrabenetho/Desktop/observation_action_pairs/episode_' + str(counter) + '.xlsx')
            
    env.close()

def main():
    for i in range(4):
        run_cartPole_asHuman(counter = i + 1)  

if __name__ == "__main__":
    main()