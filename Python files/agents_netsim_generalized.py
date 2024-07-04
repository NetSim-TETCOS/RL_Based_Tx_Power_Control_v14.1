import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN,A2C,PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch
import matplotlib.pyplot as plt
import random
# Assuming your custom environment class is defined as `g3_ue6` in a file named `env_21Jun2024.py`
from env_general import generalized
import tempfile
import csv
import os

# in Case your pc lags
# import psutil
# import os
# p = psutil.Process(os.getpid())
# p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

# Create a function to make the environment
def make_env():
    return generalized()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Get the path to the temporary directory
temp_dir = tempfile.gettempdir()
print("Temporary directory:", temp_dir)
file_name = "DeviceCount.csv"
file_path = os.path.join(temp_dir, file_name)
# Read the CSV file
if os.path.exists(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        called_once = False
        for row in reader:
            if(not called_once):
               called_once = True
            else:
                # Assuming the file has one row and two columns
                value1, value2 = row
                print("Value 1:", value1)
                print("Value 2:", value2)
else:
    print(f"The file {file_name} does not exist in the temporary directory.")


# num_gNBs = (int)(input("Enter number of gNBS"))
# num_UEs = (int)(input("Enter number of UES"))
num_gNBs = (int)(value2)
print("number of gNBs are", num_gNBs)
num_UEs = (int)(value1)
print("number of UEs are", num_UEs)
# Wrap the environment
# way 1
# env = DummyVecEnv([make_env])  
# way 2
env = gym.make('general-v1',num_gNBs=num_gNBs,num_UEs=num_UEs)
total_rewards_per_episode=[]

# Specify number of episodes we need to run for
# episodes=(int)(input("Enter Number of episodes:"))
episodes=1000
steps=1000

# Custom linear/noop activation function
class Identity(torch.nn.Module):
    def forward(self, x):
        return x
policy_kwargs = dict(
    net_arch=[64, 64],  # Two hidden layers with 64 neurons each
    activation_fn=Identity  # Use preset/custom function
)

# Create the agent DQN / A2C / PPO
model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs,n_steps=500, verbose=0) #to avoid detailed output set verbose=0 else verbose=1

def custom_learn(model, total_timesteps):
    # obs = env.reset()
    timesteps = 0

    while timesteps < total_timesteps:
        # Collect rollouts
        for _ in range(model.n_steps):
            action, _states = model.predict(obs, deterministic=False)
            new_obs, reward, done, info = env.step(action)
            model.rollout_buffer.add(obs, action, reward, new_obs, done)
            obs = new_obs
            timesteps += 1
            if done:
                obs = env.reset()
        
        # Compute returns and advantages
        model.rollout_buffer.compute_returns_and_advantage()

        # Policy and value function update
        # for epoch in range(model.n_epochs):
        #     for batch in model.rollout_buffer.get(batch_size=model.batch_size):
        #         # Compute loss and optimize
        #         loss = model.compute_loss(batch)
        #         model.optimizer.zero_grad()
        #         loss.backward()
        #         model.optimizer.step()
        model.train()
        
        # Logging (if any)
        model.logger.record("timesteps", timesteps)
# Training the model
print("model learning")
try:
    for i in range(episodes):
        # if(i==0):
        #     env.envs[0].reset(options=[1])
        # else:
        #     env.reset()
        print(f"Episode: {i}")
        model.learn(total_timesteps=steps)
        env.resetcon()
        print("episode end")
        total_rewards_per_episode.append(sum(env.total_rewards)/steps)
        print(total_rewards_per_episode[-1])
except KeyboardInterrupt:
        print("Loop terminated by user.") #If you want to stop training in between 
# for i in range(episodes):
#     for j in range(steps):
#         action, _ = model.predict(obs, deterministic=False)
#         new_obs, reward, done,_, info = env.step(action)

#         # Manually store the experience
#         model.replay_buffer.add(obs, action, reward, new_obs, done)
#         obs = new_obs

#         # if done:
#         #     obs = env.reset()

#         # Periodically update the model
#         if j % model.n_steps == 0:
#             model.train() 

    # timesteps += 1
print("learning ended")

# Saving the model
model_path = "ppo_g3_ue6_model_general_1x1.zip" # will be saved in the directory selected at runtime
# To give custom path
# model_path = "C:/Users/rpamn/Desktop/Ronak/ql new/openai gym/21Jun2024/ppo_g3_ue6_model_2kx1.5k_env3.zip"
model.save(model_path)
print(f"Model saved to {model_path}.")


# Plotting the graph for total rewards
plt.plot(total_rewards_per_episode)  # EMA
plt.title("Average rewards per episode for PPO general")
plt.xlabel("Episodes")
plt.ylabel("Average Sum Throughput(Mbps)")
plt.savefig("Average_rewards_PPO_general_1x1.png")
plt.show()

# Evaluate the trained agent
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    obs_tensor = torch.tensor(obs, dtype=torch.float64)
    # In case you use DQN
    # q_values = model.policy.q_net(obs_tensor.reshape(1, -1)).detach().numpy()
    # print("Q-values:", q_values)
    print("Chosen action:", action)
    print(rewards)
    print(obs)



