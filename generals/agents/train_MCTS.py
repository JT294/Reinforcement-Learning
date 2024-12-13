import gymnasium as gym
from generals.core.grid import GridFactory
from generals.agents import ExpanderAgent  
from generals.agents.MCTS_agent import MCTSAgent
from generals.envs.gymnasium_generals import GymnasiumGenerals
from generals.agents.deepQ import Qfunction

from collections import deque
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

model_name = "q_function_model_7"
loud = False
lr = 1e-3
tau = 5
episodes = 50
initialsize = 500 
epsilon = .2
epsilon_decay = .999 
gamma = .99
total_reward = 0
steps = 0
grid_width = 5
grid_height = 5
agent = MCTSAgent()
npc = ExpanderAgent()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grid_factory = GridFactory(    
    grid_dims=(grid_width, grid_height),  # Grid height and width
    mountain_density=0.2,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    general_positions=[(1, 2), (4, 4)],  # Positions of the generals
    seed=38,  # Seed to generate the same map every time
    )

def unwrap_game(env):
    while hasattr(env, "env"):  # 递归解包包装器
        env = env.env
    return getattr(env, "game", None)  # 获取底层环境的 game 属性

def reward_function(observation,action,done,info):
    if done:
        if info["is_winner"]:
            return 1
        else:
            return -1
    else:
        return 0
class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

env = gym.make("gym-generals-v0", grid_factory=grid_factory, agent=agent, npc=npc, render_mode="human", reward_fn=reward_function)
observation, info = env.reset()
terminated = truncated = False
replay_buffer = ExperienceReplayBuffer()
Qprincipal = Qfunction((grid_width, grid_height), lr=lr)
Qtarget =  Qfunction((grid_width, grid_height), lr=lr)
encoded_state = Qprincipal.encode_observation(observation)
action_space = 2 + grid_width * grid_height + 4 + 2
Qprincipal.initialize_model(len(encoded_state), action_space)
Qtarget.initialize_model(len(encoded_state), action_space)

ave_reward = []
for e in range(episodes):
    agent = MCTSAgent()
    observation, info = env.reset()
    terminated, truncated = False, False 
    prev_act = 0
    while not (terminated or truncated):
        game = unwrap_game(env)
        if game is None:
            raise RuntimeError("The environment does not contain a 'game' attribute.")
        action = agent.act(observation, game, terminated, prev_act) # agent.act(observation)
        
        next_obs, reward, terminated, truncated, info = env.step(action)   
        prev_act = action      
        encoded_action = Qprincipal.encode_action(action, grid_width=grid_width, grid_height=grid_height)
        next_state = Qprincipal.encode_observation(next_obs)
        state = Qprincipal.encode_observation(observation)
        replay_buffer.push((state, encoded_action, reward, next_state, terminated))
        # while replay_buffer
        if len(replay_buffer)>32:
            if loud:
                with open("replay_buffer.txt", "w") as file:
                    for idx, entry in enumerate(replay_buffer):
                        file.write(f"Entry {idx}:\n{entry}\n\n")
                
            batch = replay_buffer.sample(32)
            states, actions, rewards, next_state, done = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = [torch.tensor(action) if not isinstance(action, torch.Tensor) else action for action in actions] ### need to check problem
            actions = torch.stack(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_state = torch.FloatTensor(np.array(next_state)).to(device)
            done = torch.FloatTensor(done).to(device)
            max_next_Q = Qtarget.compute_maxQvalues(next_state)
            targets = rewards + gamma * max_next_Q * (1 - done)
            if loud: print("states shape ", states.shape, "actions shape ", actions.shape)
            loss = Qprincipal.train(states, actions, targets)
        if steps % tau == 0:
            for v,v_ in zip(Qprincipal.model.parameters(), Qtarget.model.parameters()):
                v_.data.copy_(v.data)    
        if epsilon > 0.01:
            epsilon *= epsilon_decay        
        observation = next_obs
        # next_state = next_state.numpy()[0] if isinstance(next_state, torch.Tensor) else next_state
        # state = Qprincipal.encode_observation(observation)#next_state
        total_reward += reward
        steps += 1
        #env.render()
    ave_reward.append(total_reward/steps)
    print(f"Episode {e + 1}, Total Reward: {total_reward}, Steps: {steps}")
torch.save(Qprincipal.model.state_dict(), f"{model_name}.pth")
print("Model saved.")

x = np.array(list(range(episodes)))
plt.plot(x,ave_reward)
plt.title("average reward")
plt.xlabel("episodes")
plt.ylabel("Average Reward")
plt.grid()
plt.savefig(f"{model_name}.png", dpi=300)  # 指定文件名和分辨率
plt.show()
