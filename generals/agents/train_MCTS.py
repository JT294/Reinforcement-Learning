import gymnasium as gym
from generals.core.grid import GridFactory
from generals.agents import ExpanderAgent  
from generals.agents.MCTS_agent import MCTSAgent
from generals.envs.gymnasium_generals import GymnasiumGenerals
from .deepQ import Qfunction

from collections import deque
import random
import torch
import numpy as np

loud = True
lr = 1e-3
tau = 100 
episodes = 300
initialsize = 500 
epsilon = .2
epsilon_decay = .999 
tau = 100 
gamma = .99
total_reward = 0
steps = 0
grid_width = 5
grid_height = 5
agent = MCTSAgent()
npc = ExpanderAgent()
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

env = gym.make("gym-generals-v0", grid_factory=grid_factory, agent=agent, npc=npc, render_mode="human")
observation, info = env.reset()
terminated = truncated = False
replay_buffer = deque(maxlen=10000)
Qprincipal = Qfunction((5, 5), lr=lr)
Qtarget =  Qfunction((5, 5), lr=lr)
encoded_state = Qprincipal.encode_observation(observation)
# sample_action = {"pass": 0, "cell": [0, 0], "direction": 0, "split": 0}
# encoded_action = Qprincipal.encode_action(sample_action, grid_width=grid_width, grid_height=grid_height)
action_space = 2 * grid_width * grid_height * 4 * 2
Qprincipal.initialize_model(len(encoded_state), action_space)
Qtarget.initialize_model(len(encoded_state), action_space)

for e in range(episodes):
    agent = MCTSAgent()
    observation, info = env.reset()
    state = Qprincipal.encode_observation(observation)
    terminated, truncated = False, False 
    prev_act = 0
    while not (terminated or truncated):
        game = unwrap_game(env)
        if game is None:
            raise RuntimeError("The environment does not contain a 'game' attribute.")
        action = agent.act(observation, game, terminated, prev_act) # agent.act(observation)
        # if np.random.rand() < epsilon:
        #     action = env.action_space.sample()
        # else:
        # action = Qprincipal.compute_argmaxQ(state) ### which one, need see
        
        next_obs, reward, terminated, truncated, info = env.step(action)   
        prev_act = action      
        encoded_action = Qprincipal.encode_action(action, grid_width=grid_width, grid_height=grid_height)
        next_state = Qprincipal.encode_observation(next_obs)
        replay_buffer.append((state, encoded_action, reward, next_state, terminated))
        # while replay_buffer
        if len(replay_buffer)>32:
            if loud:
                with open("replay_buffer.txt", "w") as file:
                    for idx, entry in enumerate(replay_buffer):
                        file.write(f"Entry {idx}:\n{entry}\n\n")
                
            batch = random.sample(replay_buffer, 32)
            states, actions, rewards, next_state, done = zip(*batch)
            states = torch.FloatTensor(states)
            # actions = [Qtarget.decode_action(ac, grid_width=grid_width, grid_height=grid_height) for ac in actions]
            actions = [torch.tensor(action) if not isinstance(action, torch.Tensor) else action for action in actions] ### need to check problem
            actions = torch.stack(actions)
            rewards = torch.FloatTensor(rewards)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)
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
        next_state = next_state.numpy()[0] if isinstance(next_state, torch.Tensor) else next_state
        state = next_state
        total_reward += reward
        steps += 1
        env.render()
    print(f"Episode {e + 1}, Total Reward: {total_reward}, Steps: {steps}")

torch.save(Qprincipal.model.state_dict(), "q_function_model.pth")
print("Model saved.")