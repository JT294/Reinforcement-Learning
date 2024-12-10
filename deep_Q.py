import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from generals.agents.agent import Agent
from generals.core.game import Action, Game
from generals.core.observation import Observation
from generals.agents.expander_agent import ExpanderAgent
from generals.core.grid import GridFactory
import gymnasium as gym
class Qfunction:
    def __init__(self, grid_size, lr=1e-3, gamma=0.99):
        self.grid_size = grid_size
        self.lr = lr
        self.gamma = gamma
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def encode_observation(self, observation):
        obs = observation['observation']
        obs_values = []
        for key, value in obs.items():
            if isinstance(value, (list, np.ndarray)):
                obs_values.extend(np.array(value).flatten())
            elif isinstance(value, (int, float)):
                obs_values.append(value)
        obs_flat = np.array(obs_values)
        return obs_flat  

    def initialize_model(self, input_size, action_space_size):
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, action_space_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def restore_statistic(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def compute_argmaxQ(self, encoded_obs):
        encoded_obs = encoded_obs.to(self.device)
        with torch.no_grad():
            q_values = self.model(encoded_obs)
            action_index = q_values.argmax().item()
        return action_index

    def decode_action(self, action_index, observation):
        grid_size = self.grid_size
        num_cells = grid_size[0] * grid_size[1]
        num_directions = 4
        num_splits = 2
        num_pass = 2

        pass_index = action_index // (num_cells * num_directions * num_splits)
        remainder = action_index % (num_cells * num_directions * num_splits)

        cell_index = remainder // (num_directions * num_splits)
        remainder = remainder % (num_directions * num_splits)

        direction_index = remainder // num_splits
        split_index = remainder % num_splits

        x = cell_index // grid_size[1]
        y = cell_index % grid_size[1]

        action = {
            "pass": pass_index,
            "cell": np.array([x, y]),
            "direction": direction_index,
            "split": split_index
        }
        return action

    def encode_action(self, action):
        pass_index = action["pass"]
        x, y = action["cell"]
        direction_index = action["direction"]
        split_index = action["split"]

        grid_size = self.grid_size
        num_cells = grid_size[0] * grid_size[1]
        num_directions = 4
        num_splits = 2
        num_pass = 2

        cell_index = x * grid_size[1] + y

        action_index = (pass_index * num_cells * num_directions * num_splits +
                        cell_index * num_directions * num_splits +
                        direction_index * num_splits +
                        split_index)
        return action_index

    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        q_values = self.model(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values, _ = next_q_values.max(1)
            expected_state_action_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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


class DQNAgent(Agent):
    def __init__(self, id: str = 'DQN', color: tuple[int, int, int] = (0, 245, 0), grid_size=(5, 5), lr=1e-3):
        super().__init__(id, color)
        self.grid_size = grid_size
        self.q_function = Qfunction(grid_size, lr=lr)
        self.num_cells = grid_size[0] * grid_size[1]


        self.replay_buffer = ExperienceReplayBuffer()
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.gamma = 0.99
        self.batch_size = 64
    
    def act(self, observation: Observation, game: Game, done: bool, prev_act: Action):
        encoded_obs = self.q_function.encode_observation(observation)
        input_size=len(encoded_obs)
    
        action_space_size = 2 * self.num_cells * 4 * 2
        self.q_function.initialize_model(input_size, action_space_size)

        encoded_obs_tensor = torch.FloatTensor(encoded_obs).unsqueeze(0)
        action_index = self.q_function.compute_argmaxQ(encoded_obs_tensor)
        action = self.q_function.decode_action(action_index, observation)
        mask = observation["action_mask"]
        valid_actions = np.argwhere(mask == 1)
        penalty=0
        for action_index in range(len(valid_actions)):
            if (action["cell"]==valid_actions[action_index][:2]).all() and action["direction"]==valid_actions[action_index][2]:
                return action,penalty
            else:
                action={
                    "pass": 1,
                    "cell": np.array([0, 0]),
                    "direction": 0,
                    "split": 0,
                }
                penalty=100
        return action,penalty
    def reset(self):
        pass

    
    def remember(self, state, action_index, reward, next_state, done):
        self.replay_buffer.push((state, action_index, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        self.q_function.train(batch)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def unwrap_game(env):
    while hasattr(env, "env"):  
        env = env.env
    return getattr(env, "game", None) 
 
def reward_function(observation,action,done,info):
    if done:
        if info["is_winner"]:
            return 1
        else:
            return -1
    else:
        return 0
def train_dqn(num_episodes, grid_size=(5, 5)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent =  DQNAgent(id='DQN',grid_size=grid_size)
    npc_agent = ExpanderAgent(id='Expander')
    grid_factory = GridFactory(    
            grid_dims=(5, 5), 
            mountain_density=0.2, 
            city_density=0.05,  
            general_positions=[(1, 2), (4, 4)],  
            seed=38,  
            )
    env = gym.make("gym-generals-v0", grid_factory=grid_factory, agent=dqn_agent, npc=npc_agent, render_mode="human", reward_fn=reward_function)
    for episode in range(num_episodes):
        observation, info = env.reset()
        terminated = truncated = False
        prev_action = None
        total_reward = 0
        count=0
        while not (terminated or truncated):
            game = unwrap_game(env)
            obs_dqn = observation
            action_dqn,penalty = dqn_agent.act(obs_dqn, game, terminated, prev_action)

            next_observation, reward, terminated, truncated, info = env.step(action_dqn)
            status = game.get_infos()
            mct_army, mct_land = status["DQN"]["army"], status["DQN"]["land"]
            npc_army, npc_land = status["Expander"]["army"], status["Expander"]["land"]
            army_score = mct_army/(mct_army+npc_army)
            land_score = mct_land/(mct_land+npc_land)
            reward+= 0.007 * land_score + 0.003 * army_score - penalty
            dqn_agent.remember(
                dqn_agent.q_function.encode_observation(obs_dqn),
                dqn_agent.q_function.encode_action(action_dqn),
                reward,
                dqn_agent.q_function.encode_observation(next_observation),
                terminated
            )
            dqn_agent.replay()
            observation = next_observation
            total_reward += reward
            count+=1
            if episode%500==499:
                env.render()

        print(f"Episode {episode + 1}/{num_episodes}: Average Reward = {total_reward/count}")

    # Save the trained model
    torch.save(dqn_agent.q_function.model.state_dict(), "q_function_model_1.pth")
    print("Model saved to 'q_function_model_1.pth'")

def evaluate(Q, env, episodes):
    score = 0
    for e in range(episodes):
        obs, info = env.reset()
        done = False
        rsum = 0
        while not done:
            encode_obs = Q.encode_observation(obs, with_mask=True)
            encode_obs = torch.FloatTensor(encode_obs).unsqueeze(0)
            with torch.no_grad():
                action = Q.compute_argmaxQ(encode_obs)
            newobs, r, terminated, truncated, info  = env.step(action)
            rsum += r 
            obs = newobs 
        score += rsum 
    score = score/episodes 
    return score 

if __name__ == "__main__":
    
    num_episodes = 200  # Set the number of episodes you want to train for
    train_dqn(num_episodes, grid_size=(5, 5))

    # dqn_agent =  DQNAgent(id='DQN',grid_size=(5, 5))
    # dqn_agent.q_function.restore_statistic("q_function_model_1.pth")
    # npc_agent = ExpanderAgent(id='Expander')
    # grid_factory = GridFactory(    
    #         grid_dims=(5, 5), 
    #         mountain_density=0.2, 
    #         city_density=0.05,  
    #         general_positions=[(1, 2), (4, 4)],  
    #         seed=38,  
    #         )
    # env = gym.make("gym-generals-v0", grid_factory=grid_factory, agent=dqn_agent, npc=npc_agent, render_mode="human", reward_fn=reward_function)
    # obs, info = env.reset()
    # encoded_obs = dqn_agent.q_function.encode_observation(obs)
    # input_size=len(encoded_obs)
    # action_space_size = 2 * 5 * 5 * 4 * 2
    # dqn_agent.q_function.initialize_model(input_size, action_space_size)
    # score = evaluate(dqn_agent.q_function, env, 100)
    # print("eval performance of DQN agent: {}".format(score))


