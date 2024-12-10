import numpy as np
import random
import itertools
from math import sqrt, log
from copy import deepcopy
import weakref
import gymnasium as gym
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Generals imports (adjust if needed)
from generals.core.config import Direction
from generals.core.game import Action, Game
from generals.core.observation import Observation
from generals.agents.agent import Agent
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.grid import GridFactory

###################################################
# Qfunction Class (Unified for both MCTS and DQN)  #
###################################################
class Qfunction:
    def __init__(self, grid_size, lr=1e-3, gamma=0.99):
        self.grid_size = grid_size
        self.lr = lr
        self.gamma = gamma
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

    def encode_observation(self, observation):
        obs = observation['observation'] if 'observation' in observation else observation
        obs_values = []
        for key, value in obs.items():
            if isinstance(value, (list, np.ndarray)):
                obs_values.extend(np.array(value).flatten())
            elif isinstance(value, (int, float)):
                obs_values.append(value)
        obs_flat = np.array(obs_values)
        return obs_flat

    def initialize_model(self, input_size, action_space_size):
        if self.model is None:
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
            self.criterion = nn.MSELoss()

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
        # pass can be 0 or 1
        # action_space = 2 * num_cells * num_directions * num_splits

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

#####################################################
# ExperienceReplayBuffer for DQN Training            #
#####################################################
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

#########################################
# Node and MCTS-related classes/methods #
#########################################
loud = False
debuging = False

class Node:
    shared_model = None
    def __init__(self, game, done, parent, observation, action_index):
        self.child = None
        self.T = 0  # total rewards from MCTS exploration
        self.N = 0  # visit count
        self.game = game # environment
        self.observation = observation  # observation of environment
        self.done = done # True/False
        self.parent = weakref.ref(parent) if parent is not None else None
        self.action_index = action_index
        self.model = Node.shared_model

    @classmethod
    def set_shared_model(cls, model_path, obs, grid_size, lr=1e-3):
        if os.path.exists(model_path):
            q_function = Qfunction(grid_size, lr=lr)
            en_obs = q_function.encode_observation(obs)
            action_space = 2 * grid_size[0] * grid_size[1] * 4 * 2
            q_function.initialize_model(len(en_obs), action_space)
            q_function.model.load_state_dict(torch.load(model_path))
            q_function.model.eval()
            cls.shared_model = q_function
        else:
            print(f"Model file '{model_path}' does not exist. Skipping load.")
            cls.shared_model = None

    def copynode(self):
        new_node = Node(
            game=deepcopy(self.game),
            done=self.done,
            parent=None,
            observation=self.observation,
            action_index=self.action_index,
        )
        # copy other properties
        new_node.child = deepcopy(self.child)
        new_node.T = self.T
        new_node.N = self.N
        return new_node

    def getUCBscore(self):
        if self.N == 0:
            return float('inf')
        top_node = self
        while top_node.parent:
            top_node = top_node.parent
        return (self.T/self.N) + sqrt(log(top_node.N)/self.N)

    def detach_parent(self):
        del self.parent
        self.parent = None

    def process_task(self, index, action, game):
        observation, info = game.step(action)
        return index, Node(game, game.is_done(), None, observation["MCTS"].as_dict(), action["MCTS"])

    def create_child(self):
        if self.done: return
        actions = []
        games = []
        mask = self.observation["action_mask"]
        valid_actions = np.argwhere(mask == 1)

        npc = ExpanderAgent()
        npc_observation = self.game.agent_observation("Expander").as_dict()

        for i in range(len(valid_actions)):
            cell = valid_actions[i][:2]
            direction = valid_actions[i][2]
            for j in range(2):
                action = {
                    "pass": 0,
                    "cell": cell,
                    "direction": direction,
                    "split": j,
                }
                actions.append({"MCTS": action, "Expander": npc.act(npc_observation)})
                new_game = deepcopy(self.game)
                games.append(new_game)

        child = {}
        index = 0
        with ThreadPoolExecutor() as executor:
            tasks = [
                executor.submit(self.process_task, idx, action, game)
                for idx, (action, game) in enumerate(zip(actions, games))
            ]
            for t in tasks:
                idx, node = t.result()
                node.parent = self
                child[idx] = node

        self.child = child

    def explore(self):
        current = self
        while current.child and self.done == False:
            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [a for a,c in child.items() if c.getUCBscore() == max_U]
            if len(actions) == 0:
                print("error zero length", max_U)
            action = random.choice(actions)
            current = child[action]

        if current.N < 1:
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(list(current.child.values()))
            current.T = current.T + current.rollout()
        current.N += 1

        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def rollout(self):
        if self.done:
            return 0
        step = 0
        max_step = 100
        new_game = deepcopy(self.game)
        npc = ExpanderAgent()
        mcts_observation = self.observation
        npc_observation = new_game.agent_observation("Expander").as_dict()

        while not new_game.is_done():
            step += 1
            mask = mcts_observation["action_mask"]
            valid_actions = np.argwhere(mask == 1)
            if len(valid_actions) == 0:
                action = {
                    "pass": 1,
                    "cell": np.array([0, 0]),
                    "direction": 0,
                    "split": 0,
                }
            else:
                if self.model is None:
                    pass_turn = 0 if np.random.rand() > 0.05 else 1
                    split_army = 0 if np.random.rand() > 0.25 else 1
                    action_index = np.random.choice(len(valid_actions))
                    cell = valid_actions[action_index][:2]
                    direction = valid_actions[action_index][2]
                    action = {
                        "pass": pass_turn,
                        "cell": cell,
                        "direction": direction,
                        "split": split_army,
                    }
                else:
                    encode_obs = self.model.encode_observation(mcts_observation)
                    encode_obs = torch.FloatTensor(encode_obs).unsqueeze(0)
                    with torch.no_grad():
                        action_index = self.model.compute_argmaxQ(encode_obs)
                    action = self.model.decode_action(action_index, mcts_observation)

            observation, info = new_game.step({"MCTS": action, "Expander": npc.act(npc_observation)})
            mcts_observation = observation["MCTS"].as_dict()
            npc_observation = observation["Expander"].as_dict()

            if new_game.is_done():
                if new_game.agent_won('MCTS'):
                    if loud: print("MCTS rollout win")
                    return 1
                else:
                    if loud: print("Expander rollout loss")
                    return -1

            if step >= max_step:
                status = new_game.get_infos()
                mct_army, mct_land = status["MCTS"]["army"], status["MCTS"]["land"]
                npc_army, npc_land = status["Expander"]["army"], status["Expander"]["land"]
                army_score = mct_army/(mct_army+npc_army)
                land_score = mct_land/(mct_land+npc_land)
                return 0.7 * land_score + 0.3 * army_score

    def next(self):
        if not self.child:
            return self, self.action_index
        child = self.child
        max_N = max(node.N for node in child.values())
        max_children = [c for a,c in child.items() if c.N == max_N]
        if len(max_children) == 0:
            print("error zero length", max_N)
        max_child = random.choice(max_children)
        return max_child, max_child.action_index

MCTS_POLICY_EXPLORE = 20

def Policy_MCTS(mytree):
    for i in range(MCTS_POLICY_EXPLORE):
        if loud: print("exploring", i)
        mytree.explore()
    next_tree, next_action = mytree.next()
    next_tree.detach_parent()
    return next_tree, next_action

class MCTSAgent(Agent):
    def __init__(self, id: str = 'MCTS', color: tuple[int, int, int] = (0, 245, 0)):
        super().__init__(id, color)
        self.tree = None
        self.current = None

    def act(self, observation: Observation, game: Game, done: bool, prev_act: Action) -> Action:
        if self.tree is None:
            Node.set_shared_model("123q_function_model.pth", observation, (5, 5))
            self.tree = Node(game, done, None, observation, None)
            self.current = self.tree

        self.current.observation = observation
        self.current.game = game

        mask = observation["action_mask"]
        valid_actions = np.argwhere(mask == 1)
        if len(valid_actions) == 0 or done or self.current.done:
            if loud: print("DO nothing in MCTSAgent.act")
            return {
                "pass": 1,
                "cell": np.array([0, 0]),
                "direction": 0,
                "split": 0,
            }

        if debuging:
            pass_turn = 0 if np.random.rand()>0.05 else 1
            split_army = 0 if np.random.rand()>0.25 else 1
            action_index = np.random.choice(len(valid_actions))
            cell = valid_actions[action_index][:2]
            direction = valid_actions[action_index][2]
            return {
                "pass": pass_turn,
                "cell": cell,
                "direction": direction,
                "split": split_army,
            }
        else:
            child_node, action = Policy_MCTS(self.current.copynode())
            self.current.child = child_node
            child_node.parent = self.current
            self.current = self.current.child
            return action

    def reset(self):
        pass

################################
# DQN Agent and Training Logic #
################################
def reward_function(game, observation, action, done, info):
    if done:
        if info["is_winner"]:
            return 1
        else:
            return -1
    else:
        return 0

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
        self.initialized = False

    def act(self, observation: Observation, game: Game, done: bool, prev_act: Action):
        encoded_obs = self.q_function.encode_observation(observation)
        input_size = len(encoded_obs)
        action_space_size = 2 * self.num_cells * 4 * 2

        if not self.initialized:
            self.q_function.initialize_model(input_size, action_space_size)
            self.initialized = True

        encoded_obs_tensor = torch.FloatTensor(encoded_obs).unsqueeze(0)
        if random.random() < self.epsilon:
            # random action
            action_index = random.randint(0, action_space_size-1)
        else:
            action_index = self.q_function.compute_argmaxQ(encoded_obs_tensor)
        action = self.q_function.decode_action(action_index, observation)

        # Verify action is valid; if not, choose a valid fallback
        mask = observation["action_mask"]
        valid_actions = np.argwhere(mask == 1)
        valid = False
        for idx in range(len(valid_actions)):
            if (action["cell"] == valid_actions[idx][:2]).all() and action["direction"] == valid_actions[idx][2]:
                valid = True
                break

        if not valid:
            if len(valid_actions) == 0:
                # No valid actions means must pass
                action = {
                    "pass": 1,
                    "cell": np.array([0, 0]),
                    "direction": 0,
                    "split": 0,
                }
            else:
                # Just pick a random valid action
                chosen_idx = random.randint(0, len(valid_actions)-1)
                cell = valid_actions[chosen_idx][:2]
                direction = valid_actions[chosen_idx][2]
                action = {
                    "pass": 0,
                    "cell": cell,
                    "direction": direction,
                    "split": 0,
                }
        return action

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

def train_dqn(num_episodes, grid_size=(5, 5)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQNAgent(id='DQN', grid_size=grid_size)
    npc_agent = RandomAgent(id='Expander')
    grid_factory = GridFactory(
        grid_dims=(5, 5),
        mountain_density=0.2,
        city_density=0.05,
        general_positions=[(1, 2), (4, 4)],
        seed=38,
    )
    env = gym.make("gym-generals-v0", grid_factory=grid_factory, agent=dqn_agent, npc=npc_agent, render_mode="human", reward_fn=reward_function)
    total_reward = 0
    count = 0
    for episode in range(num_episodes):
        observation, info = env.reset()
        terminated = truncated = False
        prev_action = None

        while not (terminated or truncated):
            game = unwrap_game(env)
            obs_dqn = observation
            action_dqn = dqn_agent.act(obs_dqn, game, terminated, prev_action)

            next_observation, reward, terminated, truncated, info = env.step(action_dqn)

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
            count += 1
            # Optional rendering for debug
            if episode % 100 == 99:
                total_reward=0
            #env.render()

        print(f"Episode {episode + 1}/{num_episodes}: Reward = {total_reward}")

    # Save the trained model
    torch.save(dqn_agent.q_function.model.state_dict(), "q_function_model.pth")
    print("Model saved to 'q_function_model.pth'")

if __name__ == "__main__":
    num_episodes = 20000
    train_dqn(num_episodes, grid_size=(5, 5))
