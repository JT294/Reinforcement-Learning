import numpy as np
import random
import itertools
from math import *
from copy import deepcopy 
import weakref
import gymnasium as gym
from concurrent.futures import ThreadPoolExecutor
import torch
import os

from generals.core.config import Direction
from generals.core.game import Action, Game 
from generals.core.observation import Observation 

from .agent import Agent
from .expander_agent import ExpanderAgent 
from .deepQ import Qfunction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loud = False
debuging = False
class Node:
    shared_model = None
    def __init__(self, game, done, parent, observation, action_index,lr=1e-3):
        self.child = None 
        self.T = 0  # total rewards from MCTS exploration
        self.N = 0  # visit count
        self.game = game # environment
        self.observation = observation  # observation of environment
        self.done = done # win/loss/draw ## True/False now
        self.parent = weakref.ref(parent) if parent is not None else None # link to parent
        self.action_index = action_index # action leads to current node
        self.model = Node.shared_model  ### TO DO: want to train a deepQ model


    @classmethod
    def set_shared_model(cls, model_path, obs, grid_size, lr=1e-3):
        if os.path.exists(model_path):
            q_function = Qfunction(grid_size, lr=lr)
            q_function.initialize_model(331,33)
            q_function.model.load_state_dict(torch.load(model_path,weights_only=True))
            q_function.model.eval()
            cls.shared_model = q_function
        else:
            print(f"Model file '{model_path}' does not exist. Skipping load.")
    def copynode(self):
        new_node = Node(
            game=deepcopy(self.game),
            done=self.done,
            parent=None,  # parent 使用 weakref，不需要深复制
            observation=self.observation,
            action_index=self.action_index,
        )
        # 深复制其他属性
        new_node.child = deepcopy(self.child)
        new_node.T = self.T
        new_node.N = self.N
        return new_node

    def encode_valid_action(self, valid_actions, split = 0, grid_width = 5, grid_height = 5):
        total_cells = grid_width * grid_height
        total_directions = 4
        total_splits = 2
        total_passes = 2  # Assuming 'pass' can be 0 or 1

        num_actions = valid_actions.shape[0]
        size = total_cells + total_directions + total_splits + total_passes

        # Indices for each component
        pass_indices = torch.zeros(num_actions, dtype=torch.long)
        cell_indices = valid_actions[:, 0] * grid_width + valid_actions[:, 1] + total_passes
        direction_indices = valid_actions[:, 2] + total_passes + total_cells
        split_indices = valid_actions[:, 2] + total_passes + total_cells + split  # Assuming split relates to direction

        # Creating one-hot encoded tensor
        one_hot_encoded = torch.zeros(num_actions, size, dtype=torch.int)
        
        # Using index_fill_ to set values at specific indices
        one_hot_encoded.scatter_(1, pass_indices.unsqueeze(1), 1)
        one_hot_encoded.scatter_(1, cell_indices.unsqueeze(1), 1)
        one_hot_encoded.scatter_(1, direction_indices.unsqueeze(1), 1)
        one_hot_encoded.scatter_(1, split_indices.unsqueeze(1), 1)

        return one_hot_encoded            
    def getUCBscore(self): 
        if self.N == 0:  # if unexplored nodes, maximum probability
            return  float('inf')
        top_node = self
        if top_node.parent: top_node = top_node.parent    
        return (self.T/self.N) + sqrt(log(top_node.N)/self.N)
    def detach_parent(self):
        del self.parent 
        self.parent = None 
    def process_task(self, index, action, game):
        observation, info = game.step(action)
        return index, Node(game, game.is_done(), None, observation["MCTS"].as_dict(), action["MCTS"])
    def create_child(self):
        if self.done: 
            #print("self.done in create child")
            return 
        actions = []
        games = []
        mask = self.observation["action_mask"]
        valid_actions = np.argwhere(mask == 1)  
        if loud and len(valid_actions)==0: print("No valid actions in create_child")      
        
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
                index, node = t.result()
                child[index] = node
        self.child = child
        # if loud: print("create child number",len(child))

    def explore(self):
        # find leaf node by choosing nodes with max U
        current = self
        while current.child and self.done == False:
            child = current.child 
            max_U = max(c.getUCBscore() for c in child.values()) # c in Node type
            actions = [a for a,c in child.items() if c.getUCBscore() == max_U]
            if len(actions) == 0:
                print("error zero length", max_U)
            action = random.choice(actions)
            current = child[action]
        # play a random game, or expand if needed
        if current.N < 1:  ## questions, if not visit, calculate T(total value), if visited, create child
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(list(current.child.values())) # random.choice(current.child)
            current.T = current.T + current.rollout()
        current.N += 1
        # update statistice and backpropagate
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T
    def rollout(self):
        mcts_observation = self.observation
        if self.done:
            #print("self.done in rollout")
            if self.game.agent_won('MCTS'):   # TO DO: I guess, or new_game.agents[0]
                if loud: print("MCTS rollout")
                return 1/mcts_observation["observation"]["timestep"]
            else:
                if loud: print("Expander rollout")
                return -1/mcts_observation["observation"]["timestep"]
        step = 0
        max_step = 20 ### To improve: truncated number
        new_game = deepcopy(self.game)
        npc = ExpanderAgent()
        npc_observation = new_game.agent_observation("Expander").as_dict()
        # directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        while not new_game.is_done():
            step += 1
            if step >= max_step:   ## To imporve: truncated score reward
                return -1/max_step
            mask = mcts_observation["action_mask"]
            valid_actions = np.argwhere(mask == 1)
            if len(valid_actions) == 0:  # No valid actions
                action =  {
                "pass": 1,
                "cell": np.array([0, 0]),
                "direction": 0,
                "split": 0,
            }
                #if loud: print("no valid action option in MCTS rollout")
            elif self.model == None:
                #if loud: print("valid action in MCTS rollout")    ### better model?->how about a RL model

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
                encode_obs = torch.FloatTensor(encode_obs).to(device)#.unsqueeze(0)
                t_va = torch.tensor(valid_actions)
                t_validaction = torch.concat((self.encode_valid_action(t_va, 0), self.encode_valid_action(t_va, 1)), dim = 0).to(device)
                
                with torch.no_grad():
                    q_values = self.model.compute_Qvalues(encode_obs)
                    valid_q = torch.matmul(q_values,t_validaction.float().transpose(0, 1))
                    index=valid_q.multinomial(num_samples=1).item()
                    action = self.model.decode_action(t_validaction[index], 5, 5)#self.model.compute_argmaxQ(encode_obs)

            observation, info = new_game.step({"MCTS": action, "Expander": npc.act(npc_observation)})  ## problem? need to mimic both agent and npc
            mcts_observation = observation["MCTS"].as_dict()
            npc_observation = observation["Expander"].as_dict()
            if new_game.is_done():

                if new_game.agent_won('MCTS'):   # TO DO: I guess, or new_game.agents[0]
                    if loud: print("MCTS rollout")
                    return 1/mcts_observation["observation"]["timestep"]
                else:
                    if loud: print("Expander rollout")
                    return -1/mcts_observation["observation"]["timestep"]



    def next(self):
        if not self.child: 
            print("no child in MCTS")
            action = {
                    "pass": 1,
                    "cell": (0, 0),
                    "direction": 0,
                    "split": 0,
                }
            return self, action 
        child = self.child 
        max_av_T = max(node.T/(node.N+0.01) for node in child.values())
        max_children = [c for a,c in child.items() if c.T/(c.N+0.01) == max_av_T]
        if len(max_children) == 0:
            print("error zero length", max_av_T)
        max_child = random.choice(max_children)
        return max_child, max_child.action_index


MCTS_POLICY_EXPLORE = 50   ### To improve: number of exploration

def Policy_MCTS(mytree):
    for i in range(MCTS_POLICY_EXPLORE):
        if loud: print("exploring", i)
        mytree.explore()
    next_tree, next_action = mytree.next()
    next_tree.detach_parent()
    return next_tree, next_action

class MCTSAgent(Agent):
    def __init__(self, id: str = 'MCTS', color: tuple[int, int, int] = (0, 145, 0)):
        super().__init__(id, color)
        self.current = None
    
    def encode_valid_action(self, valid_actions, split = 0, grid_width = 5, grid_height = 5):
        total_cells = grid_width * grid_height
        total_directions = 4
        total_splits = 2
        total_passes = 2  # Assuming 'pass' can be 0 or 1

        num_actions = valid_actions.shape[0]
        size = total_cells + total_directions + total_splits + total_passes

        # Indices for each component
        pass_indices = torch.zeros(num_actions, dtype=torch.long)
        cell_indices = valid_actions[:, 0] * grid_width + valid_actions[:, 1] + total_passes
        direction_indices = valid_actions[:, 2] + total_passes + total_cells
        split_indices = valid_actions[:, 2] + total_passes + total_cells + split  # Assuming split relates to direction

        # Creating one-hot encoded tensor
        one_hot_encoded = torch.zeros(num_actions, size, dtype=torch.int)
        
        # Using index_fill_ to set values at specific indices
        one_hot_encoded.scatter_(1, pass_indices.unsqueeze(1), 1)
        one_hot_encoded.scatter_(1, cell_indices.unsqueeze(1), 1)
        one_hot_encoded.scatter_(1, direction_indices.unsqueeze(1), 1)
        one_hot_encoded.scatter_(1, split_indices.unsqueeze(1), 1)

        return one_hot_encoded

    def act(self, observation: Observation, game: Game, done: bool, prev_act: Action, train = True) -> Action:  # TO DO: the type of game 
        ### TODO: define and train the MCTS model
        if self.current == None: # create root
            Node.set_shared_model("q_function_model.pth", observation,(5, 5))
            self.current = Node(game, done, None, observation, None)
        new_node = Node(game, done, self.current, observation, prev_act)
        self.current = new_node
        # self.current.observation = observation  ### do we need refresh current node?
        # self.current.game = game
        # if debuging:
        mask = observation["action_mask"]
        # observation = observation["observation"]
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
        elif train:
            if self.current.done == True: print("current done in act")
            child_node, action = Policy_MCTS(self.current.copynode())#Policy_MCTS(deepcopy(self.current))  ## TO DO: from where do MCTS
            self.current.child = child_node
            child_node.parent = self.current
            self.current = self.current.child
            # observation, info = game.step(action)  # maybe action["MCTS"]
        else: ## not train
            self.current = Node(game, done, None, observation, None)
            encode_obs = self.current.shared_model.encode_observation(observation, with_mask=True)
            encode_obs = torch.FloatTensor(encode_obs).to(device)#.unsqueeze(0)
            with torch.no_grad():
                t_va = torch.tensor(valid_actions)
                t_validaction = torch.concat((self.encode_valid_action(t_va, 0), self.encode_valid_action(t_va, 1)), dim = 0).to(device)
                q_values = self.current.shared_model.compute_Qvalues(encode_obs)
                valid_q = torch.matmul(q_values,t_validaction.float().transpose(0, 1))
                #index=valid_q.multinomial(num_samples=1).item()
                action = self.current.shared_model.decode_action(t_validaction[valid_q.argmax()], 5, 5)#.compute_argmaxQ(encode_obs)
            if loud: print('argmax action',action)
        return action 
    def reset(self):
        pass


    