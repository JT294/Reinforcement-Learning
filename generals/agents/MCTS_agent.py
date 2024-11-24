import numpy as np
import random
import itertools
from math import *
from copy import deepcopy 
import weakref
import gymnasium as gym
from concurrent.futures import ThreadPoolExecutor

from generals.core.config import Direction
from generals.core.game import Action, Game 
from generals.core.observation import Observation 

from .agent import Agent
from .expander_agent import ExpanderAgent 

loud = False 
debuging = False
class Node:
    def __init__(self, game, done, parent, observation, action_index):
        self.child = None 
        self.T = 0  # total rewards from MCTS exploration
        self.N = 0  # visit count
        self.game = game # environment
        self.observation = observation  # observation of environment
        self.done = done # win/loss/draw ## True/False now
        self.parent = weakref.ref(parent) if parent is not None else None # link to parent
        self.action_index = action_index # action leads to current node
        self.model = None  ### TO DO: want to train a deepQ model
    
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
        if self.done: return 
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
        # for action, game in zip(actions, games):
        #     observation, info = game.step(action)
        #     child[index] = Node(game, game.is_done(), self, observation["MCTS"].as_dict(), action["MCTS"])  ### need to check the obervation part
        #     index += 1
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
        if self.done:
            return 0
        step = 0
        max_step = 100 ### To improve: truncated number
        new_game = deepcopy(self.game)
        npc = ExpanderAgent()
        mcts_observation = self.observation
        npc_observation = new_game.agent_observation("Expander").as_dict()
        # directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        while not new_game.is_done():
            step += 1
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
            else:
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

            observation, info = new_game.step({"MCTS": action, "Expander": npc.act(npc_observation)})  ## problem? need to mimic both agent and npc
            mcts_observation = observation["MCTS"].as_dict()
            npc_observation = observation["Expander"].as_dict()
            if new_game.is_done():

                if new_game.agent_won('MCTS'):   # TO DO: I guess, or new_game.agents[0]
                    if loud: print("MCTS rollout")
                    return 1
                else:
                    if loud: print("Expander rollout")
                    return -1
            if step >= max_step:   ## To imporve: truncated score reward
                status = new_game.get_infos()
                mct_army, mct_land = status["MCTS"]["army"], status["MCTS"]["land"]
                npc_army, npc_land = status["Expander"]["army"], status["Expander"]["land"]
                army_score = mct_army/(mct_army+npc_army)
                land_score = mct_land/(mct_land+npc_land)
                return 0.7 * land_score + 0.3 * army_score # (2 * army_score * land_score) / (army_score + land_score)



    def next(self):
        # if self.done: raise ValueError("game has ended")
        if not self.child: return self, self.action_index# raise ValueError("no children found and game hasn't ended")
        child = self.child 
        max_N = max(node.N for node in child.values())
        max_children = [c for a,c in child.items() if c.N == max_N]
        if len(max_children) == 0:
            print("error zero length", max_N)
        if loud and len(max_children) == 1: print("only one child in next")
        max_child = random.choice(max_children)
        return max_child, max_child.action_index

MCTS_POLICY_EXPLORE = 20   ### To improve: number of exploration

def Policy_MCTS(mytree):
    # if mytree.done: return mytree, mytree.action_index ### is it right?
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
    
    def act(self, observation: Observation, game: Game, done: bool, prev_act: Action) -> Action:  # TO DO: the type of game 
        ### TODO: define and train the MCTS model
        if self.tree == None: # create root
            self.tree = Node(game, done, None, observation, None)  ### TO DO: parent?
            self.current = self.tree
        self.current.observation = observation  ### do we need refresh current node?
        self.current.game = game
        # if debuging:
        mask = observation["action_mask"]
        observation = observation["observation"]
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
            child_node, action = Policy_MCTS(deepcopy(self.current))  ## TO DO: from where do MCTS
            self.current.child = child_node
            child_node.parent = self.current
            self.current = self.current.child
            # observation, info = game.step(action)  # maybe action["MCTS"]

        return action 
    def reset(self):
        pass