import torch 
import numpy as np 
from collections import deque 
import torch.nn as nn
loud = False
class Qfunction(object):
    def __init__(self, gridsize, lr):
        self.gridsize = gridsize 
        self.lr = lr 
        self.all_action = self.get_action_onehot(gridsize[0], gridsize[1])
    def initialize_model(self, obssize, actsize):
        self.obssize = obssize
        self.actsize = actsize
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obssize, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, actsize)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        # self.criterion = nn.CrossEntropyLoss()
        
    def get_action_onehot(self, grid_width, grid_height):
        actions = []
        for pas in range(2):
            for x in range(grid_width):
                for y in range(grid_height):
                    for direction in range(4):
                        for split in range(2):
                            action = {'pass': pas, 'cell': (x, y), 'direction': direction, 'split': split}
                            actions.append(self.encode_action(action, grid_width, grid_height))
        return torch.tensor(actions)

    def encode_action(self, action, grid_width, grid_height):
        total_cells = grid_width * grid_height
        total_directions = 4
        total_splits = 2
        total_passes = 2  # Assuming 'pass' can be 0 or 1

        # Create a zero-filled array for one-hot encoding
        size = total_cells + total_directions + total_splits + total_passes
        one_hot_encoded = np.zeros(size, dtype=int)

        # Calculate indices for one-hot encoding
        pass_index = action["pass"]
        cell_index = total_passes + (action["cell"][0] * grid_width + action["cell"][1])
        direction_index = total_passes + total_cells + action["direction"]
        split_index = total_passes + total_cells + total_directions + action["split"]

        # Set indices to 1
        one_hot_encoded[pass_index] = 1
        one_hot_encoded[cell_index] = 1
        one_hot_encoded[direction_index] = 1
        one_hot_encoded[split_index] = 1

        return one_hot_encoded

    def decode_action(self, one_hot_encoded, grid_width, grid_height):
        total_cells = grid_width * grid_height
        total_directions = 4
        total_splits = 2
        total_passes = 2  # Assuming 'pass' can be 0 or 1

        # Extract indices from one-hot encoded vector
        _pass = np.argmax(one_hot_encoded[:total_passes])
        cell_index = np.argmax(one_hot_encoded[total_passes:(total_passes + total_cells)])
        _direction = np.argmax(one_hot_encoded[(total_passes + total_cells):(total_passes + total_cells + total_directions)])
        _split = np.argmax(one_hot_encoded[-total_splits:])

        _cell = np.array([cell_index // grid_width, cell_index % grid_width])

        return {
            "pass": _pass,
            "cell": _cell,
            "direction": _direction,
            "split": _split
        }

    # def encode_action(self, action, grid_width, grid_height):
    #     _pass = action["pass"]
    #     _split = action["split"]
    #     _cell = action["cell"]
    #     _direction = action["direction"]
    #     flat_index = (
    #         _pass * grid_width * grid_height * 4 * 2 +
    #         (_cell[0] * grid_width + _cell[1]) * 4 * 2 +
    #         _direction * 2 +
    #         _split
    #     )
    #     return flat_index

    # def decode_action(self, flat_index, grid_width, grid_height):
    #     total_cells = grid_width * grid_height
    #     total_directions = 4
    #     total_splits = 2

    #     # Decode components
    #     _split = flat_index % total_splits
    #     flat_index //= total_splits

    #     _direction = flat_index % total_directions
    #     flat_index //= total_directions

    #     flat_cell_index = flat_index % total_cells
    #     _cell = np.array([flat_cell_index // grid_width, flat_cell_index % grid_width])
    #     flat_index //= total_cells

    #     _pass = flat_index

    #     return {
    #         "pass": _pass,
    #         "cell": _cell,
    #         "direction": _direction,
    #         "split": _split
    #     }

    def encode_observation(self, obs, with_mask=True):
        observation = obs["observation"] if with_mask else obs 
        features = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                features.append(value.flatten())
            elif isinstance(value, (int, float, np.int64, np.float64)):
                features.append(np.array([value]))
            else:
                raise ValueError(f"Unsupported type for key {key}: {type(value)}")
        state_vector = np.concatenate(features, axis=0)
        if with_mask and "action_mask" in obs:
            mask = obs["action_mask"].flatten()
            state_vector = np.concatenate([state_vector, mask])  ### attention. need to code without mask
        return state_vector 
    def compute_Qvalues(self, states, actions=None):
        # states = self.encode_observation(states) ### attention, do I need this
        states = torch.FloatTensor(states)
        q_preds = self.model(states)
        if actions == None: return q_preds
        # action_onehot = self.encode_action(actions, self.gridsize[0], self.gridsize[1])  ### attention for actsize!
        if loud: print("q_preds size", q_preds.shape, "action size ", actions.shape)
        q_preds_selected = torch.sum(q_preds * actions, axis = -1)#q_preds.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)#
        return q_preds_selected 
    def compute_maxQvalues(self, states):
        # states = self.encode_observation(states) ### attention. do i need it
        states = torch.FloatTensor(states)
        Qvalues = self.model(states).detach()  # Avoid .cpu().numpy()
        allQ = torch.matmul(Qvalues, self.all_action.float().transpose(0, 1))
        q_preds = torch.max(allQ, dim=1)[0]  # Use PyTorch max    

        return q_preds
    def compute_argmaxQ(self, states):
        # states = self.encode_observation(states) ### attention. do i need it
        states = torch.FloatTensor(states)
        Qvalues = self.model(states).detach().cpu().numpy()  ### attention, do we need detach
        # q_action_index = np.argmax(Qvalues.flatten())
        action_dict = self.decode_action(Qvalues, self.gridsize[0], self.gridsize[1])
        return action_dict 
    # def 
    def train(self, states, actions, targets):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)
        q_preds_selected = self.compute_Qvalues(states, actions)
        # loss = self.criterion(q_preds_selected, targets)
        loss = torch.mean((q_preds_selected - targets)**2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.numpy()

