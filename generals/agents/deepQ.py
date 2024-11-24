import torch 
import numpy as np 
from collections import deque 
loud = True
class Qfunction(object):
    def __init__(self, gridsize, lr):
        self.gridsize = gridsize 
        self.lr = lr 
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

    def encode_action(self, action, grid_width, grid_height):
        _pass = action["pass"]
        _split = action["split"]
        _cell = action["cell"]
        _direction = action["direction"]
        flat_index = (
            _pass * grid_width * grid_height * 4 * 2 +
            (_cell[0] * grid_width + _cell[1]) * 4 * 2 +
            _direction * 2 +
            _split
        )
        return flat_index
        # oh_pass = torch.tensor([1, 0] if _pass == 0 else [0, 1])
        # oh_split = torch.tensor([1, 0] if _split == 0 else [0, 1])
        # flat_index = _cell[0] * grid_width + _cell[1]
        # oh_cell = torch.zeros(grid_width * grid_height)
        # oh_cell[flat_index] = 1
        # oh_direction = torch.zeros(4) 
        # oh_direction[_direction] = 1
        # return torch.cat([oh_pass, oh_cell, oh_direction, oh_split])
    def decode_action(self, flat_index, grid_width, grid_height):
        total_cells = grid_width * grid_height
        total_directions = 4
        total_splits = 2

        # Decode components
        _split = flat_index % total_splits
        flat_index //= total_splits

        _direction = flat_index % total_directions
        flat_index //= total_directions

        flat_cell_index = flat_index % total_cells
        _cell = np.array([flat_cell_index // grid_width, flat_cell_index % grid_width])
        flat_index //= total_cells

        _pass = flat_index

        return {
            "pass": _pass,
            "cell": _cell,
            "direction": _direction,
            "split": _split
        }

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
            state_vector = np.concatenate([state_vector, mask])
        return state_vector 
    def compute_Qvalues(self, states, actions):
        # states = self.encode_observation(states) ### attention, do I need this
        states = torch.FloatTensor(states)
        q_preds = self.model(states)
        # action_onehot = self.encode_action(actions, self.gridsize[0], self.gridsize[1])  ### attention for actsize!
        if loud: print("q_preds size", q_preds.shape, "action size ", actions.shape)
        q_preds_selected = q_preds.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)#torch.sum(q_preds * actions, axis = -1)
        return q_preds_selected 
    def compute_maxQvalues(self, states):
        # states = self.encode_observation(states) ### attention. do i need it
        states = torch.FloatTensor(states)
        Qvalues = self.model(states).detach()  # Avoid .cpu().numpy()
        q_preds = torch.max(Qvalues, dim=1)[0]  # Use PyTorch max
        return q_preds
    def compute_argmaxQ(self, states):
        # states = self.encode_observation(states) ### attention. do i need it
        states = torch.FloatTensor(states)
        Qvalues = self.model(states).detach().cpu().numpy()  ### attention, do we need detach
        q_action_index = np.argmax(Qvalues.flatten())
        action_dict = self.decode_action(q_action_index, self.gridsize[0], self.gridsize[1])
        return action_dict 
    def train(self, states, actions, targets):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)
        q_preds_selected = self.compute_Qvalues(states, actions)
        loss = torch.mean((q_preds_selected - targets)**2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.numpy()

