import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dueling, fc1_units=64, fc2_units=64):
        #fc1_units=64, fc2_units=64
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.dueling = dueling
        if self.dueling == False:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
        else:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc_value = nn.Linear(fc1_units, fc2_units)
            self.fc_adv = nn.Linear(fc1_units, fc2_units)
            self.value = nn.Linear(fc2_units, action_size)
            self.adv = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        if self.dueling == False:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            Q = self.fc3(x)
        else:
            y = F.relu(self.fc1(state))
            value = F.relu(self.fc_value(y))
            adv = F.relu(self.fc_adv(y))
    
            value = self.value(value)
            adv = self.adv(adv)
    
            advAverage = torch.mean(adv, dim=1, keepdim=True)
            Q = value + adv - advAverage
        
        return Q
