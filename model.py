import torch
import torch.nn as nn
import torch.nn.functional as F


#USE_CUDA = torch.cuda.is_available()
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QNetwork(nn.Module):

    """Class for neural network
    Attributes:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and model architecture"""
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Forward propagation of neural network
        Args:
            state (vector): sized (self.state_size x batch size) with environment state data
        Returns:
            Vector sized (self.action_size x batch size) with return of a neural netowrk
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, seed):
        super(DuelingDQN, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
    
    def act(self, state):
 #       state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]

        return action