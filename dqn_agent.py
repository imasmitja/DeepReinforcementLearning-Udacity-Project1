import numpy as np
import random
from collections import namedtuple, deque
import time


from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.7 #0.7             # if exp_a = 0, pure uniform random from replay buffer. if esp_a = 1, only uses priorities from replay buffer
BETA = 0.4 #0.4
INC_BETA = 0.001
MAX_BETA = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, number_steps = 1000, ddqn=False, per=False, dueling=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, dueling).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if per == False:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer_SummTree(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        #Method used ex: 'dqn', 'ddqn', etc
        self.ddqn = ddqn #Double DQN
        self.per = per #Priorized Experience Replay
        self.dueling = dueling
        
        self.priority = 1. #initial priority
        self.beta = BETA
        self.max_beta = MAX_BETA
        self.inc_beta = INC_BETA
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, self.priority)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indexes = experiences
        
        ## TODO: compute and minimize the loss
        
        # Get max predicted Q values (for next states) from target model        
        if self.ddqn == True:
            #Double DQN (DDQN):
            indx = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(0)[0]
            rows = np.array(range(len(indx)))
            Q_targets_next = self.qnetwork_target(next_states).detach()[rows,indx].unsqueeze(1)
        else:          
            #DQN:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # import pdb; pdb.set_trace()
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        if self.per == True:
            # Priorized Experience Replay
            new_priorities = abs(Q_targets - Q_expected) + 0.1 #we introduce a fixed small constant number to avoid priorities = 0.
            self.priority = self.memory.update(indexes, new_priorities)
            adjust = 1/(new_priorities)**(self.beta)
            if self.beta < self.max_beta:
                self.beta += self.inc_beta
            else:
                self.beta = self.max_beta
            Q_expected *= adjust
            Q_targets *= adjust

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        "*** YOUR CODE HERE ***"
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.sample_indx = 0
        # self.weights = deque(maxlen=buffer_size) #memory of priorizations
        self.weights = [] #memory of priorizations
        self.buffer_size = buffer_size
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.weights.append(priority)
        if len(self.weights) > self.buffer_size:
            self.weights.pop(0)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #Take random samples
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
    
    def update(self,priorities):
        """ Update the priorities of the sampled experiences to the memory """
        # import pdb; pdb.set_trace() 
        for i in range(len(priorities)):
            self.weights[self.sample_indx[i]] = priorities[i].item() #delate old element from deque
        return max(self.weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    

class ReplayBuffer_SummTree:   # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity, batch_size, seed):
        self.tree = SumTree(capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.min_error = 0.01
        self.alpha = ALPHA
        self.seed = random.seed(seed)
        self.batch_size = batch_size
       
    def _getPriority(self, error):
        return (error + self.min_error) ** self.alpha
     
    def add(self, state, action, reward, next_state, done, error):
        p = self._getPriority(error)
        e = self.experience(state, action, reward, next_state, done)
        self.tree.add(p, e) 
     
    def sample(self):
        n = self.batch_size
        experiences = []
        indexes = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            experiences.append(data)
            indexes.append(idx)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones, indexes)
     
    def update(self, idx, error):
        for i in range(len(idx)):
            p = self._getPriority(error[i].cpu().item())
            self.tree.update(idx[i], p)
        return self.tree.total()
        
    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.write
    
    
#Additional Info when using cuda
def cuda_info ():
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')