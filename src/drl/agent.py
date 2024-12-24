import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
from ..config import Config

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        hidden_sizes = [1024, 512, 256]
        dropout_rate = 0.15
        
        # Feature extraction với residual connections
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LayerNorm(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[2], hidden_sizes[2]//2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2]//2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[2], hidden_sizes[2]//2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2]//2, action_size)
        )
        
        # Noisy layers cho exploration
        self.noisy_value = NoisyLinear(hidden_sizes[2]//2, 1)
        self.noisy_advantage = NoisyLinear(hidden_sizes[2]//2, action_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        features = self.features(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling architecture combination
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_increment = 0.001
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = None):
        if len(self.buffer) == 0:
            return None
            
        if beta is None:
            self.beta = min(1.0, self.beta + self.beta_increment)
            beta = self.beta
            
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        total = len(self.buffer)
        weights = (total * probs) ** (-beta)
        weights /= weights.max()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        weights = weights[indices]
        
        batch = list(zip(*samples))
        states = np.array(batch[0], dtype=np.float32)
        actions = np.array(batch[1], dtype=np.int64)
        rewards = np.array(batch[2], dtype=np.float32)
        next_states = np.array(batch[3], dtype=np.float32)
        dones = np.array(batch[4], dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones), indices, weights
        
    def update_priorities(self, indices, priorities):
        """Cập nhật priorities cho các transitions đã sample
        
        Args:
            indices: Mảng các indices cần update
            priorities: Mảng các priority mới tương ứng
        """
        # Đảm bảo priorities là mảng numpy
        if not isinstance(priorities, np.ndarray):
            priorities = np.array(priorities)
            
        # Nếu priorities là số đơn lẻ, chuyển thành mảng
        if priorities.ndim == 0:
            priorities = np.full_like(indices, priorities)
            
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) + 1e-6
            self.max_priority = max(self.max_priority, float(priority))
            
    def __len__(self):
        return len(self.buffer)
        
    def is_full(self):
        return len(self.buffer) == self.capacity

class DRLAgent:
    def __init__(self, state_size: int, action_size: int, 
                 batch_size: int = 256,
                 learning_rate: float = 0.0003,
                 memory_size: int = 100000):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = learning_rate
        self.tau = 0.001
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.steps_done = 0

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
            nn.ReLU(), 
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, self.action_size)
        )

    def remember(self, state, action, reward, next_state, done, priority=None):
        experience = (state, action, reward, next_state, done)
        if priority:
            self.memory.append((priority, experience))
        else:
            self.memory.append(experience)

    def act(self, state, noise_scale=1.0):
        if random.random() < self.epsilon * noise_scale:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample with priorities if available
        if isinstance(self.memory[0], tuple) and len(self.memory[0]) == 2:
            priorities, experiences = zip(*self.memory)
            probs = np.array(priorities) / sum(priorities)
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            batch = [experiences[idx] for idx in indices]
        else:
            batch = random.sample(self.memory, self.batch_size)
            
        # Convert batch to numpy arrays first
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards + self.gamma * next_q_values.squeeze() * (~dones).float()
            
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Huber loss for stability
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        if self.steps_done % 100 == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), 
                                                self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1 - self.tau) * target_param.data
                )
                
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        
        return loss.item()
        
    def save_model(self, path):
        """Lưu model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
    def get_training_stats(self):
        """Return training statistics"""
        return self.training_stats