import torch
import numpy as np
from copy import deepcopy

def epsilon_greedy(agent_q: torch.Tensor, action_mask: np.ndarray, epsilon: float):
    """
    Performs epsilon-greedy action selection based on the given Q-values and action mask.

    Args:
    agent_q (torch.Tensor): The Q-values for each action computed by the agent.
    action_mask (np.ndarray): A boolean array where True indicates a valid action and False an invalid one.
    epsilon (float): The probability of choosing a random action.

    Returns:
    int: The selected action index.
    """
    if np.random.rand() < epsilon:
        # Exploration: Randomly select from the available actions
        valid_actions = np.where(action_mask)[0]  # Get indices of available actions
        if valid_actions.size == 0:
            raise ValueError("No valid actions available.")
        action = np.random.choice(valid_actions)
    else:
        # Exploitation: Select the action with the highest Q-value among the valid ones
        # Set Q-values of invalid actions to a very low number to ensure they are not selected
        masked_q_values = agent_q.clone()  # Clone to avoid modifying the original tensor
        masked_q_values[action_mask == 0] = float('-inf')  # Invalidate the masked actions
        action = torch.argmax(masked_q_values).item()  # Get the index of the highest Q-value

    return action


class EpsilonManager:
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.995, decay_type='exponential'):
        """
        Initializes the EpsilonManager with specified decay parameters.

        Args:
        epsilon_start (float): The initial epsilon value at the start of training.
        epsilon_min (float): The minimum epsilon value to which it should decay.
        decay_rate (float): The rate at which epsilon decays each episode; interpreted as linear decay rate
                            or exponential decay base depending on `decay_type`.
        decay_type (str): Type of decay ('linear' or 'exponential').
        """
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.epsilon = epsilon_start
        self.episode_count = 0

    def get_epsilon(self):
        """
        Returns the current epsilon value.
        """
        return self.epsilon

    def update_epsilon(self):
        """
        Updates the epsilon value based on the specified decay type and increments the episode count.
        """
        if self.decay_type == 'linear':
            # If the type is linear, treat decay_rate as anneal time (in episodes).
            self.epsilon = max(self.epsilon_start - ((1-self.epsilon_min) / (self.decay_rate) * self.episode_count), self.epsilon_min)
        elif self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon_start * (self.decay_rate ** self.episode_count), self.epsilon_min)

        self.episode_count += 1

    def reset(self):
        """
        Resets the epsilon to the initial value and resets the episode counter, typically used when restarting training.
        """
        self.epsilon = self.epsilon_start
        self.episode_count = 0

def boltzmann_policy(agent_q: torch.Tensor, action_mask, temperature: float) -> int:
    if temperature < 1e-5:
        temperature = 1e-5

    # Convert action_mask to a PyTorch tensor and move it to the same device as agent_q
    device = agent_q.device
    action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)

    # Subtract the maximum Q-value for numerical stability
    max_q = torch.max(agent_q)

    # Compute the exponentials of the adjusted Q-values
    exp_q = torch.exp((agent_q - max_q) / temperature)

    # Apply the action mask to zero out invalid actions
    exp_q = exp_q * action_mask_tensor

    # Normalize to get the probability distribution
    sum_exp_q = torch.sum(exp_q)
    if sum_exp_q == 0:
        raise ValueError("All actions are masked out, cannot sample an action.")
    boltzmann_q = exp_q / sum_exp_q

    # Sample an action based on the probabilities
    sampled_action = torch.multinomial(boltzmann_q, 1).item()

    return sampled_action

def greedy_action_policy(q_values: torch.Tensor, action_mask) -> int:
    device = q_values.device
    q_values[action_mask==0] = -1e5
    sampled_action = torch.argmax(q_values).item()

    return sampled_action

def get_action_from_q(q_values, n_actions_list, batch_size):
    # Flatten the Q-values to find the max
    q_values_flat = q_values.view(batch_size, -1)  # Shape: (batch_size, total_joint_actions)
    max_q_values, max_indices = q_values_flat.max(dim=1)  # Shape: (batch_size,)

    # Convert flat indices back to action indices for each agent
    joint_action_indices = []
    for idx in max_indices:
        idx = idx.item()
        actions = []
        for n_actions in reversed(n_actions_list):
            actions.append(idx % n_actions)
            idx = idx // n_actions
        joint_action_indices.append(list(reversed(actions)))

    # joint_action_indices: List of lists, each containing the action indices for each agent
    return np.array(joint_action_indices).squeeze()

def centralized_epsilon_greedy(agent_q: torch.Tensor, observation, epsilon: float, n_actions_list):
    mask = []
    available_action = []
    for i, a in enumerate(observation.keys()):
        mask.append(observation[a]['action_mask'])
        available_action.append(np.where(mask[i])[0])

    if np.random.rand() < epsilon:
        action = np.array([np.random.choice(avail_act) for avail_act in available_action])
    else:
        action = get_action_from_q(agent_q, n_actions_list, batch_size=1)

    return action

class DistributedActionController():
    def __init__(self, timeline):
        # timeline : A boolean array (1, len_episode). If an element is True, take greedy action. Else, take action 2 (No-Op).
        self.timeline       = timeline
        self.action_count   = 0

    def take_action(self, agent_qs, action_mask):
        sampled_action = {}
        for a in self.timeline.keys():
            if self.timeline[a][self.action_count]:
                sampled_action[a] = greedy_action_policy(agent_qs[a], action_mask[a])
            else:
                sampled_action[a] = 2
        self.action_count += 1
        
        return sampled_action
    
    def reset_action_count(self):
        self.action_count = 0
    
class CentralizedDistributedActionController():
    '''Poor name. I admit.'''
    def __init__(self, timeline):
        # timeline : A boolean array (n_agents, len_episode). If an element is True, take greedy action for the agent. Else, take action 2 (No-Op).
        self.timeline       = timeline
        self.action_count   = 0
        self.no_op          = np.array([0,0,1,0,0], dtype=np.int8)
        self.agents_dict    = {
            0: 'influent_flowrate',
            1: '1st_stage_pump',
            2: '2nd_stage_pump'
        }
    
    def take_action(self, agent_q, observation, n_actions_list):
        observation_copy = deepcopy(observation)
        for agent in self.agents_dict.keys():
            if self.timeline[agent, self.action_count]:
                observation_copy[agent]['action_mask'] = self.no_op.copy()
        
        sampled_action = centralized_epsilon_greedy(agent_q, observation_copy, epsilon=0.0, n_actions_list=n_actions_list)
        self.action_count += 1
        return sampled_action
    
    def reset_action_count(self):
        self.action_count = 0
