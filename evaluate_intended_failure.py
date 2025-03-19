import os
os.chdir("/home/ybang-eai/research/2024/ROMARL/ROMARL")
from TwoStageROProcessEnvironment.env.PressureControlledTwoStageROProcess_simple import TwoStageROProcessEnvironment
from algorithms.mixer.QMIX import QMixer, VDN, ReplayBuffer, PrioritizedExperienceReplay, RNNAgent, CentralizedRNNAgent, mask_and_softmax, softmax_and_mask, mask_and_nothing, centralized_mask_and_nothing
import numpy as np
import torch
from copy import copy, deepcopy
from utils.epsilon_greedy import epsilon_greedy, EpsilonManager, boltzmann_policy, greedy_action_policy, get_action_from_q, centralized_epsilon_greedy, DistributedActionController, CentralizedDistributedActionController
from matplotlib import pyplot as plt
import pandas as pd
import os
from datetime import datetime
import argparse
from tqdm import tqdm
import pickle as pkl

def load_model_parameters(mixer, agent_nets, directory):
    """
    Load the parameters of the mixing network and agent networks from the specified directory.

    Args:
        mixer (torch.nn.Module): The mixing network to which parameters will be loaded.
        agent_nets (dict): Dictionary of agent networks to which parameters will be loaded.
        directory (str): The directory from where the model parameters should be loaded.
    """
    # Load mixer parameters
    mixer_state_dict = torch.load(os.path.join(directory, 'mixer_params.pt'))
    mixer.load_state_dict(mixer_state_dict)

    # Load parameters for each agent network
    for agent_id, agent in agent_nets.items():
        agent_state_dict = torch.load(os.path.join(directory, f'agent_{agent_id}_params.pt'))
        agent.load_state_dict(agent_state_dict)


def load_parameters_at_intervals_sequential(root_path: str, interval: int, device, start_index: int = 0, window = None) -> dict:
    """
    Loads .pt parameters from directories at specified intervals sequentially.

    Parameters:
    - root_path (str): The path to the "parameters" directory.
    - interval (int): The interval at which to load parameters (e.g., every 10 steps).

    Returns:
    - dict: A dictionary where keys are training steps and values are dictionaries of parameters.
    """
    root_path = os.path.abspath(root_path)
    parameters_dict = {}

    # List all directories in the root path
    step_dirs = [d for d in os.listdir(root_path) if d.isdigit()]

    # Sort directories numerically
    step_dirs.sort(key=int)

    if window is not None:
        step_dirs = step_dirs[:window]

    # Iterate over directories at specified intervals
    for step in tqdm(range(start_index, len(step_dirs)-1, interval)):
        step_dir_name = step_dirs[step]
        step_dir_path = os.path.join(root_path, step_dir_name)

        # List all .pt files in the directory
        pt_files = [os.path.join(step_dir_path, f) for f in os.listdir(step_dir_path) if f.endswith('.pt')]

        # Initialize a dictionary to hold parameters for this step
        step_parameters = {}

        # Load each .pt file sequentially
        for file_path in pt_files:
            parameter_name = os.path.splitext(os.path.basename(file_path))[0]
            parameter_data = torch.load(file_path, map_location=torch.device(device))
            step_parameters[parameter_name] = parameter_data

        # Store in the parameters dictionary
        parameters_dict[int(step_dir_name)] = step_parameters

    return parameters_dict

def generate_distributed_control_scenario(agents, max_control_timestep, test_type, hindered_agent = None, period=None, seed=None):
    distributed_scenario = dict.fromkeys(agents, np.ones(max_control_timestep))

    def hinder_one_all(distributed_scenario, hindered_agent):
        if hindered_agent is not None:
            agent_to_hinder = hindered_agent
            scenario = deepcopy(distributed_scenario)
            scenario[agent_to_hinder] = np.zeros_like(scenario[agent_to_hinder])
        else:
            scenario = deepcopy(distributed_scenario)
        return scenario
    
    def hinder_one_given_period(distributed_scenario, period):
        if period is None:
            period = 4  #  1 day
        scenario = deepcopy(distributed_scenario)
        timestamp = 0
        # Loop through scenario and hinder random agent for given period.
        while timestamp < max_control_timestep:
            agent_to_hinder = np.random.choice(agents)
            scenario[agent_to_hinder][timestamp:timestamp+period] = False
            timestamp += period

        # If there exists left portion that is not hindered, hinder it.
        if timestamp != max_control_timestep:
            agent_to_hinder = np.random.choice(agents)
            scenario[agent_to_hinder][timestamp:] = False

        return scenario

    scenario_setting = {
        "HinderOneAll": hinder_one_all(distributed_scenario, hindered_agent),
        "HinderOneGivenPeriod": hinder_one_given_period(distributed_scenario, period),
    }

    return scenario_setting[test_type]

def read_state_dict(agent_networks, parameters_dictionary):
    for a in agent_networks.keys():
        agent_networks[a].load_state_dict(parameters_dictionary[f"agent_{a}_params"])

def main(alg_name, exp_path):
    save_dir = os.path.join('/home/ybang-eai/research/2024/ROMARL/ROMARL/evaluation', alg_name, datetime.now().strftime("%y.%m.%d.%H.%M"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    render_mode = 'silent'
    print("Environment setup ...")
    env = TwoStageROProcessEnvironment(render_mode=render_mode, len_scenario=None, save_dir=save_dir)
    print("Environment setup done.")

    agents = env.agents
    n_agents = len(agents)

    device = 'cuda:0'
    agent_nets = {
        a: RNNAgent(input_shape=env.observation_space(a).shape[0], n_hidden_dim=64, n_actions=env.action_space(a).n).to(device) for a in agents
    }

    # Load experimental information
    episode_log_path    = os.path.join(exp_path, 'episode_log.csv')
    episode_log         = pd.read_csv(episode_log_path)
    train_log_path      = os.path.join(exp_path, 'train_log.csv')
    train_log           = pd.read_csv(train_log_path)
    parameters_path     = os.path.join(exp_path, 'parameters')

    evaluation_interval = 1000
    parameters_dict     = load_parameters_at_intervals_sequential(root_path=parameters_path, interval=evaluation_interval, device=device)

    test_types = ['HinderOneAll', 'HinderOneGivenPeriod']
    for t_type in test_types:
        if not os.path.exists(os.path.join(save_dir, t_type)):
            os.makedirs(os.path.join(save_dir, t_type))
    
    days = 120.0
    dt   = 60.0 * 6
    gamma = 0.99
    reward_weight = [0.5, 0.5]
    production_term = False
    
    # Only evaluate the last episode
    episode = list(parameters_dict.keys())[-1]
    parameters_dict = {episode: parameters_dict[episode]}

    # HinderOneAll
    # Designed to evaluate the robustness in the presence of a single agent's failure. Used in the paper (Decentralized control evaluation + Data generation for SHAP analysis).
    for episode in (parameters_dict.keys()):
        read_state_dict(agent_nets, parameters_dict[episode])
        agents_to_hinder = [None]
        
        for agent_hindered in agents_to_hinder:
            hindered_scenario   = generate_distributed_control_scenario(agents=agents, max_control_timestep=env.max_control_timestep+1, test_type='HinderOneAll', hindered_agent=agent_hindered)
            action_controller   = DistributedActionController(hindered_scenario)
            evaluation_log      = pd.DataFrame(columns=['episode number', 'Reward sum', 'mean concentration', 'mean concentration', 'agent_hindered', 'hinder_type','converged'])
            
            for i, feed_concentration in tqdm(enumerate(np.linspace(300.0, 700.0, 100))):
                evaluation_log_dict = {
                        'episode number': episode,
                        'Reward sum': 0,
                        'mean concentration': 0,
                        'agent_hindered': agent_hindered,
                        'hinder_type': 'HinderOneAll',
                        'converged': None
                }
                for step in range(env.max_control_timestep):
                    # 1. Initialize agent Q (action-value) dictionary.
                    agent_qs = {}

                    # 2. Get the observation from the environment. RESET or STEP depending on the timestep.
                    if step == 0:
                        # 2.0. Reset environment and store observation.
                        initial_action = {"influent_flowrate": 1000.0, "1st_stage_pump": 10.0, "2nd_stage_pump":0.5}
                        previous_observations, _ = env.reset(hard=False, len_scenario=int(24 * days * 60.0 / dt)+1, initial_action=initial_action, feed_concentration=feed_concentration, reward_ws=reward_weight, production_term=production_term)
                        # previous_observations = env.scale_observation(previous_observations)
                        agent_hiddens = {a: agent_nets[a].init_hidden() for a in agents}
                    else:
                        # 2.1. Using the observation from the last timestep, decide which action to take.
                        previous_observations_scaled = env.scale_observation(previous_observations)
                        
                        # 2.1.0. Save the previous observations and hiddens. (For explainability analysis)
                        with open(os.path.join(save_dir, 'HinderOneAll', f'{episode}_{agent_hindered}_{feed_concentration}ppm_previous_observations_scaled_{step}.pkl'), 'wb') as f:
                            pkl.dump(previous_observations_scaled, f)
                        with open(os.path.join(save_dir, 'HinderOneAll', f'{episode}_{agent_hindered}_{feed_concentration}ppm_hiddens_{step}.pkl'), 'wb') as f:
                            pkl.dump(agent_hiddens, f)
                        
                        # 2.1.1. Estimate action-value function with agent networks and observation.
                        for a in agents:
                            q, h = agent_nets[a](torch.from_numpy(previous_observations_scaled[a]['observation']).to(device), agent_hiddens[a])
                            q = mask_and_nothing(action_mask=previous_observations_scaled[a]['action_mask'], Q=q)

                            agent_qs[a] = q
                            agent_hiddens[a] = h

                        # 2.1.1.1 Save the agent Qs. (For explainability analysis)
                        with open(os.path.join(save_dir, 'HinderOneAll', f'{episode}_{agent_hindered}_{feed_concentration}ppm_agent_qs_{step}.pkl'), 'wb') as f:
                            pkl.dump(agent_qs, f)

                        # 2.1.2. Decide actions to take with action policy (epsilon-greedy, Boltzmann, pure greedy).
                        # actions = {
                        #     a: policy(agent_qs[a], previous_observations_scaled[a]['action_mask']) for a in agents
                        # }
                        actions = action_controller.take_action(agent_qs, {a: previous_observations_scaled[a]['action_mask'] for a in agents})

                        # 2.1.3. Take STEP on the environment with the decided actions.
                        observations, rewards, truncated, terminated, _, transition = env.step(actions=actions, terminate_if_diverge=True)

                        if not env.process_valid:
                            break

                        if any(truncated.values()):
                            evaluation_log_dict['converged'] = True
                            evaluation_log_dict['mean concentration'] = np.mean(env.operational_var_1st_stage['C'])

                        previous_observations = deepcopy(observations)

                        done = {a: terminated[a] or truncated[a] for a in agents}

                        if any(done.values()):
                            evaluation_log_dict['Reward sum'] = copy(env.reward_sum_log[-1])
                            if i == 0:
                                evaluation_log = pd.DataFrame([evaluation_log_dict])
                            else:
                                evaluation_log = pd.concat([evaluation_log, pd.DataFrame([evaluation_log_dict])], ignore_index=True)
                            evaluation_log.to_csv(os.path.join(save_dir, 'HinderOneAll', f'{episode}_{agent_hindered}_hindered.csv'))
                            action_controller.reset_action_count()
                            break

    # HinderOneGivenPeriod
    # Designed to evaluate the robustness in the presence of a given period of random hindrance. Not used in the paper.
    # for do_not_hinder in [False]:
    #     for episode in (parameters_dict.keys()):
    #         read_state_dict(agent_nets, parameters_dict[episode])
            
    #         evaluation_log      = pd.DataFrame(columns=['episode number', 'Reward sum', 'mean concentration', 'mean concentration', 'iteration_hindered', 'hinder_type','converged'])

    #         if do_not_hinder:
    #             iteration_num = 1
    #         else:
    #             iteration_num = 1

    #         for iteration_hinder in range(iteration_num):
    #             hindered_scenario   = generate_distributed_control_scenario(agents=agents, max_control_timestep=env.max_control_timestep+1, test_type='HinderOneGivenPeriod')
    #             action_controller   = DistributedActionController(hindered_scenario)
    #             evaluation_log_dict = {
    #                     'episode number': episode,
    #                     'Reward sum': 0,
    #                     'mean concentration': 0,
    #                     'iteration_hindered': iteration_hinder,
    #                     'hinder_type': 'HinderOneGivenPeriod',
    #                     'converged': None
    #             }
    #             for i, feed_concentration in tqdm(enumerate(np.linspace(300.0, 700.0, 20)), desc=f"{iteration_hinder}th evaluation"):
    #                 for step in range(env.max_control_timestep):
    #                     # 1. Initialize agent Q (action-value) dictionary.
    #                     agent_qs = {}

    #                     # 2. Get the observation from the environment. RESET or STEP depending on the timestep.
    #                     if step == 0:
    #                         # 2.0. Reset environment and store observation.
    #                         initial_action = {"influent_flowrate": 1000.0, "1st_stage_pump": 10.0, "2nd_stage_pump":0.5}
    #                         previous_observations, _ = env.reset(hard=False, len_scenario=int(24 * days * 60.0 / dt)+1, initial_action=initial_action, feed_concentration=feed_concentration, reward_ws=reward_weight, production_term=production_term)
    #                         # previous_observations = env.scale_observation(previous_observations)
    #                         agent_hiddens = {a: agent_nets[a].init_hidden() for a in agents}
    #                     else:
    #                         # 2.1. Using the observation from the last timestep, decide which action to take.
    #                         previous_observations_scaled = env.scale_observation(previous_observations)

    #                         # 2.1.1. Estimate action-value function with agent networks and observation.
    #                         for a in agents:
    #                             q, h = agent_nets[a](torch.from_numpy(previous_observations_scaled[a]['observation']).to(device), agent_hiddens[a])
    #                             q = mask_and_nothing(action_mask=previous_observations_scaled[a]['action_mask'], Q=q)

    #                             agent_qs[a] = q
    #                             agent_hiddens[a] = h
                            
    #                         with open(os.path.join(save_dir, 'HinderOneGivenPeriod', f'{episode}_None_{feed_concentration}ppm_previous_observations_scaled_{step}.pkl'), 'wb') as f:
    #                             pkl.dump(previous_observations_scaled, f)
    #                         with open(os.path.join(save_dir, 'HinderOneGivenPeriod', f'{episode}_None_{feed_concentration}ppm_agent_qs_{step}.pkl'), 'wb') as f:
    #                             pkl.dump(agent_qs, f)

    #                         # 2.1.2. Decide actions to take with action policy (epsilon-greedy, Boltzmann, pure greedy).
    #                         # actions = {
    #                         #     a: policy(agent_qs[a], previous_observations_scaled[a]['action_mask']) for a in agents
    #                         # }
    #                         if do_not_hinder:
    #                             actions = {
    #                                 a: greedy_action_policy(agent_qs[a], previous_observations_scaled[a]['action_mask']) for a in agents
    #                             }
    #                         else:
    #                             actions = action_controller.take_action(agent_qs, {a: previous_observations_scaled[a]['action_mask'] for a in agents})

    #                         # 2.1.3. Take STEP on the environment with the decided actions.
    #                         observations, rewards, truncated, terminated, _, transition = env.step(actions=actions, terminate_if_diverge=True)

    #                         if not env.process_valid:
    #                             break

    #                         if any(truncated.values()):
    #                             evaluation_log_dict['converged'] = True
    #                             evaluation_log_dict['mean concentration'] = np.mean(env.operational_var_1st_stage['C'])

    #                         previous_observations = deepcopy(observations)

    #                         done = {a: terminated[a] or truncated[a] for a in agents}

    #                         if any(done.values()):
    #                             evaluation_log_dict['Reward sum']            = copy(env.reward_sum_log[-1])
    #                             if i == 0:
    #                                 evaluation_log = pd.DataFrame([evaluation_log_dict])
    #                             else:
    #                                 evaluation_log = pd.concat([evaluation_log, pd.DataFrame([evaluation_log_dict])], ignore_index=True)
                                
    #                             if do_not_hinder:
    #                                 evaluation_log.to_csv(os.path.join(save_dir, 'HinderOneGivenPeriod', f'{episode}_nothing_hindered.csv'))
    #                             else:
    #                                 evaluation_log.to_csv(os.path.join(save_dir, 'HinderOneGivenPeriod', f'{episode}_{iteration_hinder}th_hindered.csv'))
    #                                 action_controller.reset_action_count()
                                    
    #                             break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set algorithm and experiment path')
    parser.add_argument('--algorithm', type=str, help='Name of the algorithm', required=True)
    parser.add_argument('--exp_path', type=str, help='Path to the experiment directory', required=True)
    args = parser.parse_args()

    main(alg_name=args.algorithm, exp_path=args.exp_path)