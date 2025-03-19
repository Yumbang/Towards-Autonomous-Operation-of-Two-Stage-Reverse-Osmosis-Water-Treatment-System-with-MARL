from TwoStageROProcessEnvironment.env.PressureControlledTwoStageROProcess_simple import TwoStageROProcessEnvironment
from algorithms.mixer.QMIX import QMixer, QMixerRevised, VDN, ReplayBuffer, PrioritizedExperienceReplay, RNNAgent, mask_and_softmax, softmax_and_mask, mask_and_nothing
import numpy as np
import torch
from copy import copy, deepcopy
from utils.epsilon_greedy import epsilon_greedy, EpsilonManager, boltzmann_policy
from utils.descript import save_as_markdown
from matplotlib import pyplot as plt
import pandas as pd
import os
from datetime import datetime
import questionary
import argparse
import shutil


def print_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(f'{name} gradient: {parameter.grad.norm().item()}')
        else:
            print(f'{name} has no gradient')


def check_params(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    return params


def sample_episodes(current_episode, training_frequency, policy='sqrt', include_last=True, unique = True) -> np.ndarray:
    if policy == 'sqrt':
        size = int(np.sqrt(current_episode+1))
    elif policy == 'log':
        size = int(np.log2(current_episode+1))
    else:
        raise AssertionError(f"Invalid sampling policy: '{policy}'. Expected 'sqrt' or 'log'.")

    episodes_to_train = np.random.choice(np.arange(current_episode), size=size, replace=False)

    if include_last:
        if current_episode == 0:
            episodes_to_train = np.append(episodes_to_train, current_episode)
        else:
            episodes_to_train = np.append(episodes_to_train, np.arange(current_episode-training_frequency, current_episode+1))
    
    if unique:
        episodes_to_train = np.unique(episodes_to_train)

    return np.unique(episodes_to_train)


def soft_update(target, source, tau):
    """
    Perform a soft update on the target network parameters.
    For each parameter in the target network, update it towards the corresponding parameter in the source network
    based on the given interpolation factor tau.

    Args:
        target (torch.nn.Module): The target network whose parameters are to be updated.
        source (torch.nn.Module): The source network providing the new parameters.
        tau (float): The interpolation factor used in updating. Typically, is a small number (close to 0).
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def save_model_parameters(mixer, agent_nets, directory):
    """
    Save the parameters of the mixing network and agent networks to the specified directory.

    Args:
        mixer (torch.nn.Module): The mixing network.
        agent_nets (dict): Dictionary of agent networks.
        directory (str): The directory where the model parameters should be saved.
        episode (int): The current episode number for naming the files.
    """
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    torch.save(mixer.state_dict(), os.path.join(directory, f'mixer_params.pt'))

    for agent_id, agent in agent_nets.items():
        torch.save(agent.state_dict(), os.path.join(directory, f'agent_{agent_id}_params.pt'))


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


def plot_TMP(env, step):
    fig, axes = plt.subplots(2,1)
    axes[0].plot(np.array(env.state_var_1st_log[step]['TMP']).squeeze())
    axes[1].plot(np.array(env.state_var_2nd_log[step]['TMP']).squeeze())
    plt.show()

def parse_yaml(file_path):
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def sample_episodes_and_start_points(episode_lengths, len_batch, batch_size):
    sampled_episodes = []
    sampled_start_points = []

    for _ in range(batch_size):
        # 1. Sample episode ID from range [0, episode]
        if len(episode_lengths) == 1:
            episode_id = 0
        else:
            length_enough = False
            while not length_enough:
                episode_id = np.random.randint(0, len(episode_lengths) - 1)
                length_enough = (episode_lengths[episode_id] >= len_batch-1)

        len_episode = episode_lengths[episode_id]

        # 2. Sample starting point from range [0, len_episode - len_batch]
        start_point = np.random.randint(0, len_episode - len_batch)

        # Store the samples
        sampled_episodes.append(episode_id)
        sampled_start_points.append(start_point)

    return sampled_episodes, sampled_start_points

def main(config_file=None, additional_description=None):
    # Format the directory name with the current datetime
    base_dir = r"./figures"
    save_dir_root = os.path.join(base_dir, datetime.now().strftime("%y.%m.%d.%H.%M"))

    # Experiment parameter setting.
    # Configure experiment with Questionary. Depreciated.
    if config_file is None:
        device_number = questionary.select(
            "Select which GPU to use: ",
            choices=['0', '1']
        ).ask()
        description_experiment = questionary.text("Enter description of the experiment:", multiline=True).ask()
        mode = questionary.select("Select operation mode:",
                                    choices=['training', 'inference']).ask()
        if mode == 'inference':
            max_episodes_choices = ['1', '3', '10','50','100']
        else:
            max_episodes_choices = ['100', '150', '200', '250', '300', '350', '400', '500', '1000']

        max_episodes = int(questionary.select("Select maximum episodes to train: ", 
                                        choices=max_episodes_choices).ask())
        
        train_frequency = int(questionary.select("Select training frequency: ", 
                                        choices=['2', '3', '5', '7', '10']).ask())
        
        PER_mode = (questionary.select("Select PER prioritization mode: ", 
                                        choices=["UNIFORM", "RANK-BASED", "PROPORTIONAL"]).ask())
        
        n_epoch = int(questionary.select("Select the number of epoch per an episode: ", 
                                        choices=['5', '10', '15', '20', '30', '50', '75', '100']).ask())
        
        if mode == 'inference':
            epsilon_start = 0.01
            epsilon_decay = 0.995
        else:
            action_policy = questionary.select("Select action policy: "
                                            , choices=['epsilon-greedy', 'Boltzmann'], default='0.995').ask()
            if action_policy == 'epsilon-greedy':
                epsilon_start = float(questionary.select("Select starting value of epsilon: "
                                                , choices=['0.5', '0.75', '0.99'], default='0.99').ask())
                epsilon_decay = float(questionary.select("Select decay of epsilon: "
                                                , choices=['0.95','0.99', '0.995', '0.999'], default='0.995').ask())
            elif action_policy == 'Boltzmann':
                epsilon_start = 10.0
                epsilon_decay = 0.999
            
        if mode == 'training':
            days = int(questionary.select("Choose the length of days to simulate:",
                                    choices=['30', '45', '60', '120']).ask())
        elif mode == 'inference':
            days = int(questionary.select("Choose the length of days to simulate:",
                                    choices=['1', '3', '7', '30',' 45', '60', '120']).ask())
        
        production_term = questionary.select("Choose the total production term (given as penalty at the episode end if True):",
                                    choices=['True', 'False']).ask() == 'True'
        if not production_term:
            reward_weight_SEC = float(questionary.select("Choose the SEC reward weight:",
                                        choices=['0.2', '0.4', '0.5', '0.6', '0.8']).ask())
            reward_weight_eff = float(questionary.select("Choose the effluent reward weight:",
                                        choices=['0.2', '0.4', '0.5', '0.6', '0.8']).ask())
            reward_weight = [reward_weight_SEC, reward_weight_eff]
        else:
            reward_weight = [1.0, 0.0]

        tau = float(questionary.select("Select tau: "
                                        , choices=['0.001','0.005', '0.01', '0.05', '0.1', '0.25', '0.5'], default='0.01').ask())
        
        batch_size = questionary.select("Select batch size: ", choices=['1', '2', '4', '8', 'None']).ask()
        if batch_size == 'None':
            batch_size = None
        else:
            batch_size = int(batch_size)
        
        
    # Configure experiment with yaml config file. Preferred.
    else:
        print("Reading configuration from config file ...")
        config                  = parse_yaml(config_file)
        device_number           = config.get("device_number")
        description_experiment  = config.get("description_experiment")
        mode                    = config.get("mode")
        max_episodes            = config.get("max_episodes")
        train_frequency         = config.get("train_frequency")
        PER_mode                = config.get("PER_mode")
        n_epoch                 = config.get("n_epoch")
        action_policy           = config.get("action_policy")
        epsilon_start           = config.get("epsilon_start")
        epsilon_decay           = config.get("epsilon_decay")
        days                    = config.get("days")
        reward_weight           = config.get("reward_weight")
        production_term         = config.get("total_production")
        tau                     = config.get("tau")
        batch_size              = config.get("batch_size")
        pretrained_parameters   = config.get("pretrained_parameters")

    device_number = int(device_number)

    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")


    print("Using device:", device)

    print("Setting environment ...")
    render_mode = 'silent'
    env = TwoStageROProcessEnvironment(render_mode=render_mode, len_scenario=None, save_dir = save_dir_root)
    print("Done.")
    agents = env.agents
    
    # Select action policy.
    if action_policy == 'epsilon-greedy':
        policy = epsilon_greedy
    elif action_policy == 'Boltzmann':
        policy = boltzmann_policy   # It did not work as intended, but still deserve more trials.

    last_parameter = "Random"

    # Initialize the replay buffer. Although it's PER, if mode is UNIFORM, it works as same as normal replay buffer.
    buffer = PrioritizedExperienceReplay(agents=agents, prioritize=True, device=device, capacity=450000)

    experiment_description_dict = {}

    if additional_description is not None:
        description_experiment += "\n"+additional_description

    experiment_description_dict['Experiment description'] = description_experiment

    begin_train = 10
        
    pretrained_parameters = 'None'

    
    if pretrained_parameters == 'None':
        pretrained_parameters = None
    if pretrained_parameters == 'Select from different directory':
        pretrained_parameters = questionary.path("Choose directory containing pretrained parameters",
                                                 only_directories = True).ask()
    experiment_description_dict['Pretrained parameters'] = pretrained_parameters


    # Algorithm is hard-coded here. I know, it's not a good practice.
    mixer = QMixerRevised(n_state_dim=15, n_agents=len(agents), n_embedding_dim=64, device=device).to(device)
    # mixer = VDN().to(device)

    gamma = 0.99
    experiment_description_dict['gamma'] = gamma
    dt = 60.0 * 6

    agent_nets = {
        a: RNNAgent(input_shape=env.observation_space(a).shape[0], n_hidden_dim=64, n_actions=env.action_space(a).n).to(device) for a in agents
    }

    if pretrained_parameters is not None:
        load_model_parameters(mixer, agent_nets, directory=pretrained_parameters)
        print("Pretrained parameters successfully loaded.")

    print("Check model devices ...")
    for param in mixer.parameters():
        print(f"mixer | {param.device = }")
    for a in agents:
        for param in agent_nets[a].parameters():
            print(f"{a = } | {param.device = }")

    # Combine parameters from all relevant parts of the model
    all_params = list(mixer.parameters()) + [p for a in agents for p in agent_nets[a].parameters()]

    # Create the optimizer
    lr = 5e-4
    # optimizer = torch.optim.Adam(params=all_params, lr=lr)
    # optimizer = torch.optim.RMSprop(params=all_params, lr=lr)
    optimizer = torch.optim.RAdam(params=all_params, lr=lr)
    # optimizer = torch.optim.AdamW(params=all_params, lr=lr)

    experiment_description_dict['Learning rate'] = lr

    # Checking that all model parameters are in the optimizer
    model_params = set([id(p) for p in all_params])
    optimizer_params = set([id(p) for group in optimizer.param_groups for p in group['params']])

    if model_params == optimizer_params:
        print("All parameters are correctly included in the optimizer.")
    else:
        print("Some parameters are missing in the optimizer.")

    
    # Configure target network update strategy (hard or soft, update frequency and tau).
    target_update_frequency = 1
    update_hard = True
    hard_update_frequency = 200
    # tau = 0.001
    # tau = 0.01
    double_q = True
    mask_before_softmax = True

    # Copy the mixing network and agent networks to use as target networks.
    target_mixer = deepcopy(mixer).to(device)
    target_agent_nets = {a: deepcopy(agent_net).to(device) for a, agent_net in agent_nets.items()}

    agent_hiddens = {}
    agent_qs = {}
    target_agent_hiddens = {}
    target_agent_qs = {}

    transition_batch_size = 1

    if action_policy == "Boltzmann":
        epsilon_min = 0.1
    else:
        epsilon_min = 0.05
    
    epsilon_manager = EpsilonManager(epsilon_start=epsilon_start, epsilon_min=epsilon_min, decay_rate=epsilon_decay, decay_type='linear')

    experiment_description_dict.update({
        'Maximum episodes': max_episodes,
        'Training frequency': train_frequency,
        'Target network update frequenct': target_update_frequency,
        'Target network update rate (soft update)': tau,
        'Number of epochs': n_epoch,
        'Double Q-learning': double_q,
        'PER prioritization mode': PER_mode,
        'Action policy': action_policy,
        'Epsilon': f"From {epsilon_start} with decay rate {epsilon_decay}",
    })

    # # Format the directory name with the current datetime
    # save_dir_root = os.path.join(base_dir, datetime.now().strftime("%y.%m.%d.%H.%M"))
    # Ensure the directory exists
    if not os.path.exists(save_dir_root):
        os.makedirs(save_dir_root)

    if config_file is not None:
        shutil.copy(config_file, os.path.join(save_dir_root, "configuration_used.yaml"))

    save_as_markdown(experiment_description_dict, os.path.join(save_dir_root, 'Experiment description'))

    episode_terms_to_log = ['episode number', 'reward sum (without convergence correction)',
                            'times trained', 'epsilon', 'converged']
    train_terms_to_log = ['episode number', 'training number', 'final loss']
    episode_log = pd.DataFrame(columns=episode_terms_to_log)
    train_log = pd.DataFrame(columns=train_terms_to_log)
    reward_sum_log_over_episodes = []
    train_step = 0
    first_train = True
    episode_trained_last = 0
    episodic = True
        
    for episode in range(max_episodes):
        # Update and get the epislon.
        epsilon_manager.update_epsilon()
        epsilon = epsilon_manager.get_epsilon()
        episode_log_dictionary = {
            'episode number': episode,
            'reward sum (without convergence correction)': 0,
            'times trained': 0,
            'epsilon': epsilon,
            'converged': None
        }
        for step in range(env.max_control_timestep):
            # 1. Initialize agent Q (action-value) dictionary.
            agent_qs = {}

            # 2. Get the observation from the environment. RESET or STEP depending on the timestep.
            if step == 0:
                # 2.0. Reset environment and store observation.
                if not render_mode == 'silent':
                    print(f"epsilon = None", end=' | ')
                initial_action = {"influent_flowrate": 1000.0, "1st_stage_pump": 10.0, "2nd_stage_pump":0.5}
                previous_observations, _ = env.reset(hard=False, len_scenario=int(24 * days * 60.0 / dt)+1, initial_action=initial_action, reward_ws=reward_weight, production_term=production_term)
                # previous_observations = env.scale_observation(previous_observations)
                agent_hiddens = {a: agent_nets[a].init_hidden() for a in agents}
            else:
                if not render_mode == 'silent':
                    print(f"epsilon = {epsilon:.2f}", end=' | ')

                # 2.1. Using the observation from the last timestep, decide which action to take.
                previous_observations_scaled = env.scale_observation(previous_observations)

                # 2.1.1. Estimate action-value function with agent networks and observation.
                for a in agents:
                    q, h = agent_nets[a](torch.from_numpy(previous_observations_scaled[a]['observation']).to(device), agent_hiddens[a])

                    if mask_before_softmax:
                        # q = mask_and_softmax(action_mask=previous_observations_scaled[a]['action_mask'], Q=q)
                        q = mask_and_nothing(action_mask=previous_observations_scaled[a]['action_mask'], Q=q)
                    else:
                        q = softmax_and_mask(action_mask=previous_observations_scaled[a]['action_mask'], Q=q)

                    agent_qs[a] = q
                    agent_hiddens[a] = h

                # 2.1.2. Decide actions to take with action policy (epsilon-greedy, Boltzmann, pure greedy).
                actions = {
                    a: policy(agent_qs[a], previous_observations_scaled[a]['action_mask'], epsilon) for a in agents
                }

                # 2.1.3. Take STEP on the environment with the decided actions.
                observations, rewards, truncated, terminated, _, transition = env.step(actions=actions, terminate_if_diverge=True)

                if not env.process_valid:
                    break

                if any(truncated.values()):
                    pass # Debug point

                if not any(terminated.values()) and not any(truncated.values()):
                    # transition = env.transition
                    # transition["hidden"] = copy(agent_hiddens)
                    buffer.push(transition, env)
                    
                else:
                    if any(truncated.values()):
                        buffer.push(transition, env)
                        episode_log_dictionary['converged'] = 'True'

                # assert episode_log_dictionary['converged'] is None

                # observations = env.scale_observation(observations)
                # observations = observations

                previous_observations = deepcopy(observations)

                done = {a: terminated[a] or truncated[a] for a in agents}

                if any(done.values()):
                    reward_sum_log_over_episodes.append(copy(env.reward_sum_log[-1]))
                    episode_log_dictionary['Reward sum']            = copy(env.reward_sum_log[-1]) #  + credit
                    episode_log_dictionary['Parameter used last']   = last_parameter
                    episode_log = pd.concat([episode_log, pd.DataFrame([episode_log_dictionary])], ignore_index=True)
                    episode_log.to_csv(os.path.join(save_dir_root, 'episode_log.csv'))
                    print(f"Episode {episode} done at timestep {step}!")
                    break

            # Update target networks (hard update)
            if update_hard:
                if episodic:
                    if (episode % hard_update_frequency == 0) and step == 0:
                        print("Update target networks (HARD).")
                        target_mixer.load_state_dict(mixer.state_dict())
                        for a in agents:
                            target_agent_nets[a].load_state_dict(agent_nets[a].state_dict())

                        buffer.prioritize(mixer, target_mixer, agent_nets, target_agent_nets, device, env, gamma=gamma, mode=PER_mode, calculate_for_all=True)
                else:
                    if (episode * env.max_control_timestep + step) % hard_update_frequency == 0:
                        print("Update target networks (HARD).")
                        target_mixer.load_state_dict(mixer.state_dict())
                        for a in agents:
                            target_agent_nets[a].load_state_dict(agent_nets[a].state_dict())
            # Update target networks (soft update)
            else:
                soft_update(target_mixer, mixer, tau=tau)
                for a in agents:
                    soft_update(target_agent_nets[a], agent_nets[a], tau=tau)

            non_episodic_condition = ((episode * env.max_control_timestep + step) % train_frequency == 0) & ((episode * env.max_control_timestep + step) > begin_train)
            episodic_condition = ((episode_trained_last == 0) or ((episode - episode_trained_last)/train_frequency >= 1.0)) and (episode > begin_train) and step==0

            if episodic:
                train_condition = episodic_condition
            else:
                train_condition = non_episodic_condition

            if train_condition:
                if episodic:
                    episode_trained_last = episode
                train_log_dictionary = {
                        'target episodes': None,
                        'training number': int(episode/train_frequency),
                        'final loss': None
                }

                num_samples = 8
                len_samples = 60
                episode_lengths = [len(ep) for _, ep in buffer.memory.items()]

                if np.sum(episode_lengths) > buffer.memory_cap:
                    print("Buffer full! Emptying the memory ...")
                    buffer.empty_head()
                    episode_lengths = [len(ep) for _, ep in buffer.memory.items()]

                if episodic:
                    if first_train:
                        calculate_for_all = True
                        first_train = False
                    else:
                        calculate_for_all = False

                    buffer.prioritize(mixer, target_mixer, agent_nets, target_agent_nets, device, env, gamma=gamma, mode=PER_mode, calculate_for_all=calculate_for_all)
                    episodes_to_train = buffer.select_episodes(num_samples=32)
                else:
                    first_train = False
                    episodes_to_train, starting_points = sample_episodes_and_start_points(episode_lengths, len_batch=len_samples, batch_size=num_samples)
                

                for i in range(n_epoch):
                    optimizer.zero_grad()
                    # loss_sum = []
                    batch_loss = torch.tensor(0.0).to(device)

                    if episodic:
                        max_isweight = 0
                        for j, train_episode_id in enumerate(episodes_to_train):
                            buffer.calculate_loss(mixer=mixer, target_mixer=target_mixer, agent_nets=agent_nets, target_agent_nets=target_agent_nets,
                                                episode_id=train_episode_id, device=device, env=env, gamma=gamma, weighted=True)
                            loss, isweight = buffer.sample(train_episode_id)
                            print(f"Episode {train_episode_id:>6} loss: {loss.item():.4f} / isweight: {isweight:.2f}")
                            if torch.isnan(loss):
                                # raise "Nan loss detected. Terminate training ..."
                                loss = None
                                print(f'Nan loss is detected in episode {train_episode_id}. Continuing...')
                                continue
                            batch_loss += loss * isweight
                            if max_isweight < isweight:
                                max_isweight = isweight
                            # loss_sum.append(loss.item())
                        batch_loss /= max_isweight

                    else:
                        for j, train_episode_id in enumerate(episodes_to_train):
                            loss = buffer.calculate_batch_loss(mixer=mixer, target_mixer=target_mixer, agent_nets=agent_nets, target_agent_nets=target_agent_nets,
                                                episode_id=train_episode_id, device=device, env=env, gamma=gamma, starting_index=starting_points[j], batch_size=len_samples)
                            batch_loss += loss
                            if torch.isnan(loss):
                                loss = None
                                print(f'Nan loss is detected in episode {train_episode_id}. Continuing...')
                                continue

                    batch_loss /= len(episodes_to_train)
                    batch_loss.backward()

                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                    optimizer.step()

                    loss_mean = batch_loss.item()
                    print(f"Training step {train_step:<5} | Epoch [{i:<2}] | Weighted mean loss : {loss_mean:.2f}")
                    batch_loss = torch.tensor(0.0).to(device)
                    optimizer.zero_grad()

            
                train_log_dictionary['final loss'] = loss_mean
                train_log = pd.concat([train_log, pd.DataFrame([train_log_dictionary])], ignore_index=True)
                for training_episode in episodes_to_train:
                    episode_log.loc[episode_log['episode number'] == training_episode, 'times trained'] += 1
                train_log.to_csv(os.path.join(save_dir_root, 'train_log.csv'))
                episode_log.to_csv(os.path.join(save_dir_root, 'episode_log.csv'))
                param_dir = os.path.join(save_dir_root, f'parameters/{train_step}')
                if episodic:
                    save_model_parameters(mixer, agent_nets, directory=param_dir)
                elif train_step % (500/train_frequency) == 0:
                    save_model_parameters(mixer, agent_nets, directory=param_dir)
                last_parameter = copy(param_dir)
                train_step += 1

        if episodic:
            buffer.calculate_loss(mixer=mixer, target_mixer=target_mixer, agent_nets=agent_nets, target_agent_nets=target_agent_nets,
                                                episode_id=episode, device=device, env=env, gamma=gamma)
        
        save_dir = os.path.join(save_dir_root, f'episode {episode+1}')

        if episode % 4 == 0:
            env.plot_environment(save_dir=save_dir, plot_state=True)
        else:
            env.plot_environment(save_dir=save_dir, plot_state=False)

        if mode == 'inference':
            continue

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set from a YAML configuration file.')
    parser.add_argument('--yaml_file', type=str, help='Path to the YAML file', required=False)
    parser.add_argument('--description', type=str, help='Additional description about the experiment', required=False)
    args = parser.parse_args()

    main(config_file=args.yaml_file, additional_description=args.description)