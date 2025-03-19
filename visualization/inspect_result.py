import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import questionary


# Function to calculate dynamic moving average and standard deviation
def dynamic_moving_average(values, initial_window):
    moving_averages = np.zeros_like(values)
    moving_std = np.zeros_like(values)
    n = len(values)
    for i in range(n):
        current_window = min(initial_window, i + 1, n - i)  # Adjust the window size near the edges
        start_index = max(0, i - current_window + 1)
        moving_averages[i] = np.mean(values[start_index:i + 1])
        moving_std[i] = np.std(values[start_index:i + 1])

    return moving_averages, moving_std

def moving_average(values, window):
    moving_averages = np.zeros_like(values)
    moving_std = np.zeros_like(values)
    n = len(values) - window
    for i in range(n):
        start_index = i
        moving_averages[i+window] = np.mean(values[start_index:start_index+window])
        moving_std[i+window] = np.std(values[start_index:start_index+window])

    moving_averages[:window] = None
    moving_std[:window] = None

    return moving_averages, moving_std

def plot_moving_avg(data, column_name, window_size):
    values = data[column_name]
    # Calculate the moving average and standard deviation
    # data['Dynamic Moving Average'], data['Dynamic Moving Std'] = dynamic_moving_average(values, window_size)
    data['Dynamic Moving Average'], data['Dynamic Moving Std'] = moving_average(values, window_size)

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column_name], label=None, color='blue', alpha=0.1)
    plt.plot(data.index[window_size:], data['Dynamic Moving Average'], label='Moving Average', color='red')

    plt.title(f'MA Plot - {column_name}')
    plt.xlabel('Training Step')
    plt.ylabel('Target value')
    # plt.ylim([0.0, 100.0])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_moving_avg(data, column_name, window_size, ax=None, average_color = None, datapoints_color = None, data_name = None):
    values = data[column_name]
    # Calculate the moving average and standard deviation
    # data['Dynamic Moving Average'], data['Dynamic Moving Std'] = dynamic_moving_average(values, window_size)
    data['Dynamic Moving Average'], data['Dynamic Moving Std'] = moving_average(values, window_size)

    # If no axes are provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the data
    if average_color is None:
        average_color = 'red'
    if datapoints_color is None:
        datapoints_color = 'blue'
    ax.plot(data.index, data[column_name], label=None, alpha=0.1, color = datapoints_color)
    ax.plot(data.index, data['Dynamic Moving Average'], label=data_name, color = average_color)

    ax.grid(True)

    # Ensure the plot fits well
    plt.tight_layout()

from matplotlib.ticker import MultipleLocator as ML
from matplotlib.ticker import ScalarFormatter as SF
import os

# plt.style.use("default")
style = 'fast'
plt.style.use(style)

# Trained after 24/09/02
exp_paths = {
    'VDN'       :'/home/ybang4/research/ROMARL/VDN result/24.09.15.09.36',
    'QMIX'      :'/home/ybang4/research/ROMARL/config/pressure_tau/figures/24.09.23.11.14',
    'DRQN'      :'/home/ybang4/research/ROMARL/config/central_DDQN/figures/24.09.27.15.07',
}

episode_paths = {exps: os.path.join(exp_paths[exps], 'episode_log.csv') for exps in exp_paths.keys()}
training_paths = {exps: os.path.join(exp_paths[exps], 'train_log.csv') for exps in exp_paths.keys()}
reward_name = 'Reward sum'
loss_name = 'final loss'
episodes_data = {exps: pd.read_csv(episode_paths[exps]) for exps in exp_paths.keys()}
training_data = {exps: pd.read_csv(training_paths[exps]) for exps in exp_paths.keys()}

fig_exps, ax_exps = plt.subplots(figsize = (12,6))

window_size = 1000
window = 50000

exp_color_map = {
    'QMIX': 'C2',
    'VDN': 'C1',
    'DRQN': 'C4',
}

result_path = '/home/ybang4/research/ROMARL/performance plots'

if not os.path.exists(f"{result_path}/{style}"):
    os.makedirs(f"{result_path}/{style}")

for exps in exp_paths.keys():
    # total_production_credit = np.array([3.0 if row["Total Production"] > 700.0 else 0.0 for _,row in performance_data[exps].iterrows()])
    # total_production_credit = total_production_credit[:window]
    if window is not None and len(episodes_data[exps]) > window:
        episodes_data[exps] = episodes_data[exps][:window]
    # episodes_data[exps][reward_name] += total_production_credit
    plot_moving_avg(episodes_data[exps], reward_name, window_size, ax_exps, exp_color_map[exps], exp_color_map[exps], exps)

ax_exps.yaxis.set_major_locator(ML(20.0))
ax_exps.yaxis.set_minor_locator(ML(10.0))
ax_exps.yaxis.set_minor_formatter(SF())
ax_exps.tick_params(axis='y', which='minor', labelsize=8)
ax_exps.yaxis.grid(True, which='minor', alpha=0.25)

ax_exps.xaxis.set_major_locator(ML(0.5e4))
ax_exps.xaxis.set_minor_locator(ML(0.25e4))
ax_exps.xaxis.set_minor_formatter(SF())
ax_exps.xaxis.grid(True, which='minor', alpha=0.1)
ax_exps.tick_params(axis='x', which='both', labelsize='small', rotation=45)

ax_exps.set_xlim([0, window])
ax_exps.set_ylim([-50.0, 5])
ax_exps.set_title(None)
ax_exps.set_xlabel("Episodes")
ax_exps.set_ylabel("Reward Sum")
ax_exps.legend()

plt.tight_layout()
plt.savefig(f"{result_path}/{style}/Reward_Sum_Result_{window}.png", dpi=300)
plt.show()


# Plot loss

fig_loss, ax_loss = plt.subplots(figsize = (12,4.5))


for exps in exp_paths.keys():
    if window is not None and len(episodes_data[exps]) > window:
        episodes_data[exps] = episodes_data[exps][:window]
    plot_moving_avg(training_data[exps], loss_name, window_size, ax_loss, exp_color_map[exps], exp_color_map[exps], exps)

ax_loss.yaxis.set_major_locator(ML(0.2))
ax_loss.yaxis.set_minor_locator(ML(0.1))
ax_loss.yaxis.set_minor_formatter(SF())
ax_loss.tick_params(axis='y', which='minor', labelsize=8)
ax_loss.yaxis.grid(True, which='minor', alpha=0.25)

ax_loss.xaxis.set_major_locator(ML(0.5e4))
ax_loss.xaxis.set_minor_locator(ML(0.25e4))
ax_loss.xaxis.set_minor_formatter(SF())
ax_loss.xaxis.grid(True, which='minor', alpha=0.1)
ax_loss.tick_params(axis='x', which='both', labelsize='small', rotation=45)

ax_loss.set_xlim([0, window])
ax_loss.set_ylim([0.0, 1.0])
ax_loss.set_title(None)
ax_loss.set_xlabel("Episodes")
ax_loss.set_ylabel("Loss")
ax_loss.legend()

plt.tight_layout()
plt.savefig(f"{result_path}/{style}/Loss_Result_Highscale_{window}.png", dpi=300)

ax_loss.set_ylim([0.0, 0.01])
ax_loss.yaxis.set_major_locator(ML(0.002))
ax_loss.yaxis.set_minor_locator(ML(0.001))
plt.tight_layout()
plt.savefig(f"{result_path}/{style}/Loss_Result_Lowscale_{window}.png", dpi=300)
plt.show()

# Performance plotting

performance_data = {}

performance_path = {
   'VDN': f'{result_path}/VDN_performance.csv',
   'QMIX': f'{result_path}/QMIX_performance.csv',
   'DRQN': f'{result_path}/DDQN_performance.csv'
}

for exp_name, file_root_path in exp_paths.items():
    if exp_name in performance_path.keys():
        performance_data[exp_name] = pd.read_csv(performance_path[exp_name])
        continue
    tmp_episodes_dirs = os.listdir(file_root_path)
    episode_dirs = [epdir for epdir in tmp_episodes_dirs if epdir.startswith('episode ')]
    episode_ids = [int(epdir[7:]) for epdir in tmp_episodes_dirs if epdir.startswith('episode ')]

    episode_df = pd.DataFrame({
        'ID': episode_ids,
        'Directory': episode_dirs,
    })
    episode_df = episode_df.sort_values(by='ID')
    episode_df = episode_df[:-10]
    episode_df.reset_index(drop=True, inplace=True)
    episode_df['SEC'] = np.zeros_like(episode_df['Directory'])
    episode_df['Total Production'] = np.zeros_like(episode_df['Directory'])
    episode_df['Total Energy'] = np.zeros_like(episode_df['Directory'])
    episode_df['Temperature'] = np.zeros_like(episode_df['Directory'])
    episode_df['Concentration'] = np.zeros_like(episode_df['Directory'])
    ms_log = []
    tp_log = []
    te_log = []
    t_log = []
    c_log = []
    for idx, episode in episode_df.iterrows():
        path = os.path.join(file_root_path, episode['Directory'])
        SEC_log = pd.read_csv(os.path.join(path, 'ro_sec_total_df.csv'))
        op_var_1st_log = pd.read_csv(os.path.join(path, 'operational_var_1st_df.csv'))
        total_production_log = pd.read_csv(os.path.join(path, 'permeate_var_total_df.csv'))
        t_log.append(np.mean(total_production_log['T']))
        c_log.append(np.mean(op_var_1st_log['C']))
        criteria_sec = 0.56 - 0.00927 * total_production_log['T']
        criteria_mean_sec = np.mean(criteria_sec)
        mean_sec = np.mean(SEC_log.iloc[:,1])
        total_production = np.mean(total_production_log['Q'])
        total_energy = mean_sec * total_production
        ms_log.append(mean_sec)
        tp_log.append(total_production)
        te_log.append(total_energy)
        print(f"{idx} done.")
    episode_df['SEC'] = ms_log
    episode_df['Total Production'] = tp_log
    episode_df['Total Energy'] = te_log
    episode_df['Temperature'] = t_log
    episode_df['Concentration'] = c_log
    episode_df.to_csv(f"{result_path}/{exp_name}_performance.csv")
    performance_data[exp_name] = episode_df


performance_names = ['Total Production', 'SEC', 'Total Energy']

for performance_name in performance_names:
    fig_performance, ax_performance = plt.subplots(figsize = (12,4.5))

    for exps in exp_paths.keys():
        if window is not None and len(performance_data[exps]) > window:
            performance_data[exps] = performance_data[exps][:window]
        plot_moving_avg(performance_data[exps], performance_name, window_size, ax_performance, exp_color_map[exps], exp_color_map[exps], exps)

    # ax_performance.yaxis.set_major_locator(ML(0.2))
    # ax_performance.yaxis.set_minor_locator(ML(0.1))
    ax_performance.yaxis.set_minor_formatter(SF())
    ax_performance.tick_params(axis='y', which='minor', labelsize=8)
    ax_performance.yaxis.grid(True, which='minor', alpha=0.25)

    ax_performance.xaxis.set_major_locator(ML(0.5e4))
    ax_performance.xaxis.set_minor_locator(ML(0.25e4))
    ax_performance.xaxis.set_minor_formatter(SF())
    ax_performance.xaxis.grid(True, which='minor', alpha=0.1)
    ax_performance.tick_params(axis='x', which='both', labelsize='small', rotation=45)

    ax_performance.set_xlim([0, window])
    # ax_performance.set_ylim([0.0, 1.0])
    ax_performance.set_title(None)
    ax_performance.set_xlabel("Episodes")
    ax_performance.set_ylabel(performance_name)
    ax_performance.legend()

    plt.tight_layout()
    plt.savefig(f"{result_path}/{style}/{performance_name}_Result_{window}.png", dpi=300)
    plt.show()
