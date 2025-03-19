from __future__ import annotations
import pandas as pd
import numpy as np
import os
import gymnasium
import pickle
import juliacall
from matplotlib import pyplot as plt
from pettingzoo.utils.env import AgentID, ObsType, ActionType
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete, Dict
from datetime import datetime
from copy import copy, deepcopy
from juliacall import convert as jlconvert



class TwoStageROProcessEnvironment(ParallelEnv):
    metadata = {
        "name": "Pressure_2stage_ro_process_environment_v1.1"
    }

    # Initialize environment.
    def __init__(self, save_dir, len_scenario=None, render_mode='text'):

        # Setup Julia.
        # Default dt value is 1.0 (1 minute). If one needs to change dt, go to "ro_element.jl" file and modify "dt" variale.
        self.julia_file_path = os.path.abspath("/home/ybang-eai/research/2024/ROMARL/ROMARL/TwoStageROProcessEnvironment/julia modules")
        self.jl = juliacall.Main
        self.save_dir = save_dir
        julia_path = os.path.join(self.julia_file_path, "pressure_controlled_ro_simple.jl")
        self.jl.seval(f"include(raw\"{julia_path}\")")
        self.pressure_controlled_ro = self.jl.PressureControlledRO.pressure_controlled_2stage_ro_simple

        self.len_scenario = len_scenario
        self.render_mode = render_mode


        # Initialize seed state variables and operational variables.
        self.dt = 60.0 * 6
        self.days = 30.0  # Model for 1 month
        self.control_dt = 60.0 * 6 # Control every 6 hours.
        self.control_interval = int(self.control_dt / self.dt)
        self.max_timestep = int(self.days * 24 * 60 / self.dt)
        self.max_control_timestep = int(self.max_timestep / self.control_interval)
        self.episode_id = 0
        self.timestep = None
        self.control_timestep = None

        # Initialize seed state variables.
        self.state_var_1st_stage = {
            'timestep': None
        }
        self.state_var_2nd_stage = {
            'timestep': None
        }


        # Initialize necessary constants for modeling and SEC calculation.
        self.operational_var_1st_stage = None
        self.operational_var_2nd_stage = None
        self.ro_1st_pvs = 84.0
        self.ro_2nd_pvs = 48.0
        self.ro_1st_hpp_eff = 0.8
        self.ro_2nd_boosting_eff = 0.8

        # Define permeate and brine variables
        self.permeate_var_1st_stage = None
        self.brine_var_1st_stage = None

        self.permeate_var_2nd_stage = None
        self.brine_var_2nd_stage = None

        self.permeate_total = None
        self.brine_total = None

        self.HPP_last = None
        self.IBP_last = None

        # Define observed SEC variables
        self.ro_1st_SEC = None
        self.ro_2nd_SEC = None
        self.ro_total_SEC = None

        # Define observed recovery rate variables
        self.ro_1st_recovery = None
        self.ro_2nd_recovery = None
        self.ro_total_recovery = None

        # Define observed rejection rate variables
        self.ro_1st_rejection = None
        self.ro_2nd_rejection = None
        self.ro_total_rejection = None

        # Define action variables
        self.influent_flowrate = None
        self.ro_1st_pressure = None
        self.ro_2nd_pressure = None
        self.ro_cip = None

        self.converged = None
        self.process_valid = None

        # Define previous observation and state variables
        self.previous_observation = None
        self.previous_state = None

        self.start_point = None

        self.transition = None

        self.w_SEC = None
        self.w_eff = None
        self.total_production_term = None

        # Initialize lists for logging.
        # Note that state_var_N'th_log's are only storing 'snapshot' of state variables. They are large as 🌸🌸🌸🌸.
        self.operational_var_1st_log = []
        self.permeate_var_1st_log = []
        self.brine_var_1st_log = []
        self.state_var_1st_log = []

        self.influent_flowrate_log = []
        self.ro_1st_pressure_log = []
        self.ro_2nd_pressure_log = []

        self.operational_var_2nd_log = []
        self.permeate_var_2nd_log = []
        self.brine_var_2nd_log = []
        self.state_var_2nd_log = []

        self.permeate_total_log = []
        self.brine_total_log = []

        self.ro_1st_SEC_log = []
        self.ro_2nd_SEC_log = []
        self.ro_total_SEC_log = []

        self.ro_1st_recovery_log = []
        self.ro_2nd_recovery_log = []
        self.ro_total_recovery_log = []

        self.ro_1st_rejection_log = []
        self.ro_2nd_rejection_log = []
        self.ro_total_rejection_log = []

        self.action_log = []
        self.reward_total_log = []
        self.reward_sum_log = []

        self.observation_log = []
        self.state_log = []

        # Define agents. 🔫😎
        self.agents = [
            "influent_flowrate",
            "1st_stage_pump",
            "2nd_stage_pump",
        ]

        self.possible_agents = [
            "influent_flowrate",
            "1st_stage_pump",
            "2nd_stage_pump",
        ]

        self.reward_total = None
        self.reward_sum = None

        # Define feed scenario.
        self.total_simulation_time = int(365 * 24 * 60 / self.dt)
        self.water_temperature = np.ones(self.total_simulation_time) * 20.0
        self.concentration = 500.0 * np.ones(self.total_simulation_time)
        self.influent_pressure = 1e-5 * np.ones(self.total_simulation_time)
        self.C_CF = 10e-3

        noise1 = np.random.normal(loc=0, scale=1, size=self.total_simulation_time)
        noise2 = np.random.normal(loc=0, scale=1, size=self.total_simulation_time)

        self.feed_scenario_total = np.vstack([
            self.water_temperature + noise1 * 0.5,
            self.concentration + noise2 * 25.0,
            self.influent_pressure,
        ]).transpose()
        self.feed_scenario = None

        # Define observation spaces for each agent.
        self.observation_spaces = {
            "influent_flowrate": Box(
                low  = np.array([700.0  / self.ro_1st_pvs,  300.0, 100.0  / self.ro_2nd_pvs,  100.0, 100.0, 0.0]),
                high = np.array([1400.0 / self.ro_1st_pvs, 900.0, 1000.0 / self.ro_2nd_pvs, 300.0, 300.0, 40.0]),
                dtype=np.float32
            ),
            "1st_stage_pump": Box(
                low  = np.array([0.0,  200.0,  700.0  / self.ro_1st_pvs,  500.0,  100.0,  0.0,   100.0,  0.0,  0.0,  0.0]),
                high = np.array([40.0, 1000.0, 1400.0 / self.ro_1st_pvs, 2500.0, 1000.0, 100.0, 1000.0, 25.0, 25.0, 1.0]),
                dtype=np.float32
            ),
            "2nd_stage_pump": Box(
                low  = np.array([0.0,  500.0,  100.0  / self.ro_2nd_pvs,  0.0,  1000.0, 50.0,  50.0,  50.0,  0.0,  0.0,  0.0]),
                high = np.array([40.0, 2500.0, 1000.0 / self.ro_2nd_pvs, 25.0, 5000.0, 500.0, 250.0, 500.0, 25.0, 25.0, 1.0]),
                dtype=np.float32
            ),
        }

        # Define action spaces for each agent.
        self.action_spaces = {
            # +--------------+-----------------------------------+
            # | Action Value |        Pump Agent's Action        |
            # +--------------+-----------------------------------+
            # |            0 | Decrease pressure by 0.25 bar     |
            # |            1 | Decrease pressure by 0.125 bar    |
            # |            2 | Remain pressure as same as before |
            # |            3 | Increase pressure by 0.125 bar    |
            # |            4 | Increase pressure by 0.25 bar     |
            # +--------------+-----------------------------------+
            # | Action Value |      Influent Agent's Action      |
            # +--------------+-----------------------------------+
            # |            0 | Decrease flowrate by    5 m3/hr   |
            # |            1 | Decrease flowrate by  2.5 m3/hr   |
            # |            2 | Remain flowrate as same as before |
            # |            3 | Increase flowrate by  2.5 m3/hr   |
            # |            4 | Increase flowrate by    5 m3/hr   |
            # +--------------+-----------------------------------+
            "influent_flowrate": Discrete(5, start=0),
            "1st_stage_pump": Discrete(5, start=0),
            "2nd_stage_pump": Discrete(5, start=0),
        }


        # atexit.register(self.cleanup)

    def reset(self, hard=False, len_scenario=None, initial_action: dict | None = None, seed: int | None = None, feed_concentration: float = None, reward_ws = [1.0, 1.0], production_term=True) -> \
    tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """
         Reset environment before running.
         Prepare to run and return the dictionary first observation and info of agents.
         Resets attributes logging lists to clean up the last trial.
        """
        if hard:
            self.episode_id = 0
        else:
            self.episode_id += 1

        self.agents = copy(self.possible_agents)
        self.timestep = 1
        self.control_timestep = 1

        # Set initial state_var values and PID controllers for RO modules.
        self.state_var_1st_stage = jlconvert(T=self.jl.Dict, x={
            "timestep": 1.0
        })
        self.state_var_2nd_stage = jlconvert(T=self.jl.Dict, x={
            "timestep": 1.0
        })

        self.reward_total = None
        self.reward_sum = 0
        self.converged = True

        self.HPP_last = None
        self.IBP_last = None

        # Reset SEC variables
        self.ro_1st_SEC = 0.0
        self.ro_2nd_SEC = 0.0
        self.ro_total_SEC = 0.0

        # Reset recovery rate variables
        self.ro_1st_recovery = 0.0
        self.ro_2nd_recovery = 0.0
        self.ro_total_recovery = 0.0

        # Reset rejection rate variables
        self.ro_1st_rejection = 0.0
        self.ro_2nd_rejection = 0.0
        self.ro_total_rejection = 0.0

        assert (self.len_scenario is None or len_scenario is None), "Length of scenario is not implemented."

        if len_scenario is None:
            len_scenario = self.len_scenario

        self.feed_scenario, self.max_timestep = self.sample_scenario(len_scenario=len_scenario, concentration=feed_concentration, range = 200.0, noise=True)

        self.transition = None

        self.set_reward_w(reward_Ws=reward_ws)
        self.total_production_term = production_term

        # Initialize lists for logging.
        self.operational_var_1st_log = []
        self.permeate_var_1st_log = []
        self.brine_var_1st_log = []
        self.state_var_1st_log = []

        self.influent_flowrate_log = []
        self.ro_1st_pressure_log = []
        self.ro_2nd_pressure_log = []

        self.operational_var_2nd_log = []
        self.permeate_var_2nd_log = []
        self.brine_var_2nd_log = []
        self.state_var_2nd_log = []

        self.permeate_total_log = []
        self.brine_total_log = []

        self.ro_1st_SEC_log = []
        self.ro_2nd_SEC_log = []
        self.ro_total_SEC_log = []

        self.ro_1st_recovery_log = []
        self.ro_2nd_recovery_log = []
        self.ro_total_recovery_log = []

        self.ro_1st_rejection_log = []
        self.ro_2nd_rejection_log = []
        self.ro_total_rejection_log = []

        self.action_log = []
        self.reward_total_log = []
        self.reward_sum_log = []

        self.observation_log = []
        self.state_log = []

        if initial_action is not None:
            # Receive initial operational condition and configure attributes.
            self.influent_flowrate = initial_action['influent_flowrate']
            self.ro_1st_pressure = initial_action['1st_stage_pump']
            self.ro_2nd_pressure = initial_action['2nd_stage_pump']
        else:
            # Configure initial operational point to get the first observation.
            # Will be implemented to be provided seed operational variables as scenario in v1.
            self.influent_flowrate = 1000.0
            self.ro_1st_pressure = 0.60
            self.ro_2nd_pressure = 0.60

        self.influent_flowrate_log.append(copy(self.influent_flowrate))
        self.ro_1st_pressure_log.append(copy(self.ro_1st_pressure))
        self.ro_2nd_pressure_log.append(copy(self.ro_2nd_pressure))

        # Process numerical modeling for {self.control_interval} length.
        self.process_valid = self._process_modeling(feed_scenario=self.feed_scenario, starting_index=0, modeling_length=self.control_interval)

        action_masks = self._generate_action_mask()
        influent_flowrate_action_mask = action_masks['influent_flowrate']
        ro_1st_action_mask = action_masks['1st_stage_pump']
        ro_2nd_action_mask = action_masks['2nd_stage_pump']

        # Generate observations.
        observations = {
            "influent_flowrate": {'observation': np.array([self.operational_var_1st_stage['Q'], self.permeate_var_1st_stage['Q'], self.operational_var_2nd_stage['Q'],
                                                        self.permeate_var_2nd_stage['Q'], self.brine_var_2nd_stage['Q'], self.operational_var_1st_stage['T']],
                                                        dtype=np.float32),
                                'action_mask': influent_flowrate_action_mask},
            "1st_stage_pump": {'observation': np.array([self.operational_var_1st_stage['T'], self.operational_var_1st_stage['C'], self.operational_var_1st_stage['Q'],
                                                        self.brine_var_1st_stage['C'], self.brine_var_1st_stage['Q'], self.permeate_var_1st_stage['C'], self.permeate_var_1st_stage['Q'], 
                                                        self.brine_var_1st_stage['P'], self.operational_var_1st_stage['P'] - np.mean(self.feed_scenario[:self.control_interval, 2]),
                                                        self.ro_1st_recovery],
                                                       dtype=np.float32),
                               'action_mask': ro_1st_action_mask
                               },
            "2nd_stage_pump": {'observation': np.array([self.operational_var_2nd_stage['T'], self.operational_var_2nd_stage['C'], self.operational_var_2nd_stage['Q'], self.brine_var_1st_stage['P'],
                                                        self.brine_var_2nd_stage['C'], self.brine_var_2nd_stage['Q'], self.permeate_var_2nd_stage['C'], self.permeate_var_2nd_stage['Q'],
                                                        self.brine_var_2nd_stage['P'], self.operational_var_2nd_stage['P'] - self.brine_var_1st_stage['P'],
                                                        self.ro_1st_recovery], dtype=np.float32),
                               'action_mask': ro_2nd_action_mask},
        }

        infos = {
            a: {} for a in self.agents
        }

        self.timestep += self.control_interval
        self.control_timestep += 1

        self.render(mode=self.render_mode)


        self.previous_observation = copy(observations)
        self.previous_state = copy(self.state(normalize=True))

        self.observation_log.append(copy(self.previous_observation))

        return observations, infos

    def step(self, actions: dict[AgentID, ActionType], terminate_if_diverge=True) -> tuple[
        dict[AgentID, ObsType], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict], dict]:
        """
        Apply actions to action values attributes.
        The method must return dictionaries of observation, reward, truncation and termination for each agent.
        """

        # Log actions.
        self.action_log.append(copy(actions))

        self.influent_flowrate += (actions['influent_flowrate'] - 2.0) * 10.0 / 2
        self.ro_1st_pressure   += (actions['1st_stage_pump'] - 2.0) * 0.25 / 4
        self.ro_2nd_pressure   += (actions['2nd_stage_pump'] - 2.0) * 0.25 / 4

        # Model process and update attributes.
        self.process_valid = self._process_modeling(feed_scenario=self.feed_scenario, starting_index=self.timestep, modeling_length=self.control_interval)

        self.influent_flowrate_log.append(copy(self.influent_flowrate))
        self.ro_1st_pressure_log.append(copy(self.ro_1st_pressure))
        self.ro_2nd_pressure_log.append(copy(self.ro_2nd_pressure))

        action_masks = self._generate_action_mask()
        influent_flowrate_action_mask = action_masks['influent_flowrate']
        ro_1st_action_mask = action_masks['1st_stage_pump']
        ro_2nd_action_mask = action_masks['2nd_stage_pump']

        # Generate observations.
        observations = {
            "influent_flowrate": {'observation': np.array([self.operational_var_1st_stage['Q'], self.permeate_var_1st_stage['Q'], self.operational_var_2nd_stage['Q'],
                                                        self.permeate_var_2nd_stage['Q'], self.brine_var_2nd_stage['Q'], self.operational_var_1st_stage['T']],
                                                        dtype=np.float32),
                                'action_mask': influent_flowrate_action_mask},
            "1st_stage_pump": {'observation': np.array([self.operational_var_1st_stage['T'], self.operational_var_1st_stage['C'], self.operational_var_1st_stage['Q'],
                                                        self.brine_var_1st_stage['C'], self.brine_var_1st_stage['Q'], self.permeate_var_1st_stage['C'], self.permeate_var_1st_stage['Q'], 
                                                        self.brine_var_1st_stage['P'], self.operational_var_1st_stage['P'] - np.mean(self.feed_scenario[:self.control_interval, 2]),
                                                        self.ro_1st_recovery],
                                                       dtype=np.float32),
                               'action_mask': ro_1st_action_mask
                               },
            "2nd_stage_pump": {'observation': np.array([self.operational_var_2nd_stage['T'], self.operational_var_2nd_stage['C'], self.operational_var_2nd_stage['Q'], self.brine_var_1st_stage['P'],
                                                        self.brine_var_2nd_stage['C'], self.brine_var_2nd_stage['Q'], self.permeate_var_2nd_stage['C'], self.permeate_var_2nd_stage['Q'],
                                                        self.brine_var_2nd_stage['P'], self.operational_var_2nd_stage['P'] - self.brine_var_1st_stage['P'],
                                                        self.ro_1st_recovery], dtype=np.float32),
                               'action_mask': ro_2nd_action_mask},
        }

        # if not self.converged:
        #     print(f'Failed to converge in timestep {self.timestep}')

        if terminate_if_diverge:
            termination_total = (not self.converged)
        else:
            termination_total = False

        terminated = {a: termination_total for a in self.agents}

        # Evaluate truncation based on timestep
        truncation_total = (self.control_timestep >= self.max_control_timestep)
        truncated = {a: truncation_total for a in self.agents}

        done = {a: terminated[a] or truncated[a] for a in self.agents}

        # Calculate total reward as weighted sum of SEC and permeate concentration.
        self.reward_total = self._calculate_reward(terminate_if_diverge=terminate_if_diverge)

        continuous_credit = False
        criteria_flowrate = 787.5

        if truncation_total & (not termination_total):
            if self.total_production_term:
                average_total_product = np.mean([self.permeate_total_log[step]['Q'] for step in range(len(self.permeate_total_log))])

                if continuous_credit:
                    total_product_credit = np.minimum(criteria_flowrate/50.0, average_total_product/50.0)
                else:
                    total_product_credit = 30.0 if average_total_product > criteria_flowrate else -90.0
                
                self.reward_total += total_product_credit
                print(f"Gave extra credit of {total_product_credit:.1f}")

        # Log reward and calculate reward sum.
        self.reward_total_log.append(copy(self.reward_total))
        self.reward_sum = np.sum(np.array(self.reward_total_log))
        self.reward_sum_log.append(copy(self.reward_sum))

        rewards = {
            # As it is impossible to calculate reward for each agent, just return total reward.
            a: self.reward_total for a in self.agents
        }

        infos = {
            a: {} for a in self.agents
        }

        # Update timestep.
        self.timestep += self.control_interval
        self.control_timestep += 1
        
        self.render(mode=self.render_mode)

        # Update the transition from previous timestep to current timestep.
        transition = self._get_transition(actions=actions, observations=observations, rewards=rewards, done=done)

        self.previous_observation = deepcopy(observations)
        self.previous_state = deepcopy(self.state(normalize=True))

        self.observation_log.append(deepcopy(self.previous_observation))

        return observations, rewards, truncated, terminated, infos, deepcopy(transition)

    def render(self, mode=None):
        if mode == 'human':
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
            axes[0].plot(np.array(self.state_var_1st_stage['TMP']).squeeze())
            axes[0].set_title(f"1st Stage RO TMP at timestep {self.timestep}")
            axes[1].plot(np.array(self.state_var_2nd_stage['TMP']).squeeze())
            axes[1].set_title(f"2nd Stage RO TMP at timestep {self.timestep}")
            plt.tight_layout()
            plt.show()
        elif mode == 'text':
            simple = True
            if not simple:
                if self.reward_total is None:
                    print(
                        f"Timestep [{self.timestep: <5}] | Flowrate: {self.influent_flowrate:.1f} | 1st Pressure: {self.ro_1st_pressure:.3f} | 2nd Pressure: {self.ro_2nd_pressure:.3f} | Reward: None")
                else:
                    print(
                        f"Timestep [{self.timestep: <5}] | Flowrate: {self.influent_flowrate:.1f} | 1st Pressure: {self.ro_1st_pressure:.3f} | 2nd Pressure: {self.ro_2nd_pressure:.3f} | Reward: {self.reward_total:.2f}")
            else:
                if self.reward_total is None:
                    print(
                        f"T [{self.timestep: <5}] | Q: {self.influent_flowrate:.1f} | P1: {self.ro_1st_pressure:.2f} | P2: {self.ro_2nd_pressure:.2f} | r: None")
                else:
                    print(
                        f"T [{self.timestep: <5}] | Q: {self.influent_flowrate:.1f} | P1: {self.ro_1st_pressure:.2f} | P2: {self.ro_2nd_pressure:.2f} | r: {self.reward_total:.2f}")
        elif mode == 'tensorboard':
            pass
        elif mode is None:
            pass
        elif mode == 'silent':
            pass

    def state(self, normalize=True):
        """
         Returns state variable to describe the environment.
        state = [Influent mean (Q, C, T), 1st / 2nd recovery setpoint, HPP P, IBP P, Concentrate mean (Q, P, C) of 1st and 2nd RO, Permeate mean (Q, C)  of 1st and 2nd RO]
        :return: np.array(state)
        """
        def normalize_P(P_in, P_min, P_max):
            return (P_in - P_min) / (P_max - P_min)

        P_min = 0.0
        P_max = 25.0
        Q_max = 1394.0
        T_max = 50.0
        C_feed_max  = 1000.0
        C_brine_1st_max = 2000.0
        C_brine_2nd_max = 20000.0
        C_perm_max  = 100.0

        if normalize:
            state = [
                self.influent_flowrate / Q_max, self.feed_scenario[self.timestep - 2, 1] / C_feed_max,
                self.feed_scenario[self.timestep - 2, 0] / T_max,
                normalize_P(self.operational_var_1st_stage["P"], P_min, P_max), normalize_P(self.operational_var_2nd_stage["P"] - self.brine_var_1st_stage["P"], P_min, P_max),
                self.brine_var_1st_stage['Q'] / Q_max, normalize_P(self.brine_var_1st_stage['P'], P_min, P_max),
                self.brine_var_1st_stage['C'] / C_brine_1st_max,
                self.brine_var_2nd_stage['Q'] / Q_max, normalize_P(self.brine_var_2nd_stage['P'], P_min, P_max),
                self.brine_var_2nd_stage['C'] / C_brine_2nd_max,
                self.permeate_var_1st_stage['Q'] / Q_max, self.permeate_var_1st_stage['C'] / C_perm_max,
                self.permeate_var_2nd_stage['Q'] / Q_max, self.permeate_var_2nd_stage['C'] / C_perm_max
            ]
        else:
            # state = [
            #     self.influent_flowrate, self.feed_scenario[self.timestep - 1, 0],
            #     self.feed_scenario[self.timestep - 1, 1],
            #     self.ro_1st_hpp_pressure, self.ro_2nd_boosting_pressure,
            #     self.brine_var_1st_stage['Q'], self.brine_var_1st_stage['P'], self.brine_var_1st_stage['C'],
            #     self.brine_var_2nd_stage['Q'], self.brine_var_2nd_stage['P'], self.brine_var_2nd_stage['C'],
            #     self.permeate_var_1st_stage['Q'], self.permeate_var_1st_stage['C'],
            #     self.permeate_var_2nd_stage['Q'], self.permeate_var_2nd_stage['C']
            # ]
            assert "Do Normalization."
        return np.array(state, dtype=np.float32)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def plot_environment(self, save_dir, plot_state = False):
        episode = self.episode_id
        if save_dir is None:
            save_dir = os.curdir
            print(f"No directory is given. Saving figures in current directory...")
            save_dir = os.path.join(save_dir, f"{datetime.now():%y.%m.%d.%H.%M}", f'episode {episode}')

        os.makedirs(os.path.join(save_dir), exist_ok=True)

        action_df = pd.DataFrame(self.action_log)

        operational_var_1st_df = pd.DataFrame(self.operational_var_1st_log)
        operational_var_2nd_df = pd.DataFrame(self.operational_var_2nd_log)
        brine_var_1st_df = pd.DataFrame(self.brine_var_1st_log)
        brine_var_2nd_df = pd.DataFrame(self.brine_var_2nd_log)

        permeate_var_1st_df = pd.DataFrame(self.permeate_var_1st_log)
        permeate_var_2nd_df = pd.DataFrame(self.permeate_var_2nd_log)
        permeate_var_total_df = pd.DataFrame(self.permeate_total_log)
        ro_sec_1st_df = pd.DataFrame(self.ro_1st_SEC_log)
        ro_sec_2nd_df = pd.DataFrame(self.ro_2nd_SEC_log)
        ro_sec_total_df = pd.DataFrame(self.ro_total_SEC_log)
        rejection_1st_df = pd.DataFrame(self.ro_1st_rejection_log)
        rejection_2nd_df = pd.DataFrame(self.ro_2nd_rejection_log)
        rejection_total_df = pd.DataFrame(self.ro_total_rejection_log)
        
        action_df.to_csv(os.path.join(save_dir, 'action_log.csv'))
        operational_var_1st_df.to_csv(os.path.join(save_dir, 'operational_var_1st_df.csv'))
        operational_var_2nd_df.to_csv(os.path.join(save_dir, 'operational_var_2nd_df.csv'))
        brine_var_1st_df.to_csv(os.path.join(save_dir, 'brine_var_1st_df.csv'))
        brine_var_2nd_df.to_csv(os.path.join(save_dir, 'brine_var_2nd_df.csv'))
        permeate_var_1st_df.to_csv(os.path.join(save_dir, 'permeate_var_1st_df.csv'))
        permeate_var_2nd_df.to_csv(os.path.join(save_dir, 'permeate_var_2nd_df.csv'))
        permeate_var_total_df.to_csv(os.path.join(save_dir, 'permeate_var_total_df.csv'))
        ro_sec_1st_df.to_csv(os.path.join(save_dir, 'ro_sec_1st_df.csv'))
        ro_sec_2nd_df.to_csv(os.path.join(save_dir, 'ro_sec_2nd_df.csv'))
        ro_sec_total_df.to_csv(os.path.join(save_dir, 'ro_sec_total_df.csv'))
        rejection_1st_df.to_csv(os.path.join(save_dir, 'rejection_1st_df.csv'))
        rejection_2nd_df.to_csv(os.path.join(save_dir, 'rejection_2nd_df.csv'))
        rejection_total_df.to_csv(os.path.join(save_dir, 'rejection_total_df.csv'))

        # if plot_state:
        #     with open(os.path.join(save_dir, 'state_var_1st.pkl'), 'wb') as file:
        #         pickle.dump(self.state_var_1st_log, file)
        #     with open(os.path.join(save_dir, 'state_var_2nd.pkl'), 'wb') as file:
        #         pickle.dump(self.state_var_2nd_log, file)

    def scale_observation(self, observations):
        observations_copy = deepcopy(observations)
        for a in self.agents:
            obs_space = self.observation_space(a)
            observations_copy[a]['observation'] = (observations[a]['observation'] - obs_space.low) / (obs_space.high - obs_space.low)
        return observations_copy
    
    def set_reward_w(self, reward_Ws):
        self.w_SEC = reward_Ws[0]
        self.w_eff = reward_Ws[1]
    
    def blackbox(self, blackbox_1st, blackbox_2nd, path):
        # If the process diverged, save blackbox of the process.
        if not self.converged:
            with open(os.path.join(path, f"EPISODE {self.episode_id}_1ST_BLACKBOX.pkl"), 'wb') as file:
                pickle.dump(blackbox_1st, file)
            with open(os.path.join(path, f"EPISODE {self.episode_id}_2ND_BLACKBOX.pkl"), 'wb') as file:
                pickle.dump(blackbox_2nd, file)            


    """
     The following methods are internal methods used for modeling and variables processing.
    """
    def _get_transition(self, actions, observations, rewards, done) -> Dict:
        """
         Gather information from environment and form into Transition named tuple.
        """
        transition = {
            'episode_id':self.episode_id, 'previous_state':copy(self.previous_state), 'previous_observations':copy(self.previous_observation),
            'state':copy(self.state(normalize=True)), 'actions':copy(actions), 'rewards':rewards[self.possible_agents[0]], 'observations':copy(observations),
            'done':done
        }
        return transition

    def _process_modeling(self, feed_scenario: np.ndarray, starting_index, modeling_length):
        """
         This method is internal method used to model 2 stage RO process and save the values on attributes.
         After defining self.operational_var_1st_stage (with self.ro_1st_hpp_pressure) and self.ro_2nd_boosting_pressure,
        attributes are updated and logged by calling this method.
         
        """

        sliced_feed_scenario = feed_scenario[starting_index: starting_index + modeling_length, :]
        sliced_feed_scenario = jlconvert(T = self.jl.Array, x = sliced_feed_scenario)

        # State variables and action values (flowrate, recovery setpoints) must be configured beforehand calling this method.
        state_var_updated_1st_stage, state_var_updated_2nd_stage, permeate_1st_log, permeate_2nd_log, brine_1st_log, brine_2nd_log, recovery_1st_log, recovery_2nd_log, op_var_1st_log, op_var_2nd_log, blackbox_1st, blackbox_2nd, SEC_1st_log, SEC_2nd_log, SEC_total_log, converged \
            = self.pressure_controlled_ro(sliced_feed_scenario, self.influent_flowrate, self.ro_1st_pressure, self.ro_2nd_pressure, self.state_var_1st_stage, self.state_var_2nd_stage)

        # Save snapshots of state variables.
        self.state_var_1st_stage = copy(state_var_updated_1st_stage)
        self.state_var_2nd_stage = copy(state_var_updated_2nd_stage)

        self.converged = converged

        if (not self.converged) and (len(permeate_1st_log) <= 1):
            print("Not enough step is proceeded to make result.")
            return False

        target_logs = {
            "OP_VAR_1"  : op_var_1st_log,
            "OP_VAR_2"  : op_var_2nd_log,
            "PERM_VAR_1": permeate_1st_log,
            "PERM_VAR_2": permeate_2nd_log,
            "CONC_VAR_1": brine_1st_log,
            "CONC_VAR_2": brine_2nd_log,
            "RECOVERY_1": recovery_1st_log,
            "RECOVERY_2": recovery_2nd_log,
            "SEC_1": SEC_1st_log,
            "SEC_2": SEC_2nd_log,
            "SEC_TOTAL": SEC_total_log
        }
        
        logs_raw = {}
        logs_mean = {}
        for target_var_name, target_log in target_logs.items():
            logs_raw[target_var_name] = {}
            logs_mean[target_var_name] = {}
            if self.jl.isa(target_log[0], self.jl.Dict):
                for key in target_log[0].keys():
                    logs_raw[target_var_name][key] = (np.array([item[key] for item in target_log]))
                    if not self.converged:
                        logs_raw[target_var_name][key] = logs_raw[target_var_name][key][:-1]
                    if key in ['C', 'C_CF']:
                        logs_mean[target_var_name][key] = (np.average(logs_raw[target_var_name][key], weights=logs_raw[target_var_name]['Q']))
                    else:
                        logs_mean[target_var_name][key] = (np.mean(logs_raw[target_var_name][key]))
            else:
                logs_raw[target_var_name] = np.array(target_log)
                if not self.converged:
                    logs_raw[target_var_name] = logs_raw[target_var_name][:-1]
                if target_var_name == 'SEC_1':
                    logs_mean[target_var_name] = np.average(logs_raw[target_var_name], weights=logs_raw['PERM_VAR_1']['Q'])
                if target_var_name == 'SEC_2':
                    logs_mean[target_var_name] = np.average(logs_raw[target_var_name], weights=logs_raw['PERM_VAR_2']['Q'])
                if target_var_name == 'SEC_TOTAL':
                    logs_mean[target_var_name] = np.average(logs_raw[target_var_name], weights=logs_raw['PERM_VAR_1']['Q'] + logs_raw['PERM_VAR_2']['Q'])
        
        if not os.path.exists(os.path.join(self.save_dir, f"episode {self.episode_id}")):
            os.makedirs(os.path.join(self.save_dir, f"episode {self.episode_id}"))
        # with open(os.path.join(self.save_dir, f"episode {self.episode_id}/{self.control_timestep}_RAW_LOG.pkl"), 'wb') as file:
        #     pickle.dump(logs_raw, file)

        # Save the results in class attributes.
        # Note that it is mostly for compatiablility between the previous version of the environment,
        # thus someday need to be implemented in more precise and efficient manner.
        self.operational_var_1st_stage = copy(logs_mean["OP_VAR_1"])
        self.operational_var_2nd_stage = copy(logs_mean["OP_VAR_2"])
        self.permeate_var_1st_stage = copy(logs_mean["PERM_VAR_1"])
        self.permeate_var_2nd_stage = copy(logs_mean["PERM_VAR_2"])
        self.brine_var_1st_stage = copy(logs_mean["CONC_VAR_1"])
        self.brine_var_2nd_stage = copy(logs_mean["CONC_VAR_2"])

        # Save the results in log lists.
        self.operational_var_1st_log.append(copy(logs_mean["OP_VAR_1"]))
        self.permeate_var_1st_log.append(copy(logs_mean["PERM_VAR_1"]))
        self.brine_var_1st_log.append(copy(logs_mean["CONC_VAR_1"]))

        # Save the results in log lists.
        self.operational_var_2nd_log.append((copy(logs_mean["OP_VAR_2"])))
        self.permeate_var_2nd_log.append((copy(logs_mean["PERM_VAR_2"])))
        self.brine_var_2nd_log.append((copy(logs_mean["CONC_VAR_2"])))

        # Calculate the final products (permeates and brines).
        self._mix_permeates()
        self.brine_total = copy(self.brine_var_2nd_stage)

        # Calculate and save SEC of 1st, 2nd and total process.
        self.ro_1st_SEC     = np.average(a=logs_raw["SEC_1"], weights=logs_raw["PERM_VAR_1"]["Q"])
        self.ro_2nd_SEC     = np.average(a=logs_raw["SEC_2"], weights=logs_raw["PERM_VAR_2"]["Q"])
        self.ro_total_SEC   = np.average(a=logs_raw["SEC_TOTAL"], weights=logs_raw["PERM_VAR_1"]["Q"]+logs_raw["PERM_VAR_2"]["Q"])
        self.ro_1st_SEC_log.append(copy(self.ro_1st_SEC))
        self.ro_2nd_SEC_log.append(copy(self.ro_2nd_SEC))
        self.ro_total_SEC_log.append(copy(self.ro_total_SEC))

        # Calculate and save recovery of 1st, 2nd and total process.
        self.ro_1st_recovery     = np.average(a=logs_raw["RECOVERY_1"], weights=logs_raw["PERM_VAR_1"]["Q"])
        self.ro_2nd_recovery     = np.average(a=logs_raw["RECOVERY_2"], weights=logs_raw["PERM_VAR_2"]["Q"])
        self.ro_total_recovery   = self.operational_var_1st_stage["Q"] * self.ro_1st_pvs / self.permeate_total["Q"]
        self.ro_1st_recovery_log.append(copy(self.ro_1st_recovery))
        self.ro_2nd_recovery_log.append(copy(self.ro_2nd_recovery))
        self.ro_total_recovery_log.append(copy(self.ro_total_recovery))

        # Calculate and save rejection of 1st, 2nd and total process.
        self.ro_1st_rejection     = 1 - np.average(a=logs_raw["PERM_VAR_1"]["C"]/logs_raw["OP_VAR_1"]["C"], weights=logs_raw["PERM_VAR_1"]["Q"])
        self.ro_2nd_rejection     = 1 - np.average(a=logs_raw["PERM_VAR_2"]["C"]/logs_raw["OP_VAR_2"]["C"], weights=logs_raw["PERM_VAR_2"]["Q"])
        self.ro_total_rejection   = 1 - self.permeate_total["C"]/self.operational_var_1st_stage["C"]
        self.ro_1st_rejection_log.append(copy(self.ro_1st_rejection))
        self.ro_2nd_rejection_log.append(copy(self.ro_2nd_rejection))
        self.ro_total_rejection_log.append(copy(self.ro_total_rejection))

        # Save the results in log lists
        self.permeate_total_log.append(copy(self.permeate_total))
        self.brine_total_log.append(copy(self.brine_total))
        self.state_var_1st_log.append(copy(self.state_var_1st_stage))
        self.state_var_2nd_log.append(copy(self.state_var_2nd_stage))

        self.HPP_last = np.maximum(logs_raw["OP_VAR_1"]["P"][-1], 0.0)
        self.IBP_last = np.maximum(logs_raw["OP_VAR_2"]["P"][-1] - logs_raw["CONC_VAR_1"]["P"][-1], 0.0)

        # self.blackbox(blackbox_1st=blackbox_1st, blackbox_2nd=blackbox_2nd, path = self.save_dir)
        return True


    def _mix_permeates(self):
        """
        This method calculates information of permeates of 1st and 2nd stage RO mixed, and updates self.permeate_total.
        The flowrate is summed up, and the other terms are calculated as weighted mean, with flowrate values as weights.
        """
        self.permeate_total = {
            'Q': self.permeate_var_1st_stage['Q'] + self.permeate_var_2nd_stage['Q'],
            'C': np.average([self.permeate_var_1st_stage['C'], self.permeate_var_2nd_stage['C']],
                            weights=[self.permeate_var_1st_stage['Q'], self.permeate_var_2nd_stage['Q']]),
            'T': np.average([self.permeate_var_1st_stage['T'], self.permeate_var_2nd_stage['T']],
                            weights=[self.permeate_var_1st_stage['Q'], self.permeate_var_2nd_stage['Q']]),
            'P': np.average([self.permeate_var_1st_stage['P'], self.permeate_var_2nd_stage['P']],
                            weights=[self.permeate_var_1st_stage['Q'], self.permeate_var_2nd_stage['Q']])
        }

    def _calculate_SEC(self):
        """
        This method calculates SEC of 1st, 2nd and total SEC as kWh/m3 and update attributes.
        """

        # Calculate the power required as kW for each RO stages.
        ''' Unit Conversion
         Multiply 1e5 to convert from bar to Pa
         Multiply self.ro_nst_pvs to match vessel-wise operational_var with stage-wise permeate_var.
         Divide 60 twice to convert from hour to seconds (W is Joule/Sec)
         Divide 1e3 to convert from W to kW
        '''
        ro_1st_power_required = (self.ro_1st_hpp_pressure * 1e5) * \
                                (self.operational_var_1st_stage['Q'] * self.ro_1st_pvs) \
                                / 60 / 60 / self.ro_1st_hpp_eff / 1e3
        # TODO: Implement Energy Recovery Device (ERD)
        ro_2nd_power_required = (self.ro_2nd_boosting_pressure * 1e5) * \
                                (self.operational_var_2nd_stage['Q'] * self.ro_2nd_pvs) \
                                / 60 / 60 / self.ro_2nd_boosting_eff / 1e3

        # Calculate the SEC for each RO stages and total.
        self.ro_1st_SEC     = ro_1st_power_required / self.permeate_var_1st_stage['Q']
        self.ro_2nd_SEC     = ro_2nd_power_required / self.permeate_var_2nd_stage['Q']
        self.ro_total_SEC   = (ro_1st_power_required + ro_2nd_power_required) / self.permeate_total['Q']

        if self.ro_1st_SEC < 0:
            self.state_var_1st_stage['converged'] = False
        if self.ro_2nd_SEC < 0:
            self.state_var_2nd_stage['converged'] = False

        # Save the results in log lists
        self.ro_1st_SEC_log.append(copy(self.ro_1st_SEC))
        self.ro_2nd_SEC_log.append(copy(self.ro_2nd_SEC))
        self.ro_total_SEC_log.append(copy(self.ro_total_SEC))

    def _calculate_rejection(self):
        """
        Rejection rate calculation. Calculate rejection rate as real number (not percent).
        """
        self.ro_1st_rejection   = 1 - (self.permeate_var_1st_stage['C'] / self.operational_var_1st_stage['C'])
        self.ro_2nd_rejection   = 1 - (self.permeate_var_2nd_stage['C'] / self.operational_var_2nd_stage['C'])
        self.ro_total_rejection = 1 - (self.permeate_total['C'] / self.operational_var_1st_stage['C'])

        if self.ro_1st_rejection < 0 or self.ro_1st_rejection > 1:
            self.state_var_1st_stage['converged'] = False
        if self.ro_2nd_rejection < 0 or self.ro_2nd_rejection > 1:
            self.state_var_2nd_stage['converged'] = False

        self.ro_1st_rejection_log.append(copy(self.ro_1st_rejection))
        self.ro_2nd_rejection_log.append(copy(self.ro_2nd_rejection))
        self.ro_total_rejection_log.append(copy(self.ro_total_rejection))

    def _calculate_recovery(self):
        """
        Recovery rate calculation. Calculate recovery rate as real number (not percent).
        """
        self.ro_1st_recovery    = (self.permeate_var_1st_stage['Q'] / (self.operational_var_1st_stage['Q'] * self.ro_1st_pvs))
        self.ro_2nd_recovery    = (self.permeate_var_2nd_stage['Q'] / (self.operational_var_2nd_stage['Q'] * self.ro_2nd_pvs))
        self.ro_total_recovery  = (self.permeate_total['Q'] / self.influent_flowrate)

        if self.ro_1st_recovery < 0 or self.ro_1st_recovery > 1:
            self.state_var_1st_stage['converged'] = False
        if self.ro_2nd_recovery < 0 or self.ro_2nd_recovery > 1:
            self.state_var_2nd_stage['converged'] = False

        self.ro_1st_recovery_log.append(copy(self.ro_1st_recovery))
        self.ro_2nd_recovery_log.append(copy(self.ro_2nd_recovery))
        self.ro_total_recovery_log.append(copy(self.ro_total_recovery))

    def _calculate_reward(self, terminate_if_diverge=True, total_production_term = True):
        # Reward weight factors.
        w_SEC = self.w_SEC
        # w_eff = 0.5
        w_eff = self.w_eff
        # Penalty weight factors.
        # p_CIP = 20.0
        p_diverge = 1.0

        w_reward = 1.0

        # Performance measurement.
        total_production_criteria = 1000.0
        total_production_lower_lim = 700.0

        def calculate_production_term(total_production):
            production_term = np.minimum(1.0, 1.0/(total_production_criteria-total_production_lower_lim) * (total_production-total_production_lower_lim))
            return production_term

        performance_SEC = -3.0 * (self.ro_total_SEC)

        performance_eff = 0

        if total_production_term:
            performance_eff += calculate_production_term(self.permeate_total["Q"])

        if not self.converged:
            divergence = float(not self.converged)
            reward_total = -p_diverge * divergence
            # reward_total = 0.0
        else:
            # Reward calculation.
            reward_total = ((w_SEC * performance_SEC + w_eff * performance_eff))
        return reward_total * w_reward
        
    def _calculate_osmo_p(self, temperature, concentration):
        """
         Calculate osmotic pressure by feed water as Pa.
        """
        osmo_p = (2 / 58.44e3) * concentration * 8.3145e3 * (temperature + 273.15)
        return osmo_p
    
    def _generate_action_mask(self) -> dict[AgentID, np.ndarray]:
        """
         Define action masks.
         Note that the restriction by action masks can be loose or eliminated, and let agents themselves learn to avoid situations
        that used to require action masking (too big or small pressure, negative value, etc.)
        """
        # For flowrate agent, the maximum flowrate is 17.0 m3/hr * 82 = 1394 m3/hr. 17.0 is from ESPA2 membrane datasheet.
        # Mask influent flowrate action by flowrate threshold.

        mode = "AUTO"
        # mode = "MANUAL"

        influent_flowrate_threshold = [750.0 / self.ro_1st_pvs, 1394.0 / self.ro_1st_pvs]
        ro_1st_threshold = [5.0, 15.0]
        ro_2nd_threshold = [0.25, 2.5]

        influent_flowrate_action_mask = np.ones(5, dtype=np.int8)
        if self.operational_var_1st_stage['Q'] < influent_flowrate_threshold[0]:
            influent_flowrate_action_mask[0:2] = 0
        elif self.operational_var_1st_stage['Q'] > influent_flowrate_threshold[1]:
            influent_flowrate_action_mask[3:] = 0

        # Mask 1st pump action by recovery rate threshold.
        ro_1st_action_mask = np.ones(5, dtype=np.int8)
        if self.ro_1st_pressure < ro_1st_threshold[0]:
            ro_1st_action_mask[0:2] = 0
        elif self.ro_1st_pressure > ro_1st_threshold[1]:
            ro_1st_action_mask[3:] = 0

        # Mask 2nd pump action by recovery rate threshold.
        # ro_2nd_osmo_p = self._calculate_osmo_p(temperature = self.operational_var_2nd_stage['T'], concentration = self.operational_var_2nd_stage['C']) / 1e5
        ro_2nd_action_mask = np.ones(5, dtype=np.int8)
        if self.ro_2nd_pressure < ro_2nd_threshold[0]:
            ro_2nd_action_mask[0:2] = 0
        elif self.ro_2nd_pressure > ro_2nd_threshold[1]:
            ro_2nd_action_mask[3:] = 0

        # CIP_action_mask = np.ones(2, dtype=np.int8)
        # CIP_action_mask[1] = 0

        if mode == "MANUAL":
            influent_flowrate_action_mask[0:2] = 0
            influent_flowrate_action_mask[3:] = 0
            ro_1st_action_mask[0:2] = 0
            ro_1st_action_mask[3:] = 0
            ro_2nd_action_mask[0:2] = 0
            ro_2nd_action_mask[3:] = 0
        
        return {
            "influent_flowrate": influent_flowrate_action_mask,
            "1st_stage_pump": ro_1st_action_mask,
            "2nd_stage_pump": ro_2nd_action_mask,
            # "Clean_In_Place": CIP_action_mask
        }

    def sample_scenario(self, len_scenario, concentration=None, range=None, noise=False):
        self.start_point = np.random.choice(np.arange(self.total_simulation_time - (len_scenario + 1)))
        if range is None:
            feed_scenario = deepcopy(self.feed_scenario_total[self.start_point:self.start_point+len_scenario, :])
        else:
            # According to range, add variation in concentration term.
            feed_scenario = deepcopy(self.feed_scenario_total[self.start_point:self.start_point+len_scenario, :])
            if concentration is None:
                variation = np.random.rand() * range * 2 - range
                feed_scenario[:, 1] += variation
            else:
                feed_scenario[:,1] = np.ones_like(feed_scenario[:,1])*concentration
                if noise:
                    feed_scenario[:,1] += np.random.normal(loc=0, scale=1, size=feed_scenario[:,1].shape)
        max_timestep = feed_scenario.shape[0] - 1

        return feed_scenario, max_timestep


    def cleanup(self):
        print("Terminating program. Goodbye 😘")


if __name__ == '__main__':
    print("Don't run this environment file directly.")