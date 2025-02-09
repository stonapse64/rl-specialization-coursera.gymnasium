"""
An action-value method-based experiment for a multi-armed bandit.

"""

# import standard libs
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor
# import third party libs
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
# import local libs
from mab_bandit import Bandit
from mab_player import Player


class ExperimentMAB:
    """
    A class to represent an experiment for a multi-armed bandit problem.

    Attributes
    ----------
    casino_config : dict
        Configuration for the experiment runs and timesteps.
    bandit_config : dict
        Configuration for the bandit.
    player_configs : dict
        Configuration for the players.
    system_load : float
        The system load for distributing the processes.

    Methods
    -------
    batch_sizing():
        Determines the batch size for running the experiment.
    run_experiment():
        Runs the multi-armed bandit experiment.
    run_batches(_batch_size):
        Runs batches of the multi-armed bandit experiment.
    plot_results(average_rewards_all_runs, percentage_optimal_actions, save_plots=False):
        Plots the results of the experiment.

    Example configuration dictionaries
    ----------------------------------

    casino_config = {"runs": 10, "timesteps": 1000}

    bandit_config = {"bandit_actions": 10, "initial_action_value": 0.5,
                     "q_drift_mean": 0.0, "q_drift_std_dev": 0.01,
                     "stationary": False, "random_seed": None}

    player_configs = {
        "sample average": {"epsilon": 0.1, "alpha": 0., "random_seed": None},
        "constant stepsize": {"epsilon": 0.1, "alpha": 0.1,
                              "random_seed": None}
    }

    For fun and comparison: you can add a cheater to your player
    configuration that would always draw the arm with the maximum
    reward. epsilon and alpha can be arbitrary as they won't be used.

    player_configs = {
    "cheater": {"epsilon": 1., "alpha": 1., "random_seed": None},
    "other player": {...},
    ...,
    }

    Example usage
    -------------

    experiment = ExperimentMAB(casino_config, bandit_config, player_configs, system_load=0.8)
    rewards, loss, optimal_actions = experiment.run_experiment()
    experiment.plot_results(rewards, loss, optimal_actions, save_plots=True)

    """

    def __init__(self, casino_config, bandit_config, player_configs, system_load=0.8):
        """
        Initializes the ExperimentMAB class with given configurations.

        Parameters
        ----------
        casino_config : dict
            Configuration for the experiment runs and timesteps.
        bandit_config : dict
            Configuration for the bandit.
        player_configs : dict
            Configuration for the players.
        system_load : float, optional
            The system load for distributing the processes (default is 0.8).
        """
        self.casino_config = casino_config
        self.bandit_config = bandit_config
        self.player_configs = player_configs
        self.system_load = system_load
        self.average_rewards_all_runs, self.average_loss_all_runs, self.optimal_actions_taken_all_runs = {}, {}, {}
        self.batches, self.batch_size = self.size_batches()

    def size_batches(self):
        """
        Determines the batch size for running the experiment based on system load and available CPU cores.

        Returns
        -------
        tuple
            A tuple containing the number of batches and the batch size.
        """
        _num_cores = os.cpu_count()
        _batches = int(_num_cores * self.system_load)
        _batch_size = self.casino_config["runs"] // _batches
        _effective_runs = _batches * _batch_size
        self.casino_config["runs"] = _effective_runs
        print(f'{_effective_runs} total runs from {_batches} batches with {_batch_size} runs each.')
        return _batches, _batch_size

    def run_experiment(self):
        """
        Runs the multi-armed bandit experiment with parallel processing.

        Returns
        -------
        tuple
            A tuple containing average rewards and percentage of optimal actions for all players across all runs.
        """
        with ProcessPoolExecutor(max_workers=self.batches) as executor:
            results = list(executor.map(self.run_batches, [self.batch_size for _ in range(self.batches)]))

        cumulative_rewards = {player: 0 for player in self.player_configs}
        cumulative_loss = {player: 0 for player in self.player_configs}
        cumulative_optimal_actions = {player: 0 for player in self.player_configs}

        for result in results:
            rewards, loss, optimal_actions = result
            for player in self.player_configs:
                cumulative_rewards[player] += rewards[player]
                cumulative_loss[player] += loss[player]
                cumulative_optimal_actions[player] += optimal_actions[player]

        self.average_rewards_all_runs = {player: cumulative_rewards[player] / self.batches for player in self.player_configs}
        self.average_loss_all_runs = {player: cumulative_loss[player] / self.batches for player in self.player_configs}
        self.optimal_actions_taken_all_runs = {player: cumulative_optimal_actions[player] / self.batches for player in self.player_configs}

        return self.average_rewards_all_runs, self.average_loss_all_runs, self.optimal_actions_taken_all_runs

    def run_batches(self, _batch_size):
        """
        Runs batches of the multi-armed bandit experiment.

        Parameters
        ----------
        _batch_size : int
            The number of runs per batch.

        Returns
        -------
        tuple
            A tuple containing average rewards and percentage of optimal actions per timestep across all runs.
        """
        runs, timesteps = _batch_size, self.casino_config["timesteps"]
        players, rewards_run, loss_run, optimal_actions, average_rewards_all_runs, average_loss_all_runs, percentage_optimal_actions = {}, {}, {}, {}, {}, {}, {}

        for player in self.player_configs:
            average_rewards_all_runs[player] = np.zeros(timesteps)
            average_loss_all_runs[player] = np.zeros(timesteps)
            optimal_actions[player] = np.zeros(timesteps)
            percentage_optimal_actions[player] = np.zeros(timesteps)

        for _run in trange(runs, desc="Runs"):
            bandit = Bandit(self.bandit_config)
            for player in self.player_configs:
                players[player] = Player(self.player_configs[player], bandit)
                rewards_run[player] = np.zeros(timesteps)
                loss_run[player] = np.zeros (timesteps)

            for timestep in range(timesteps):
                for player in players:

                    if player == "cheater":
                        action, reward = np.argmax(bandit.arms), np.max(bandit.arms)
                    else:
                        action, reward = players[player].move()

                    rewards_run[player][timestep] = reward
                    optimal_actions[player][timestep] += 1 if action == np.argmax(bandit.arms) else 0
                    loss_run[player][timestep] = np.max(bandit.arms) - reward

                    bandit.drift()

            for player in players:
                average_rewards_all_runs[player] += (1 / (runs + 1)) * (rewards_run[player] - average_rewards_all_runs[player])
                average_loss_all_runs[player] += (1 / (runs + 1)) * (
                            loss_run[player] - average_loss_all_runs[
                        player])

        for player in players:
            percentage_optimal_actions[player] = optimal_actions[player] / runs

        return average_rewards_all_runs, average_loss_all_runs, percentage_optimal_actions

    def plot_results(self, average_rewards_all_runs, average_loss_all_runs,
                     percentage_optimal_actions, save_plots=False):
        """
        Plots the results of the multi-armed bandit experiment.

        Parameters
        ----------
        average_rewards_all_runs : dict
            Average rewards for all runs.
        percentage_optimal_actions : dict
            Percentage of optimal actions for all runs.
        average_loss_all_runs : dict
            Average loss for all runs.
        save_plots : bool, optional
            Whether to save the plots (default is False).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%I_%M_%S_%p")
        runs = self.batches * self.batch_size

        # loss
        plt.figure(figsize=(10, 6))
        title = ""
        for i, player in enumerate(self.player_configs):
            plt.plot(average_loss_all_runs[player][:], label=f'{player} player', linestyle="--" if player == "cheater" else None, alpha=self.alpha(i))
            title += f'{player} player: {self.player_configs[player]}\n'
        plt.title(f'Average loss per timestep over {runs} runs\n{title}')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Loss')
        plt.legend()
        if save_plots:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(f'plots/average_loss_all_runs_{timestamp}.png',
                        bbox_inches='tight')
            print(f'Plot Average loss per timestep over {runs} runs saved to plots/average_loss_all_runs{timestamp}.png')
        plt.show()

        # rewards
        plt.figure(figsize=(10, 6))
        title = ""
        for i, player in enumerate(self.player_configs):
            plt.plot(average_rewards_all_runs[player][:], label=f'{player} player', linestyle="--" if player == "cheater" else None, alpha=self.alpha(i))
            title += f'{player} player: {self.player_configs[player]}\n'
        plt.title(f'Average rewards per timestep over {runs} runs\n{title}')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward')
        plt.legend()
        if save_plots:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(f'plots/average_rewards_all_runs_{timestamp}.png',
                        bbox_inches='tight')
            print(f'Plot Average rewards per timestep over {runs} runs saved to plots/average_rewards_all_runs{timestamp}.png')
        plt.show()

        # rewards cum.
        plt.figure(figsize=(10, 6))
        title = ""
        for i, player in enumerate(self.player_configs):
            plt.plot(np.cumsum(average_rewards_all_runs[player]), label=f'{player} player', linestyle="--" if player == "cheater" else None, alpha=self.alpha(i))
            title += f'{player} player: {self.player_configs[player]}\n'
        plt.title(f'Cumulative Average rewards per timestep over {runs} runs\n{title}')
        plt.xlabel('Timesteps')
        plt.ylabel('Cumulative Average Reward')
        plt.legend()
        if save_plots:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(f'plots/cumulative_average_rewards_all_run'
                        f's_{timestamp}.png',
                        bbox_inches='tight')
            print(f'Plot Cumulative Average rewards per timestep over {runs} runs saved to plots/average_rewards_all_runs{timestamp}.png')
        plt.show()

        # optimal actions
        plt.figure(figsize=(10, 6))
        title = ""
        for i, player in enumerate(self.player_configs):
            plt.plot(percentage_optimal_actions[player][:], label=f'{player} player', linestyle="--" if player == "cheater" else None, alpha=self.alpha(i))
            title += f'{player} player: {self.player_configs[player]}\n'
        plt.title(f'Percentage optimal actions per timestep over {runs} runs\n{title}')
        plt.xlabel('Timesteps')
        plt.ylabel('% Optimal Actions')
        plt.legend()
        if save_plots:
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(f'plots/percentage_optimal_actions_all_runs'
                        f'_{timestamp}.png', bbox_inches='tight')
            print(f'(Plot Percentage optimal actions per timestep over {runs} runs saved to plots/percentage_optimal_actions_all_runs{timestamp}.png)')
        plt.show()

        # sample random walk of arms values

    def alpha(self, i:int) -> float:
        """
        Calculates the alpha value for matplotlib.plt calls so that the
        plots do not completely cover each other.
        :param i: layer number, generated by code
        :return: alpha value
        """
        x = len(self.player_configs) -i
        a, b, c, d = 0.8, 0.5, 0.7,  0.2
        alpha = a * np.exp(-b * x**c) + d
        return alpha
