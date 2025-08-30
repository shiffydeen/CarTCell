# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation import Simulation # Assuming simulation.py is in the same directory

# --- Constants ---
SIMULATION_STEPS = 1600

class CarTCellEnv(gym.Env):
    """ Custom Gymnasium Environment for the CAR T-Cell Simulation. """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, control_interval=32):
        super().__init__()
        self.simulation = Simulation()
        self.control_interval = control_interval # Number of sim steps between agent actions
        self.current_step = 0
        self.max_steps = SIMULATION_STEPS // control_interval
        self.previous_avg_potency = 0
        self.previous_potent_cells = 0

        # Action space: 0: Add beads, 1: Remove beads, 2: Skip
        self.action_space = spaces.Discrete(3)

        # Observation space (tabular): [total_cells, num_activated, avg_potency, bead_count, time_left]
        # Using Box space for continuous/integer values.
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )

    def _get_obs(self):
        obs_dict = self.simulation.get_observation()
        time_left = (self.max_steps - self.current_step) / self.max_steps
        return np.array([
            obs_dict['total_cells'],
            obs_dict['num_activated'],
            obs_dict['avg_potency'],
            obs_dict['bead_count'],
            time_left
        ], dtype=np.float32)

    def _calculate_reward(self, obs_dict):
        """ Calculate reward based on the paper's description. """
        if self.current_step >= self.max_steps:
            # Final reward
            potent_cells = [c for c in self.simulation.cells if c.potency > 0.8]
            return sum(c.potency for c in potent_cells) * 100
        
        # Intermediate reward based on potency change
        current_avg_potency = obs_dict['avg_potency']
        reward = 0
        if self.previous_avg_potency > 0: # Avoid division by zero
            ratio = current_avg_potency / self.previous_avg_potency
            if ratio > 0.9:
                reward = 5
            elif ratio > 0.8:
                reward = 1
            elif ratio > 0.5:
                reward = -1
            else:
                reward = -5
        
        self.previous_avg_potency = current_avg_potency
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.reset()
        self.current_step = 0
        self.previous_avg_potency = 0
        self.previous_potent_cells = 0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # 1. Apply action
        if action == 0:
            self.simulation.add_beads()
        elif action == 1:
            self.simulation.remove_beads()
        # action == 2 means do nothing

        # 2. Run simulation for the control interval
        for _ in range(self.control_interval):
            self.simulation.run_step()

        self.current_step += 1

        # 3. Get observation, reward, and termination status
        obs_dict = self.simulation.get_observation()
        observation = self._get_obs()
        reward = self._calculate_reward(obs_dict)
        
        terminated = self.current_step >= self.max_steps
        truncated = False # Not used here, but required by the API
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        # TODO: Implement visualization using Pygame if needed
        pass

    def close(self):
        # pygame.quit() # If using pygame for rendering
        pass
