# main.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import CarTCellEnv # Assuming environment.py is in the same directory
import os

# Create directories to save results
os.makedirs("results", exist_ok=True)
model_dir = "results/models"
os.makedirs(model_dir, exist_ok=True)

# 1. Instantiate the Environment
env = CarTCellEnv()

# Optional: Check if the environment follows the Gymnasium API
# It's good practice to run this once to ensure your environment is set up correctly.
# try:
#     check_env(env)
#     print("Environment check passed!")
# except Exception as e:
#     print(f"Environment check failed: {e}")

# 2. Instantiate the RL Agent (PPO is a good start, as per the paper)
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=4096,  # Increased for more experience collection
    gamma=0.999,  # Increased for long-term reward focus
    learning_rate=0.0001,  # Slightly reduced for stability
    ent_coef=0.01,  # Increased to encourage exploration
    verbose=1,
    tensorboard_log="./results/ppo_cart_tensorboard/"
)

# 3. Train the Agent
# The paper mentions 1M timesteps, but we can start with less to test.
TIMESTEPS = 1000000
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
model_path = f"{model_dir}/ppo_cart_{TIMESTEPS}.zip"
model.save(model_path)

print(f"--- Training Complete ---")
print(f"Model saved to {model_path}")

# 4. Evaluate the Trained Agent
print("--- Evaluating Trained Agent ---")

# Re-load the trained model (optional, good practice)
# model = PPO.load(model_path, env=env)

obs, _ = env.reset()
for i in range(env.max_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    action_map = {0: "ADD_BEADS", 1: "REMOVE_BEADS", 2: "SKIP"}
    # Use .item() to handle both 0-dim and 1-dim numpy arrays from model.predict()
    print(f"Step {i+1}: Action: {action_map[action.item()]}, Reward: {reward:.2f}")
    if terminated:
        print("Episode finished.")
        final_obs = env.simulation.get_observation()
        print(f"Final State: {final_obs}")
        break
env.close()
