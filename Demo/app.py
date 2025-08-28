# app.py
import sys
import os

# Add the parent directory to the Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from stable_baselines3 import PPO
from environment import CarTCellEnv
import time
import threading
import numpy as np

# --- Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Simulation Profiles (connecting to GSE data narrative) ---
SIMULATION_PROFILES = {
    "default": {
        "CONVERSION_PROBABILITY": 0.1,
        "BEAD_EXHAUSTION_RATE": 0.01,
        "PROLIFERATION_PROBABILITY": 0.05,
        "NATURAL_EXHAUSTION_RATE": 0.001,
        "name": "Standard Cell Profile"
    },
    "Patient_GSE246342": {  # More aggressive/easily exhausted cell type
        "CONVERSION_PROBABILITY": 0.15,
        "BEAD_EXHAUSTION_RATE": 0.02,
        "PROLIFERATION_PROBABILITY": 0.06,
        "NATURAL_EXHAUSTION_RATE": 0.002,
        "name": "High-Response Patient (GSE246342)"
    },
    "Patient_GSM7867490": {  # More resilient cell type
        "CONVERSION_PROBABILITY": 0.08,
        "BEAD_EXHAUSTION_RATE": 0.005,
        "PROLIFERATION_PROBABILITY": 0.04,
        "NATURAL_EXHAUSTION_RATE": 0.0005,
        "name": "Resilient Patient (GSM7867490)"
    },
    "Patient_GSM7867492": {  # Moderate response
        "CONVERSION_PROBABILITY": 0.12,
        "BEAD_EXHAUSTION_RATE": 0.015,
        "PROLIFERATION_PROBABILITY": 0.055,
        "NATURAL_EXHAUSTION_RATE": 0.0015,
        "name": "Moderate Response Patient (GSM7867492)"
    }
}

# --- Global Variables ---
env = None
model = None
current_profile = "default"
simulation_running = False
simulation_thread = None

# --- Helper Functions ---
def apply_profile_to_simulation(profile_name):
    """Apply simulation profile parameters to the environment."""
    global env
    if profile_name in SIMULATION_PROFILES:
        profile = SIMULATION_PROFILES[profile_name]
        
        # Import simulation module to modify constants
        import simulation
        simulation.CONVERSION_PROBABILITY = profile["CONVERSION_PROBABILITY"]
        simulation.BEAD_EXHAUSTION_RATE = profile["BEAD_EXHAUSTION_RATE"]
        simulation.PROLIFERATION_PROBABILITY = profile["PROLIFERATION_PROBABILITY"]
        simulation.NATURAL_EXHAUSTION_RATE = profile["NATURAL_EXHAUSTION_RATE"]
        
        # Reset environment to apply new parameters
        if env:
            env.reset()

def load_model():
    """Load the trained PPO model."""
    global model, env
    try:
        env = CarTCellEnv()
        # Try to load a pre-trained model, fallback to creating a new one
        model_paths = [
            "/home/emms/Downloads/CarTCell/results/models/ppo_cart_1000000.zip"
        ]
        
        model_loaded = False
        for path in model_paths:
            try:
                model = PPO.load(path, env=env)
                print(f"Model loaded from {path}")
                model_loaded = True
                break
            except:
                continue
        
        if not model_loaded:
            print("No pre-trained model found, creating new PPO model for demo")
            model = PPO("MlpPolicy", env, verbose=1)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        env = CarTCellEnv()
        model = PPO("MlpPolicy", env, verbose=1)

# --- Web Routes ---
@app.route('/')
def index():
    return render_template('index.html', profiles=SIMULATION_PROFILES)

# --- WebSocket Handlers ---
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send available profiles to client
    emit('profiles_available', {
        'profiles': {k: v['name'] for k, v in SIMULATION_PROFILES.items()},
        'current': current_profile
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global simulation_running
    simulation_running = False

@socketio.on('load_profile')
def handle_load_profile(data):
    """Handle profile change request from client."""
    global current_profile
    profile_name = data.get('profile_name', 'default')
    if profile_name in SIMULATION_PROFILES:
        current_profile = profile_name
        apply_profile_to_simulation(profile_name)
        print(f"Profile changed to: {SIMULATION_PROFILES[profile_name]['name']}")
        emit('profile_loaded', {
            'profile_name': profile_name, 
            'profile_display': SIMULATION_PROFILES[profile_name]['name']
        })

@socketio.on('run_scenario')
def handle_run_scenario(data):
    """Handle scenario execution request from client."""
    global simulation_running, simulation_thread
    
    if simulation_running:
        return  # Already running
    
    scenario_name = data.get('scenario_name', 'standard_protocol')
    simulation_running = True
    
    # Start simulation in a background task managed by SocketIO
    socketio.start_background_task(
        target=run_simulation_scenario, 
        scenario_name=scenario_name
    )

@socketio.on('reset_simulation')
def handle_reset():
    """Handle simulation reset request."""
    global simulation_running
    simulation_running = False
    if env:
        env.reset()
    emit('simulation_reset')

def run_simulation_scenario(scenario_name):
    """Run the specified simulation scenario."""
    global simulation_running, env, model
    
    with app.app_context():  # This fixes the context issue
        if not env or not model:
            load_model()
        
        try:
            obs, _ = env.reset()
            step_count = 0
            max_steps = env.max_steps
            
            socketio.emit('scenario_started', {'scenario': scenario_name})
            
            while simulation_running and step_count < max_steps:
                if scenario_name == 'standard_protocol':
                    # Standard protocol: Add beads on first step, then skip
                    action = 0 if step_count == 0 else 2
                elif scenario_name == 'ai_strategy':
                    # AI strategy: Let the model decide
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = 2  # Default to skip
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Get simulation state and send to client
                state_json = env.simulation.to_json()
                state_json['last_action'] = action
                state_json['step'] = step_count
                state_json['scenario'] = scenario_name
                state_json['reward'] = float(reward)
                
                socketio.emit('update_state', state_json)
                
                step_count += 1
                
                if terminated or truncated:
                    break
                
                # Control demo speed
                socketio.sleep(0.5)
            
            # Send episode complete signal
            final_metrics = env.simulation.get_observation()
            potent_cells = len([c for c in env.simulation.cells if c.potency > 0.8])
            
            socketio.emit('episode_complete', {
                'scenario': scenario_name,
                'final_metrics': final_metrics,
                'potent_cells': potent_cells,
                'total_steps': step_count
            })
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            socketio.emit('simulation_error', {'error': str(e)})
        finally:
            simulation_running = False

# --- Initialize ---
load_model()
apply_profile_to_simulation(current_profile)

# --- Main Execution ---
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
