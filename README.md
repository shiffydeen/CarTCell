# CAR T-Cell Manufacturing Digital Twin

This project is an interactive simulation and reinforcement learning demonstration for optimizing the manufacturing process of CAR T-cells. It uses a digital twin of a cell culture to train an AI agent that can discover novel manufacturing protocols.

The current implementation features an **Online Reinforcement Learning** agent trained in a synthetic environment.

## Core Components

The project is built on three main Python modules: `simulation.py`, `environment.py`, and `Demo/app.py`.

### 1. `simulation.py` - The Virtual Petri Dish

This module contains the low-level logic for the cell culture simulation. It defines the rules of the "game" that the AI agent will play.

-   **`Cell` Class**: Represents a single T-cell with properties like position (`x`, `y`), `is_activated`, `potency`, and `age`.
-   **`Simulation` Class**: Manages the entire grid, a collection of `Cell` objects, and a list of activation `beads`.
-   **Core Mechanics**:
    -   **Activation**: Naive cells become activated and gain 100% potency when they are near an activation bead.
    -   **Exhaustion**: Activated cells gradually lose `potency` over time. This process is accelerated if they remain in contact with a bead.
    -   **Proliferation**: Activated cells with high potency can multiply, creating new naive cells.
    -   **State Tracking**: The simulation tracks key metrics like total cell count, number of activated cells, and average potency.

### 2. `environment.py` - The RL Playground

This module wraps the `simulation` in a standard `gymnasium.Env` interface, making it compatible with reinforcement learning libraries like `stable-baselines3`. This is where we define the rules of the RL problem.

-   **State (Observation)**: At each decision point, the agent observes a 5-element vector representing the current state of the culture:
    1.  Total number of cells
    2.  Number of activated cells
    3.  Average potency of all cells
    4.  Number of activation beads present
    5.  Time remaining in the simulation

-   **Actions**: The agent can choose one of three discrete actions:
    1.  **`0: Add Beads`**: Add a new batch of activation beads to the culture.
    2.  **`1: Remove Beads`**: Remove all existing beads.
    3.  **`2: Skip`**: Do nothing and let the simulation continue.

-   **Reward**: The agent is rewarded based on how its actions affect the cell culture's health. The goal is to encourage the growth of highly potent cells.
    -   It receives small positive or negative rewards throughout the simulation based on the change in average cell potency.
    -   It receives a large final reward based on the total number of highly potent cells at the end of the process.

### 3. The Reinforcement Learning Agent (Online PPO)

The AI brain of the operation is an agent trained using **Proximal Policy Optimization (PPO)**, a state-of-the-art **Online RL** algorithm from the `stable-baselines3` library.

-   **Training**: The agent is trained by having it run the `CarTCellEnv` environment for millions of steps. Through trial and error, it learns a **policy**—a strategy that maps a given `state` to an optimal `action`.
-   **The Learned Policy**: The agent learns a nuanced strategy. For example, it might learn to add beads to activate cells, but then remove them after a certain period to prevent the cells from becoming exhausted, a strategy that can lead to a higher final yield of potent cells.
-   **The Model**: The trained agent, including its learned policy, is saved in the file `results/models/ppo_cart_1000000.zip`.

## Interactive Web Demo (`Demo/app.py`)

The web demo provides a visual interface to see the trained AI agent in action and compare its performance to a standard protocol.

-   **Backend**: A `Flask` and `Flask-SocketIO` server that loads the pre-trained PPO model.
-   **Scenarios**:
    1.  **Standard Protocol**: A hard-coded, fixed strategy that adds beads at the beginning and does nothing else.
    2.  **AI Strategy**: The loaded PPO agent observes the state of the simulation and makes decisions in real-time.
-   **Visualization**: The backend runs the `CarTCellEnv` and streams the full state (cell positions, potency, bead locations) to the frontend via WebSockets, which then renders the animated simulation on an HTML canvas.

## How to Run the Demo

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -r Demo/requirements.txt
    ```

2.  **Run the Web Server**:
    ```bash
    python Demo/app.py
    ```

3.  Open a web browser and navigate to `http://127.0.0.1:5000`.

## Future Work: Moving to an Offline RL Approach

The current implementation is a successful proof-of-concept using synthetic data. The next phase of this project, as outlined in `extracts/rl cart cell.md`, involves a significant evolution:

-   **Using Real Data**: We will leverage the real patient single-cell data in the `GSE/` folder.
-   **Cell-State Encoder**: A new model will be trained to translate the high-dimensional gene expression data of each cell into a rich, low-dimensional "latent state."
-   **Offline RL**: Because we will be learning from a fixed, historical dataset of patient trajectories, we will shift from *online* RL (PPO) to **Offline RL** algorithms (e.g., CQL, IQL). This will allow the agent to learn a policy without needing to interact with a live environment.
