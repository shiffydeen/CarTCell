# simulation.py
import pygame
import random

# --- Constants from the paper (can be tuned) ---
GRID_WIDTH = 50
GRID_HEIGHT = 50
INITIAL_CELL_COUNT = 20
SIMULATION_STEPS = 1600 # 7 days * 24 hours/day * 60 min/hour / 6 min/step

# --- New constants for simulation logic ---
MOVE_PROBABILITY = 0.8
CONVERSION_PROBABILITY = 0.1
BEAD_EXHAUSTION_RATE = 0.01
NATURAL_EXHAUSTION_RATE = 0.001
PROLIFERATION_PROBABILITY = 0.05
MIN_PROLIFERATION_POTENCY = 0.6
# Age is in simulation steps (6 min/step). 2 days = 480 steps.
PROLIFERATION_AGE_MIN_STEPS = 480
PROLIFERATION_AGE_MAX_STEPS = 960

class Cell:
    """ Represents a single T-cell with its properties. """
    def __init__(self, x, y, is_activated=False, potency=0.0):
        self.x = x
        self.y = y
        self.is_activated = is_activated
        self.potency = potency
        self.age = 0

    def update(self, occupied_spaces, bead_locations):
        """ Update cell state for one time step. """
        self.age += 1

        # 1. Movement Logic
        if random.random() < MOVE_PROBABILITY:
            possible_moves = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    new_x, new_y = self.x + dx, self.y + dy
                    if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and (new_x, new_y) not in occupied_spaces:
                        possible_moves.append((new_x, new_y))
            if possible_moves:
                self.x, self.y = random.choice(possible_moves)

        # 2. Activation & Exhaustion Logic
        if self.is_activated:
            # Natural exhaustion for all activated cells
            self.potency -= NATURAL_EXHAUSTION_RATE
            # Bead-induced exhaustion
            if (self.x, self.y) in bead_locations:
                self.potency -= BEAD_EXHAUSTION_RATE
            self.potency = max(0, self.potency)
        else:
            # Activation for naive cells
            if (self.x, self.y) in bead_locations and random.random() < CONVERSION_PROBABILITY:
                self.is_activated = True
                self.potency = 1.0

class Simulation:
    """ Manages the entire simulation grid, cells, and beads. """
    def __init__(self):
        self.cells = []
        self.beads = []
        self._seed_cells()

    def _seed_cells(self):
        """ Place initial naive T-cells randomly on the grid, ensuring no overlaps. """
        self.cells = []
        occupied_spaces = set()
        while len(self.cells) < INITIAL_CELL_COUNT:
            x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
            if (x, y) not in occupied_spaces:
                self.cells.append(Cell(x, y))
                occupied_spaces.add((x,y))

    def add_beads(self, count=10):
        """ Add activation beads to the simulation. """
        for _ in range(count):
            x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
            self.beads.append((x, y))

    def remove_beads(self):
        """ Remove all beads from the simulation. """
        self.beads = []

    def run_step(self):
        """ Run one step of the Monte Carlo simulation. """
        occupied_spaces = {(cell.x, cell.y) for cell in self.cells}
        bead_locations = set(self.beads)
        newly_created_cells = []

        # Update each cell
        for cell in self.cells:
            original_pos = (cell.x, cell.y)
            occupied_spaces.remove(original_pos)
            cell.update(occupied_spaces, bead_locations)
            occupied_spaces.add((cell.x, cell.y))

        # Handle proliferation after all cells have moved
        for cell in self.cells:
            if cell.is_activated and cell.potency > MIN_PROLIFERATION_POTENCY and \
               PROLIFERATION_AGE_MIN_STEPS <= cell.age <= PROLIFERATION_AGE_MAX_STEPS and \
               random.random() < PROLIFERATION_PROBABILITY:
                
                # Find an empty spot for the new cell
                possible_spots = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        new_x, new_y = cell.x + dx, cell.y + dy
                        if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and (new_x, new_y) not in occupied_spaces:
                            possible_spots.append((new_x, new_y))
                
                if possible_spots:
                    new_cell_x, new_cell_y = random.choice(possible_spots)
                    new_cell = Cell(new_cell_x, new_cell_y) # Born as a naive cell
                    newly_created_cells.append(new_cell)
                    occupied_spaces.add((new_cell_x, new_cell_y))
                    cell.age = 0 # Parent cell's age resets

        self.cells.extend(newly_created_cells)

    def get_observation(self):
        """ Gathers the current state of the simulation for the RL agent. """
        if not self.cells:
            return {
                "total_cells": 0,
                "num_activated": 0,
                "num_naive": 0,
                "avg_potency": 0,
                "bead_count": len(self.beads),
            }

        num_activated = sum(1 for cell in self.cells if cell.is_activated)
        avg_potency = sum(c.potency for c in self.cells) / len(self.cells)
        
        return {
            "total_cells": len(self.cells),
            "num_activated": num_activated,
            "num_naive": len(self.cells) - num_activated,
            "avg_potency": avg_potency,
            "bead_count": len(self.beads),
        }

    def reset(self):
        """ Resets the simulation to its initial state. """
        self.__init__()

    def to_json(self):
        """ Serializes the simulation state to a JSON-friendly dictionary. """
        return {
            'cells': [
                {'x': cell.x, 'y': cell.y, 'is_activated': cell.is_activated, 'potency': cell.potency}
                for cell in self.cells
            ],
            'beads': self.beads,
            'metrics': self.get_observation()
        }
