# Basic utility functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, interact, IntSlider, widgets
from IPython.display import display
import os
import importlib.machinery


# Simple utility functions
def hello(message):
    """Print a message to the console."""
    print(message)


def simulate_bankruptcy():
    """Placeholder for the simulate_bankruptcy function."""
    print("Simulating bankruptcy...")


def create_dashboard():
    """Placeholder for the create_dashboard function."""
    print("Creating dashboard...")


# Function to load a module by file path
def load_module(file_path):
    """Load a Python module from a file path."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    loader = importlib.machinery.SourceFileLoader(module_name, file_path)
    return loader.load_module()


# Run insurance fundamentals demo
def run_insurance_fundamentals():
    """
    Main function that runs the insurance fundamentals demo.
    Loads the demonstration functions from their respective files.
    """
    cwd = os.getcwd()

    # Load each module directly by file path
    try:
        # Load the run_insurance_fundamentals module
        run_module_path = os.path.join(cwd, "utils", "run_insurance_fundamentals.py")
        run_module = load_module(run_module_path)

        # Call the function from the loaded module
        run_module.run_insurance_fundamentals()
    except Exception as e:
        print(f"Error running insurance fundamentals: {e}")