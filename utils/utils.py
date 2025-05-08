# Basic utility functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, interact, IntSlider, widgets
from IPython.display import display

# Add any existing utility functions here
def hello(message):
    """Print a message to the console."""
    print(message)

def simulate_bankruptcy():
    """Placeholder for the simulate_bankruptcy function."""
    print("Simulating bankruptcy...")

def create_dashboard():
    """Placeholder for the create_dashboard function."""
    print("Creating dashboard...")

# This function is added to maintain backward compatibility
# It imports and calls run_insurance_fundamentals from the dedicated module
def run_insurance_fundamentals():
    """
    Wrapper function that calls the run_insurance_fundamentals function from the dedicated module
    """
    from utils.run_insurance_fundamentals import run_insurance_fundamentals as run
    run()