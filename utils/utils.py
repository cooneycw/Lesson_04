# utils.py - Utility functions for bankruptcy simulation analysis

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, interact


def hello(name):
    """Simple test function to verify module loading"""
    print(name)


def simulate_bankruptcy(initial_capital=1000000, claim_rate=5, avg_claim_size=50000,
                        investment_return=0.05, num_simulations=100):
    """
    Simulate multiple claim processes to determine time to bankruptcy

    Parameters:
    - initial_capital: Starting capital amount (default: $1,000,000)
    - claim_rate: Average number of claims per year (default: 5)
    - avg_claim_size: Average size of each claim (default: $50,000)
    - investment_return: Annual return on invested capital (default: 5%)
    - num_simulations: Number of simulations to run (default: 100)

    Returns:
    - numpy array of bankruptcy times (in years) for each simulation
    """
    bankruptcy_times = []

    for _ in range(num_simulations):
        capital = initial_capital
        time = 0

        while capital > 0 and time < 100:  # Max 100 years
            # Apply investment returns first
            capital *= (1 + investment_return)

            # Generate claims for this year using Poisson distribution
            num_claims = np.random.poisson(claim_rate)

            # Process each claim
            for _ in range(num_claims):
                # Claim amounts follow exponential distribution
                claim_amount = np.random.exponential(avg_claim_size)
                capital -= claim_amount

                if capital <= 0:
                    break

            time += 1

        bankruptcy_times.append(time)

    return np.array(bankruptcy_times)


def create_dashboard():
    """
    Create an interactive dashboard for bankruptcy simulation

    This function creates sliders for claim rate and average claim size,
    then generates interactive plots showing bankruptcy time distribution
    and survival curves based on the selected parameters.
    """
    # Create widgets for user input
    claim_rate_slider = FloatSlider(
        min=1,
        max=20,
        step=0.5,
        value=5,
        description='Claims/Year:',
        continuous_update=False  # Only update when user releases slider
    )

    avg_claim_size_slider = FloatSlider(
        min=10000,
        max=200000,
        step=5000,
        value=50000,
        description='Avg Claim ($):',
        continuous_update=False
    )

    def update_plot(claim_rate, avg_claim_size):
        """Update plots based on slider values"""
        # Run simulation with current parameters
        bankruptcy_times = simulate_bankruptcy(
            claim_rate=claim_rate,
            avg_claim_size=avg_claim_size
        )

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Histogram of bankruptcy times
        ax1.hist(bankruptcy_times, bins=20, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Years until Bankruptcy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Bankruptcy Times')
        ax1.grid(True, alpha=0.3)

        # Add mean and median lines
        mean_time = np.mean(bankruptcy_times)
        median_time = np.median(bankruptcy_times)
        ax1.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.1f}')
        ax1.axvline(median_time, color='green', linestyle='--', label=f'Median: {median_time:.1f}')
        ax1.legend()

        # Subplot 2: Survival curve
        sorted_times = np.sort(bankruptcy_times)
        survival_prob = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)

        ax2.plot(sorted_times, survival_prob, linewidth=2)
        ax2.set_xlabel('Years')
        ax2.set_ylabel('Survival Probability')
        ax2.set_title('Survival Curve')
        ax2.grid(True, alpha=0.3)

        # Add key probability markers
        prob_5year = np.mean(bankruptcy_times > 5)
        prob_10year = np.mean(bankruptcy_times > 10)
        ax2.axhline(prob_5year, color='red', linestyle=':', alpha=0.7)
        ax2.axhline(prob_10year, color='green', linestyle=':', alpha=0.7)
        ax2.text(1, prob_5year + 0.02, f'5-year survival: {prob_5year:.1%}', color='red')
        ax2.text(1, prob_10year + 0.02, f'10-year survival: {prob_10year:.1%}', color='green')

        plt.tight_layout()
        plt.show()

        # Display summary statistics
        print(f"Simulation Results (Claims/Year: {claim_rate}, Avg Claim: ${avg_claim_size:,})")
        print(f"Mean bankruptcy time: {mean_time:.1f} years")
        print(f"Median bankruptcy time: {median_time:.1f} years")
        print(f"Standard deviation: {np.std(bankruptcy_times):.1f} years")
        print(f"5-year survival probability: {prob_5year:.1%}")
        print(f"10-year survival probability: {prob_10year:.1%}")

    # Create interactive widget
    interact(update_plot,
             claim_rate=claim_rate_slider,
             avg_claim_size=avg_claim_size_slider)