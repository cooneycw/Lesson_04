# bankruptcy_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider
import ipywidgets as widgets


def simulate_bankruptcy(initial_capital=1000000, claim_rate=5, avg_claim_size=50000,
                        investment_return=0.05, num_simulations=100):
    """Simulate multiple claim processes to determine time to bankruptcy"""
    bankruptcy_times = []

    for _ in range(num_simulations):
        capital = initial_capital
        time = 0

        while capital > 0 and time < 100:  # Max 100 years
            capital *= (1 + investment_return)
            num_claims = np.random.poisson(claim_rate)

            for _ in range(num_claims):
                claim_amount = np.random.exponential(avg_claim_size)
                capital -= claim_amount

                if capital <= 0:
                    break

            time += 1

        bankruptcy_times.append(time)

    return np.array(bankruptcy_times)


def create_dashboard():
    """Create an interactive dashboard for bankruptcy simulation"""
    # Create widgets
    claim_rate_slider = FloatSlider(min=1, max=20, step=0.5, value=5,
                                    description='Claims/Year:')
    avg_claim_size_slider = FloatSlider(min=10000, max=200000, step=5000,
                                        value=50000, description='Avg Claim ($):')

    def update_plot(claim_rate, avg_claim_size):
        # Run simulation
        bankruptcy_times = simulate_bankruptcy(claim_rate=claim_rate,
                                               avg_claim_size=avg_claim_size)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(bankruptcy_times, bins=20, edgecolor='black')
        ax1.set_xlabel('Years until Bankruptcy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Bankruptcy Times')

        # Survival curve
        sorted_times = np.sort(bankruptcy_times)
        survival_prob = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax2.plot(sorted_times, survival_prob)
        ax2.set_xlabel('Years')
        ax2.set_ylabel('Survival Probability')
        ax2.set_title('Survival Curve')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Create interactive widget
    interact(update_plot,
             claim_rate=claim_rate_slider,
             avg_claim_size=avg_claim_size_slider)