# improved_law_of_large_numbers.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, interact
from IPython.display import display


def demonstrate_law_of_large_numbers(true_probability=0.05):
    """
    Demonstrates the Law of Large Numbers using an insurance claims example
    with improved number formatting
    """
    # History tracking
    history = []

    def update_plot(true_probability=0.05):
        # Keep track of results for display
        nonlocal history

        # Create sample sizes (increasing exponentially)
        sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000]

        # Run experiment - simulate accidents for each sample size
        results = []
        for size in sample_sizes:
            # Generate random accidents (1 = accident, 0 = no accident)
            accidents = np.random.random(size) < true_probability
            observed_probability = np.mean(accidents)
            results.append({
                'sample_size': size,
                'observed_probability': observed_probability,
                'true_probability': true_probability,
                'error': abs(observed_probability - true_probability)
            })

        # Add to history (limit to 5 items)
        history.append({
            'true_probability': true_probability,
            'results': results.copy()
        })
        if len(history) > 5:
            history.pop(0)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Observed probability vs sample size
        df = pd.DataFrame(results)
        ax1.semilogx(df['sample_size'], df['observed_probability'], 'bo-', linewidth=2, markersize=8)
        ax1.axhline(true_probability, color='red', linestyle='--', label=f'True probability: {true_probability:.1%}')
        ax1.set_xlabel('Number of Drivers')
        ax1.set_ylabel('Observed Accident Rate')
        ax1.set_title('Observed Accident Rate vs. Sample Size')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Format x-tick labels to avoid scientific notation for numbers under 1 million
        def format_number(x, pos):
            if x >= 1000000:
                return f'{x / 1000000:.0f}M'
            return f'{x:.0f}'

        from matplotlib.ticker import FuncFormatter
        ax1.xaxis.set_major_formatter(FuncFormatter(format_number))

        # Add text annotations for each point
        for i, row in df.iterrows():
            ax1.annotate(f"{row['observed_probability']:.1%}",
                         (row['sample_size'], row['observed_probability']),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        # Plot 2: Error vs sample size
        ax2.loglog(df['sample_size'], df['error'], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Drivers')
        ax2.set_ylabel('Error (|Observed - True|)')
        ax2.set_title('Error vs. Sample Size')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(FuncFormatter(format_number))

        # Add text annotations for each point
        for i, row in df.iterrows():
            ax2.annotate(f"{row['error']:.3f}",
                         (row['sample_size'], row['error']),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        plt.tight_layout()
        plt.show()

        # Display history table
        print("\nHistory of Recent Simulations:")
        history_table = []
        for i, h in enumerate(history, 1):
            history_table.append({
                'Run': i,
                'True Probability': f"{h['true_probability']:.1%}",
                'Small Sample (n=10)': f"{h['results'][0]['observed_probability']:.1%}",
                'Medium Sample (n=1000)': f"{h['results'][5]['observed_probability']:.1%}",
                'Large Sample (n=50000)': f"{h['results'][7]['observed_probability']:.1%}"
            })

        display(pd.DataFrame(history_table).set_index('Run'))

        # Display insurance interpretation
        print("\nInsurance Interpretation:")
        print(f"• With only 10 drivers, the observed accident rate was {results[0]['observed_probability']:.1%}, " +
              f"which is {results[0]['error'] * 100:.1f} percentage points away from the true rate of {true_probability:.1%}")
        print(f"• With 50,000 drivers, the observed accident rate was {results[7]['observed_probability']:.1%}, " +
              f"which is {results[7]['error'] * 100:.1f} percentage points away from the true rate")
        print("\nInsurance companies rely on large numbers of policyholders to make accurate predictions!")

    # Create interactive widget
    slider = FloatSlider(
        min=0.01,
        max=0.25,
        step=0.01,
        value=0.05,
        description='True Accident Probability:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    interact(update_plot, true_probability=slider)