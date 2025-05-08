
# risk_pooling.py module contents here
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, interact
from IPython.display import display

def demonstrate_risk_pooling():
    """
    Demonstrates the concept of risk pooling in insurance
    """
    # History tracking
    history = []

    def update_plot(accident_probability=0.05, claim_amount=10000, num_policyholders=100):
        # Keep track of results for display
        nonlocal history

        # Run the simulation - generate random accidents
        accidents = np.random.random(num_policyholders) < accident_probability

        # Calculate results
        individual_costs = np.where(accidents, claim_amount, 0)
        total_losses = np.sum(individual_costs)
        fair_premium = accident_probability * claim_amount
        pool_premium_total = fair_premium * num_policyholders

        # Calculate stats
        max_individual_loss = np.max(individual_costs)
        percent_with_loss = np.mean(accidents) * 100
        pool_performance = total_losses / pool_premium_total

        # Store in history (limit to 5 items)
        history.append({
            'accident_probability': accident_probability,
            'claim_amount': claim_amount,
            'num_policyholders': num_policyholders,
            'fair_premium': fair_premium,
            'total_losses': total_losses,
            'pool_premium_total': pool_premium_total,
            'percent_with_loss': percent_with_loss,
            'pool_performance': pool_performance
        })
        if len(history) > 5:
            history.pop(0)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Individual outcomes (first 50 policyholders)
        display_n = min(50, num_policyholders)
        x = np.arange(display_n)
        ax1.bar(x, individual_costs[:display_n], color='blue', alpha=0.7)
        ax1.axhline(fair_premium, color='red', linestyle='--',
                    label=f'Fair premium: ${fair_premium:,.0f}')
        ax1.set_xlabel('Individual Policyholders')
        ax1.set_ylabel('Cost ($)')
        ax1.set_title(f'Individual Risk (showing first {display_n} policyholders)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add explanatory text
        ax1.text(0.5, 0.95,
                 f"Without insurance: {percent_with_loss:.1f}% of people pay ${claim_amount:,.0f}\nWith insurance: Everyone pays ${fair_premium:,.0f}",
                 transform=ax1.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

        # Plot 2: Pooled outcome
        ax2.bar(['Premiums Collected', 'Actual Losses'],
                [pool_premium_total, total_losses],
                color=['green', 'blue'], alpha=0.7)
        ax2.set_ylabel('Amount ($)')
        ax2.set_title(f'Pooled Risk ({num_policyholders:,} policyholders)')
        ax2.grid(True, alpha=0.3)

        # Add explanatory text
        performance_text = "Surplus" if pool_performance < 1 else "Deficit"
        performance_color = "green" if pool_performance < 1 else "red"
        ax2.text(0.5, 0.95,
                 f"Expected losses: ${pool_premium_total:,.0f}\nActual losses: ${total_losses:,.0f}\n{performance_text}: ${abs(pool_premium_total - total_losses):,.0f}",
                 transform=ax2.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

        # Add annotation for ratio
        ax2.text(1, total_losses + 0.05 * max(pool_premium_total, total_losses),
                 f"Actual/Expected: {pool_performance:.2f}", ha='center', color=performance_color)

        plt.tight_layout()
        plt.show()

        # Display history table
        print("\nHistory of Recent Simulations:")
        history_table = []
        for i, h in enumerate(history, 1):
            history_table.append({
                'Run': i,
                'Accident Probability': f"{h['accident_probability']:.1%}",
                'Premium': f"${h['fair_premium']:,.0f}",
                'Policyholders': f"{h['num_policyholders']:,}",
                'Actual Losses': f"${h['total_losses']:,.0f}",
                'Actual/Expected': f"{h['pool_performance']:.2f}"
            })

        display(pd.DataFrame(history_table).set_index('Run'))

        # Display insurance interpretation
        print("\nInsurance Interpretation:")
        print(f"• Individual Risk: Each person has a {accident_probability:.1%} chance of a ${claim_amount:,.0f} loss.")
        print(
            f"• Without Insurance: {percent_with_loss:.1f}% of people faced a ${claim_amount:,.0f} loss in this simulation.")
        print(f"• With Insurance: Everyone pays a fair premium of ${fair_premium:.0f}.")
        print(
            f"• Risk Pooling Result: The pool collected ${pool_premium_total:,.0f} and paid ${total_losses:,.0f} in claims.")
        if pool_performance < 1:
            print(f"• This year the insurance pool had a surplus (collected more than paid out).")
        else:
            print(f"• This year the insurance pool had a deficit (paid out more than collected).")
        print(f"• As the number of policyholders increases, the ratio of actual to expected losses approaches 1.0.")

    # Create interactive widget
    probability_slider = FloatSlider(
        min=0.01,
        max=0.25,
        step=0.01,
        value=0.05,
        description='Accident Probability:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    interact(update_plot, accident_probability=probability_slider)