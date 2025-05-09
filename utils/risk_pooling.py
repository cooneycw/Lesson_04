# improved_risk_pooling.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, IntSlider, interact
from IPython.display import display


def demonstrate_risk_pooling():
    """
    Demonstrates the concept of risk pooling in insurance with improved visualizations
    """
    # History tracking
    history = []

    # Fixed claim amount at $20,000
    CLAIM_AMOUNT = 20000

    def update_plot(accident_probability=0.05, num_policyholders=100):
        # Keep track of results for display
        nonlocal history

        # Fixed claim amount
        claim_amount = CLAIM_AMOUNT

        # Run the simulation - generate random accidents
        accidents = np.random.random(num_policyholders) < accident_probability

        # Calculate results
        individual_costs = np.where(accidents, claim_amount, 0)
        total_losses = np.sum(individual_costs)
        fair_premium = accident_probability * claim_amount
        pool_premium_total = fair_premium * num_policyholders

        # Calculate stats
        num_with_loss = np.sum(accidents)
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Individual outcomes with improved visualization
        display_n = min(50, num_policyholders)

        # Create a categorical plot to better show binary outcomes
        categories = ['No Loss', 'Loss']
        counts = [display_n - np.sum(accidents[:display_n]), np.sum(accidents[:display_n])]

        # Blue bar chart for individual outcomes
        ax1.bar(
            ['Without Insurance'],
            [claim_amount],
            color='lightblue',
            alpha=0.3,
            width=0.6,
            label='Potential loss amount'
        )

        # Overlay scatter plot showing actual outcomes
        x_positions = np.ones(display_n) * 0  # All points at x=0 ("Without Insurance")
        y_positions = individual_costs[:display_n]  # Each person's actual outcome

        # Add jitter to x positions for better visualization
        x_jitter = np.random.uniform(-0.2, 0.2, size=display_n)
        x_positions += x_jitter

        # Plot the actual outcomes as scatter points
        ax1.scatter(
            x_positions,
            y_positions,
            color='blue',
            alpha=0.7,
            label=f'Individual outcomes (n={display_n})'
        )

        # Add a bar for premium with insurance
        ax1.bar(
            ['With Insurance'],
            [fair_premium],
            color='green',
            alpha=0.7,
            width=0.6,
            label='Insurance premium'
        )

        # Annotation showing how many people experienced a loss
        ax1.annotate(
            f"{num_with_loss} out of {num_policyholders} people\nexperienced a ${claim_amount:,} loss",
            xy=(0, claim_amount / 2),
            xytext=(0, claim_amount * 0.7),
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        )

        # Annotation explaining insurance premium
        ax1.annotate(
            f"Everyone pays\n${fair_premium:,.0f}",
            xy=(1, fair_premium / 2),
            xytext=(1, fair_premium * 1.5),
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8)
        )

        ax1.set_ylabel('Cost ($)')
        ax1.set_title('Individual Risk Outcomes vs Pooled Outcomes')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend(loc='upper center')

        # Set y-axis limit to ensure visibility of premium
        ax1.set_ylim(0, claim_amount * 1.1)

        # Plot 2: Pooled outcome (insurer perspective)
        ax2.bar(['Premiums Collected', 'Actual Losses'],
                [pool_premium_total, total_losses],
                color=['green', 'blue'], alpha=0.7)
        ax2.set_ylabel('Amount ($)')
        ax2.set_title('Insurer\'s Perspective')
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

        # No history table for risk pooling slide

        # Display insurance interpretation
        print("\nInsurance Interpretation:")
        print(f"• Individual Risk: Each person has a {accident_probability:.1%} chance of a ${claim_amount:,.0f} loss.")
        print(
            f"• Without Insurance: {num_with_loss} people ({percent_with_loss:.1f}%) faced a ${claim_amount:,.0f} loss in this simulation.")
        print(f"• With Insurance: Everyone pays a premium of ${fair_premium:.0f}.")
        print(
            f"• Risk Pooling Result: The insurer collected ${pool_premium_total:,.0f} and paid ${total_losses:,.0f} in claims.")

        if pool_performance < 1:
            print(
                f"• This year the insurance pool had a ${abs(pool_premium_total - total_losses):,.0f} surplus (collected more than paid out).")
            print(f"• The surplus can be held as capital to handle future years when claims exceed premiums.")
        else:
            print(
                f"• This year the insurance pool had a ${abs(pool_premium_total - total_losses):,.0f} deficit (paid out more than collected).")
            print(f"• The deficit must be covered by the insurer's capital reserves.")

        print(f"\n• Key Insight: As the number of policyholders increases, the 'Actual/Expected' ratio approaches 1.0,")
        print(f"  making the insurance pool's results more predictable and stable.")

    # Create interactive widgets
    probability_slider = FloatSlider(
        min=0.01,
        max=0.25,
        step=0.01,
        value=0.05,
        description='Accident Probability:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    policyholder_slider = IntSlider(
        min=10,
        max=1000,
        step=10,
        value=100,
        description='Number of Policyholders:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    interact(update_plot,
             accident_probability=probability_slider,
             num_policyholders=policyholder_slider)