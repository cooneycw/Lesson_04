# capital_role.py - Demonstrating the role of capital in preventing insurance company ruin

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, IntSlider, interact
import pandas as pd


def demonstrate_capital_role():
    """
    Demonstrates how capital protects an insurance company from bankruptcy

    Shows how different capital levels affect the probability of ruin
    under various scenarios
    """
    # History tracking
    history = []

    def update_plot(capital_ratio=0.5, num_years=10):
        """Update the capital simulation"""
        # Keep track of results for display
        nonlocal history

        # Base parameters
        annual_premium = 1000000  # $1M annual premium
        expected_loss_ratio = 0.65  # Expected losses are 65% of premium
        expense_ratio = 0.30  # Expenses are 30% of premium
        investment_return = 0.05  # 5% annual return on investments

        # Calculate capital amount based on selected ratio
        initial_capital = annual_premium * capital_ratio

        # Run 100 simulations
        num_simulations = 100
        results = []

        for sim in range(num_simulations):
            # Start with initial values
            capital = initial_capital
            years_survived = 0

            # Run simulation for specified number of years or until bankruptcy
            for year in range(1, num_years + 1):
                if capital <= 0:
                    break

                # Generate random loss ratio for this year (normally distributed around expected)
                # Standard deviation of 0.15 means about 1/3 of years have loss ratios above 80% or below 50%
                actual_loss_ratio = np.random.normal(expected_loss_ratio, 0.15)
                actual_loss_ratio = max(0.2, actual_loss_ratio)  # Floor at 20%

                # Calculate this year's underwriting result
                losses = annual_premium * actual_loss_ratio
                expenses = annual_premium * expense_ratio
                underwriting_profit = annual_premium - losses - expenses

                # Apply investment return to beginning capital
                investment_income = capital * investment_return

                # Update capital
                capital += underwriting_profit + investment_income
                years_survived = year

            # Store results of this simulation
            results.append({
                'survived_years': years_survived,
                'final_capital': capital
            })

        # Calculate statistics
        years_survived = [r['survived_years'] for r in results]
        final_capital = [r['final_capital'] for r in results]

        survival_rate = sum(1 for y in years_survived if y >= num_years) / num_simulations
        average_years = np.mean(years_survived)
        average_final_capital = np.mean([c for c in final_capital if c > 0])

        # Store in history (limit to 5 items)
        history.append({
            'capital_ratio': capital_ratio,
            'initial_capital': initial_capital,
            'num_years': num_years,
            'survival_rate': survival_rate,
            'average_years': average_years
        })
        if len(history) > 5:
            history.pop(0)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Survival distribution
        survived = [y >= num_years for y in years_survived]
        ax1.pie([sum(survived), len(survived) - sum(survived)],
                labels=[f'Survived all {num_years} years', 'Failed before year ' + str(num_years)],
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90,
                explode=(0.1, 0))
        ax1.set_title(f'Survival Rate with {capital_ratio:.0%} Capital Ratio')

        # Plot 2: Years survived histogram
        bins = np.arange(0.5, num_years + 1.5, 1)
        ax2.hist(years_survived, bins=bins, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(1, num_years + 1))
        ax2.set_xlabel('Years Survived')
        ax2.set_ylabel('Number of Companies')
        ax2.set_title('Distribution of Survival Years')

        # Add a vertical line for average
        ax2.axvline(average_years, color='red', linestyle='--',
                    label=f'Average: {average_years:.1f} years')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Display survival curve
        plt.figure(figsize=(10, 6))

        # Calculate survival curve
        survival_counts = []
        for year in range(1, num_years + 1):
            survival_counts.append(sum(1 for y in years_survived if y >= year) / num_simulations)

        plt.plot(range(1, num_years + 1), survival_counts, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Year')
        plt.ylabel('Survival Probability')
        plt.title(f'Survival Curve with {capital_ratio:.0%} Capital Ratio')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        # Add annotations
        for year, rate in enumerate(survival_counts, 1):
            plt.annotate(f'{rate:.0%}', (year, rate), textcoords="offset points",
                         xytext=(0, 10), ha='center')

        # Add text box with key statistics
        stats_text = f"Initial Capital: ${initial_capital:,.0f}\n" \
                     f"Capital Ratio: {capital_ratio:.0%}\n" \
                     f"Survival Rate (All {num_years} Years): {survival_rate:.1%}\n" \
                     f"Average Survival: {average_years:.1f} years"

        plt.text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        # Display history table
        print("\nHistory of Recent Simulations:")
        history_table = []
        for i, h in enumerate(history, 1):
            history_table.append({
                'Run': i,
                'Capital Ratio': f"{h['capital_ratio']:.0%}",
                'Initial Capital': f"${h['initial_capital']:,.0f}",
                'Simulation Years': h['num_years'],
                'Survival Rate': f"{h['survival_rate']:.1%}",
                'Average Survival': f"{h['average_years']:.1f} years"
            })

        display(pd.DataFrame(history_table).set_index('Run'))

        # Display insurance interpretation
        print("\nInsurance Interpretation:")
        print(f"• Capital serves as a buffer against unexpected losses.")
        print(
            f"• With a {capital_ratio:.0%} capital ratio (${initial_capital:,.0f}), {survival_rate:.1%} of companies survived all {num_years} years.")
        print(f"• Higher capital ratios mean better protection against insolvency.")
        print(f"• Insurance regulators require minimum capital levels to ensure companies can pay claims.")

        if survival_rate < 0.90:
            print(f"• Warning: This capital level may be inadequate for long-term stability.")
            print(f"• Recommendation: Increase capital ratio to improve survival probability.")
        else:
            print(f"• This capital level appears adequate with a high survival probability.")

    # Create interactive widgets
    capital_slider = FloatSlider(
        min=0.1,
        max=1.0,
        step=0.1,
        value=0.5,
        description='Capital Ratio:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    year_slider = IntSlider(
        min=5,
        max=30,
        step=5,
        value=10,
        description='Simulation Years:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    interact(update_plot,
             capital_ratio=capital_slider,
             num_years=year_slider)

