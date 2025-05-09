# fixed_capital_role.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, IntSlider, interact
from IPython.display import display


def demonstrate_capital_role():
    """
    Demonstrates how capital protects an insurance company from bankruptcy
    """
    # History tracking
    history = []

    def update_plot(capital_amount=50, num_years=10):
        """Update the capital simulation using absolute capital in millions"""
        # Keep track of results for display
        nonlocal history

        # Set random seed for consistent results
        np.random.seed(42)

        # Base parameters
        annual_premium = 100.0  # $100M annual premium - consistent with balance sheet
        expected_loss_ratio = 0.65  # Expected losses are 65% of premium
        expense_ratio = 0.30  # Expenses are 30% of premium
        investment_return = 0.05  # 5% annual return on investments

        # Calculate capital amount based on selected amount in millions
        initial_capital = capital_amount
        capital_ratio = initial_capital / annual_premium

        # Run more simulations for better consistency
        num_simulations = 500
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
            'capital_amount': capital_amount,
            'initial_capital': initial_capital,
            'num_years': num_years,
            'survival_rate': survival_rate,
            'average_years': average_years
        })
        if len(history) > 5:
            history.pop(0)

        # Create figure - stacked vertically for better readability
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot 1: Survival distribution
        survived = [y >= num_years for y in years_survived]
        ax1.pie([sum(survived), len(survived) - sum(survived)],
                labels=[f'Survived all {num_years} years', 'Failed before year ' + str(num_years)],
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90,
                explode=(0.1, 0), textprops={'fontsize': 12})
        ax1.set_title(f'Survival Rate with ${capital_amount:.1f}M Initial Capital', fontsize=14)

        # Plot 2: Years survived histogram - Adjust bins and labels based on years
        if num_years <= 25:
            # For smaller year ranges, use every year
            bins = np.arange(0.5, num_years + 1.5, 1)
            xticks = range(1, num_years + 1)
        else:
            # For larger year ranges, use bins at 5-year intervals
            bins = np.arange(0.5, num_years + 1.5, 5)  # Bins every 5 years
            xticks = range(5, num_years + 1, 5)  # Show labels every 5 years

        ax2.hist(years_survived, bins=bins, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_xticks(xticks)
        ax2.set_xlabel('Years Survived', fontsize=12)
        ax2.set_ylabel('Number of Companies', fontsize=12)
        ax2.set_title('Distribution of Survival Years', fontsize=14)

        # Add a vertical line for average
        ax2.axvline(average_years, color='red', linestyle='--',
                    label=f'Average: {average_years:.1f} years')
        ax2.legend(fontsize=12)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.show()

        # Display survival curve with adjusted labels for large year ranges
        plt.figure(figsize=(12, 6))

        # Calculate survival curve
        survival_counts = []

        # Handle labeling differently based on number of years
        if num_years <= 25:
            # For smaller year ranges, calculate and show every year
            years_to_show = range(1, num_years + 1)
            for year in years_to_show:
                survival_counts.append(sum(1 for y in years_survived if y >= year) / num_simulations)

            # Plot every year point
            plt.plot(years_to_show, survival_counts, 'bo-', linewidth=2, markersize=8)

            # Add annotations (every year)
            for year, rate in enumerate(survival_counts, 1):
                plt.annotate(f'{rate:.0%}', (year, rate), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=10)
        else:
            # For larger year ranges, calculate every year but show every 5 years
            full_survival_counts = []
            for year in range(1, num_years + 1):
                full_survival_counts.append(sum(1 for y in years_survived if y >= year) / num_simulations)

            # Plot a smooth line using all years
            plt.plot(range(1, num_years + 1), full_survival_counts, 'b-', linewidth=2)

            # Plot markers only at 5-year intervals
            years_to_show = range(5, num_years + 1, 5)
            markers_survival = [full_survival_counts[y - 1] for y in years_to_show]
            plt.plot(years_to_show, markers_survival, 'bo', markersize=8)

            # Add annotations (every 5 years)
            for i, year in enumerate(years_to_show):
                rate = markers_survival[i]
                plt.annotate(f'{rate:.0%}', (year, rate), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=10)

        # Add text box with key statistics - fixed at top-right for consistency
        stats_text = f"Initial Capital: ${initial_capital:.1f}M\n" \
                     f"Annual Premium: ${annual_premium:.1f}M\n" \
                     f"Capital Ratio: {capital_ratio:.1f}x Premium\n" \
                     f"Survival Rate (All {num_years} Years): {survival_rate:.1%}\n" \
                     f"Average Survival: {average_years:.1f} years"

        # Simple positioning at top-right
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        plt.show()

        # Display history table
        print("\nHistory of Recent Simulations:")
        history_table = []
        for i, h in enumerate(history, 1):
            history_table.append({
                'Run': i,
                'Initial Capital': f"${h['capital_amount']:.1f}M",
                'Simulation Years': h['num_years'],
                'Survival Rate': f"{h['survival_rate']:.1%}",
                'Average Survival': f"{h['average_years']:.1f} years"
            })

        display(pd.DataFrame(history_table).set_index('Run'))

        # Display insurance interpretation
        print("\nInsurance Interpretation:")
        print(f"• Capital serves as a buffer against unexpected losses.")
        print(
            f"• With ${capital_amount:.1f}M of initial capital ({capital_ratio:.1f}x annual premium), {survival_rate:.1%} of companies survived all {num_years} years.")
        print(f"• Higher capital amounts mean better protection against insolvency.")
        print(f"• Insurance regulators require minimum capital levels to ensure companies can pay claims.")

        if survival_rate < 0.90:
            print(f"• Warning: This capital level may be inadequate for long-term stability.")
            print(f"• Recommendation: Increase capital to improve survival probability.")
        else:
            print(f"• This capital level appears adequate with a high survival probability.")

    # Create interactive widgets - now using capital in absolute dollars
    capital_slider = FloatSlider(
        min=10,
        max=100,
        step=10,
        value=50,
        description='Initial Capital ($M):',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    year_slider = IntSlider(
        min=5,
        max=100,
        step=5,
        value=10,
        description='Simulation Years:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    interact(update_plot,
             capital_amount=capital_slider,
             num_years=year_slider)