# Add these imports to the top of your utils.py file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, interact, IntSlider, widgets
from IPython.display import display


# Your existing functions (hello, simulate_bankruptcy, create_dashboard) stay as they are

# Add these new functions to the end of your utils.py file:

# 1. LAW OF LARGE NUMBERS
def demonstrate_law_of_large_numbers(true_probability=0.05):
    """
    Demonstrates the Law of Large Numbers using an insurance claims example
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


# 2. RISK POOLING
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


# 3. BALANCE SHEET
def demonstrate_balance_sheet():
    """
    Demonstrates the components of an insurance company balance sheet
    """
    # History tracking
    history = []

    def update_plot(loss_ratio=0.65):
        """Update the balance sheet visualization based on loss ratio"""
        # Keep track of results for display
        nonlocal history

        # Base values for a small insurance company (in millions)
        premium_revenue = 100  # $100M in premium

        # Calculate components based on loss ratio
        # Loss ratio = losses/premium
        expected_losses = premium_revenue * loss_ratio
        expenses = premium_revenue * 0.25  # Fixed expense ratio of 25%
        required_capital = premium_revenue * 0.5  # Regulatory minimum capital

        # Calculate underwriting profit/loss
        underwriting_result = premium_revenue - expected_losses - expenses

        # Calculate investment income (5% return on invested premium+capital)
        investable_assets = premium_revenue + required_capital
        investment_return_rate = 0.05
        investment_income = investable_assets * investment_return_rate

        # Total profit/loss
        total_profit = underwriting_result + investment_income

        # Balance sheet components
        assets = {
            'Cash & Investments': investable_assets,
            'Premiums Receivable': premium_revenue * 0.1,  # 10% of premium not yet collected
            'Other Assets': premium_revenue * 0.05  # 5% in other assets
        }

        liabilities = {
            'Loss Reserves': expected_losses,
            'Unearned Premium': premium_revenue * 0.5,  # Assume 50% of premiums unearned
            'Other Liabilities': expenses * 0.5  # Half of expenses not yet paid
        }

        # Total assets and liabilities
        total_assets = sum(assets.values())
        total_liabilities = sum(liabilities.values())

        # Capital (equity) = assets - liabilities
        capital = total_assets - total_liabilities

        # Capital ratio = capital / premium
        capital_ratio = capital / premium_revenue

        # Store in history (limit to 5 items)
        history.append({
            'loss_ratio': loss_ratio,
            'underwriting_result': underwriting_result,
            'investment_income': investment_income,
            'total_profit': total_profit,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'capital': capital,
            'capital_ratio': capital_ratio
        })
        if len(history) > 5:
            history.pop(0)

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

        # Plot 1: Balance Sheet
        assets_data = pd.Series(assets)
        liabilities_data = pd.Series(liabilities)

        # Add capital to liabilities side (to balance)
        liabilities_plot = liabilities_data.copy()
        liabilities_plot['Capital (Equity)'] = capital

        # Create stacked bar chart
        assets_data.plot(kind='bar', ax=ax1, position=0, width=0.4, color='lightblue')
        liabilities_plot.plot(kind='bar', ax=ax1, position=1, width=0.4,
                              color=['lightgreen', 'lightgreen', 'lightgreen', 'gold'])

        ax1.set_title('Balance Sheet (in $ millions)')
        ax1.set_ylabel('Amount ($ millions)')
        ax1.set_xticklabels(['Assets', 'Liabilities & Capital'])

        # Add totals on top of bars
        ax1.text(0, total_assets + 2, f'${total_assets:.1f}M', ha='center')
        ax1.text(1, total_assets + 2, f'${total_liabilities + capital:.1f}M', ha='center')

        # Add annotations
        for i, (name, value) in enumerate(assets.items()):
            ax1.text(0, sum(list(assets.values())[:i]) + value / 2, f'{name}\n${value:.1f}M', ha='center')

        offset = 0
        for i, (name, value) in enumerate(liabilities_plot.items()):
            color = 'black'
            ax1.text(1, offset + value / 2, f'{name}\n${value:.1f}M', ha='center', color=color)
            offset += value

        # Plot 2: Income components
        income_components = {
            'Premium': premium_revenue,
            'Losses': -expected_losses,
            'Expenses': -expenses,
            'Investment Income': investment_income,
            'Profit/Loss': total_profit
        }

        # Calculate positions for waterfall chart
        cumulative = 0
        bottoms = []
        heights = []

        for name, value in income_components.items():
            if name == 'Profit/Loss':
                bottoms.append(0)
                heights.append(total_profit)
            else:
                bottoms.append(cumulative if value > 0 else cumulative + value)
                heights.append(abs(value))
                cumulative += value

        # Colors based on positive/negative values
        colors = ['blue', 'red', 'red', 'green', 'purple' if total_profit >= 0 else 'red']

        # Create waterfall chart
        bars = ax2.bar(income_components.keys(), heights, bottom=bottoms, color=colors, alpha=0.7)

        # Add values
        for i, (name, value) in enumerate(income_components.items()):
            if name == 'Profit/Loss':
                ax2.text(i, total_profit / 2 if total_profit >= 0 else total_profit / 2,
                         f'${value:.1f}M', ha='center', va='center', color='white' if abs(value) > 10 else 'black')
            else:
                position = bottoms[i] + heights[i] / 2
                ax2.text(i, position, f'${value:.1f}M', ha='center', va='center',
                         color='white' if abs(value) > 10 else 'black')

        ax2.set_title('Income Statement (in $ millions)')
        ax2.set_ylabel('Amount ($ millions)')
        ax2.grid(axis='y', alpha=0.3)

        # Plot 3: Capital Ratio
        min_capital_ratio = 0.5  # Regulatory minimum

        # Create horizontal bar for capital ratio
        ax3.barh(['Capital Ratio'], [capital_ratio], color='green' if capital_ratio >= min_capital_ratio else 'red',
                 alpha=0.7)
        ax3.barh(['Minimum Required'], [min_capital_ratio], color='red', alpha=0.3)

        ax3.set_title('Capital Ratio (Capital / Premium)')
        ax3.set_xlabel('Ratio')
        ax3.set_xlim(0, max(1.0, capital_ratio * 1.2))

        # Add annotations
        ax3.text(capital_ratio, 0, f'{capital_ratio:.2f}', va='center')
        ax3.text(min_capital_ratio, 1, f'{min_capital_ratio:.2f} (Minimum)', va='center')

        # Add interpretation text
        status = "ADEQUATE" if capital_ratio >= min_capital_ratio else "INADEQUATE"
        color = "green" if capital_ratio >= min_capital_ratio else "red"

        ax3.text(0.5, 0.5, f"Capital Status: {status}", transform=ax3.transAxes,
                 ha='center', va='center', fontsize=14, color=color,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout()
        plt.show()

        # Display history table
        print("\nHistory of Recent Simulations:")
        history_table = []
        for i, h in enumerate(history, 1):
            history_table.append({
                'Run': i,
                'Loss Ratio': f"{h['loss_ratio']:.2f}",
                'Underwriting Profit': f"${h['underwriting_result']:.1f}M",
                'Investment Income': f"${h['investment_income']:.1f}M",
                'Total Profit': f"${h['total_profit']:.1f}M",
                'Capital Ratio': f"{h['capital_ratio']:.2f}"
            })

        display(pd.DataFrame(history_table).set_index('Run'))

        # Display insurance interpretation
        print("\nInsurance Interpretation:")
        print(
            f"• Loss Ratio: {loss_ratio:.2f} (${expected_losses:.1f}M in losses per ${premium_revenue:.1f}M in premium)")
        print(f"• Underwriting Result: ${underwriting_result:.1f}M")
        print(f"• Investment Income: ${investment_income:.1f}M")
        print(f"• Total Profit: ${total_profit:.1f}M")
        print(f"• Capital: ${capital:.1f}M (Capital Ratio: {capital_ratio:.2f})")

        if capital_ratio < min_capital_ratio:
            print(f"• ALERT: Capital ratio is below the regulatory minimum of {min_capital_ratio:.2f}!")
            print(
                f"• The company needs at least ${(min_capital_ratio * premium_revenue - capital):.1f}M more capital to meet minimum requirements.")
        else:
            surplus = capital - (min_capital_ratio * premium_revenue)
            print(f"• The company has ${surplus:.1f}M of capital surplus above the regulatory minimum.")

    # Create interactive widget
    slider = FloatSlider(
        min=0.40,
        max=1.0,
        step=0.05,
        value=0.65,
        description='Loss Ratio:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    interact(update_plot, loss_ratio=slider)


# 4. PREMIUM CALCULATION
def demonstrate_premium_calculation():
    """
    Demonstrates how insurance premiums are calculated
    """
    # History tracking
    history = []

    def update_plot(accident_frequency=0.05, claim_severity=8000):
        """Update the premium calculation visualization"""
        # Keep track of results for display
        nonlocal history

        # Calculate components
        expected_loss = accident_frequency * claim_severity
        expense_ratio = 0.25  # Fixed at 25% of premium
        risk_margin_ratio = 0.05  # Fixed at 5% of premium

        # Premium components (solving the equation)
        # Premium = Expected Loss + Expense Ratio × Premium + Risk Margin × Premium
        # Premium = Expected Loss / (1 - Expense Ratio - Risk Margin)
        premium = expected_loss / (1 - expense_ratio - risk_margin_ratio)
        expenses = premium * expense_ratio
        risk_margin = premium * risk_margin_ratio

        # Loading factor
        loading_factor = premium / expected_loss

        # Store in history (limit to 5 items)
        history.append({
            'accident_frequency': accident_frequency,
            'claim_severity': claim_severity,
            'expected_loss': expected_loss,
            'expenses': expenses,
            'risk_margin': risk_margin,
            'premium': premium,
            'loading_factor': loading_factor
        })
        if len(history) > 5:
            history.pop(0)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Premium components
        components = ['Expected Loss', 'Expenses', 'Risk Margin']
        values = [expected_loss, expenses, risk_margin]
        colors = ['blue', 'orange', 'green']

        bars = ax1.bar(components, values, color=colors, alpha=0.7)
        ax1.set_title('Premium Components')
        ax1.set_ylabel('Amount ($)')
        ax1.grid(axis='y', alpha=0.3)

        # Add a line for total premium
        ax1.axhline(premium, color='red', linestyle='--', label=f'Total Premium: ${premium:.2f}')
        ax1.legend()

        # Add text annotations for each component
        for bar, value, component in zip(bars, values, components):
            percentage = value / premium * 100
            ax1.text(bar.get_x() + bar.get_width() / 2, value / 2,
                     f'${value:.2f}\n({percentage:.1f}%)',
                     ha='center', va='center',
                     color='white' if value > 100 else 'black')

        # Plot 2: Breakdown in pie chart
        ax2.pie(values, labels=components, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Premium Breakdown (Total: ${premium:.2f})')

        # Add a text box explaining the premium formula
        formula_text = f"Premium Calculation:\n\n" \
                       f"• Expected Loss = Frequency × Severity\n" \
                       f"  = {accident_frequency:.1%} × ${claim_severity:,.0f}\n" \
                       f"  = ${expected_loss:.2f}\n\n" \
                       f"• Premium = Expected Loss / (1 - Expense% - Risk%)\n" \
                       f"  = ${expected_loss:.2f} / (1 - {expense_ratio:.0%} - {risk_margin_ratio:.0%})\n" \
                       f"  = ${premium:.2f}\n\n" \
                       f"• Loading Factor = {loading_factor:.2f}"

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.95, 0.05, formula_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        plt.show()

        # Display history table
        print("\nHistory of Recent Simulations:")
        history_table = []
        for i, h in enumerate(history, 1):
            history_table.append({
                'Run': i,
                'Frequency': f"{h['accident_frequency']:.1%}",
                'Severity': f"${h['claim_severity']:,.0f}",
                'Expected Loss': f"${h['expected_loss']:.2f}",
                'Expenses': f"${h['expenses']:.2f}",
                'Risk Margin': f"${h['risk_margin']:.2f}",
                'Premium': f"${h['premium']:.2f}"
            })

        display(pd.DataFrame(history_table).set_index('Run'))

        # Display insurance interpretation
        print("\nInsurance Interpretation:")
        print(f"• Accident Frequency: {accident_frequency:.1%} (probability of claim per year)")
        print(f"• Average Claim Severity: ${claim_severity:,.0f} (average cost when a claim occurs)")
        print(f"• Expected Loss: ${expected_loss:.2f} (pure cost of risk)")
        print(f"• Expenses: ${expenses:.2f} ({expense_ratio:.0%} of premium for administration, commissions, etc.)")
        print(f"• Risk Margin: ${risk_margin:.2f} ({risk_margin_ratio:.0%} of premium for profit and uncertainty)")
        print(f"• Final Premium: ${premium:.2f}")
        print(f"• Loading Factor: {loading_factor:.2f} (Premium ÷ Expected Loss)")
        print("\nThis is the base premium before applying individual rating factors like age, driving history, etc.")

    # Create interactive widgets
    frequency_slider = FloatSlider(
        min=0.01,
        max=0.20,
        step=0.01,
        value=0.05,
        description='Accident Frequency:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    severity_slider = IntSlider(
        min=2000,
        max=20000,
        step=1000,
        value=8000,
        description='Claim Severity ($):',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    interact(update_plot,
             accident_frequency=frequency_slider,
             claim_severity=severity_slider)


# 5. CAPITAL ROLE
def demonstrate_capital_role():
    """
    Demonstrates how capital protects an insurance company from bankruptcy
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


# MAIN FUNCTION TO RUN THE TABBED INTERFACE
def run_insurance_fundamentals():
    """
    Main entry point for the insurance fundamentals demonstrations
    """
    # Create output widget for each tab
    law_of_large_numbers_output = widgets.Output()
    risk_pooling_output = widgets.Output()
    balance_sheet_output = widgets.Output()
    premium_calculation_output = widgets.Output()
    capital_role_output = widgets.Output()

    # Create tabs with proper titles
    tabs = widgets.Tab(children=[
        law_of_large_numbers_output,
        risk_pooling_output,
        balance_sheet_output,
        premium_calculation_output,
        capital_role_output
    ])

    # Set tab titles
    tabs.set_title(0, "1. Law of Large Numbers")
    tabs.set_title(1, "2. Risk Pooling")
    tabs.set_title(2, "3. Balance Sheet")
    tabs.set_title(3, "4. Premium Calculation")
    tabs.set_title(4, "5. Role of Capital")

    # Function to run the appropriate demo when tab is selected
    def on_tab_selected(change):
        if change['new'] == 0:
            with law_of_large_numbers_output:
                law_of_large_numbers_output.clear_output()
                demonstrate_law_of_large_numbers()
        elif change['new'] == 1:
            with risk_pooling_output:
                risk_pooling_output.clear_output()
                demonstrate_risk_pooling()
        elif change['new'] == 2:
            with balance_sheet_output:
                balance_sheet_output.clear_output()
                demonstrate_balance_sheet()
        elif change['new'] == 3:
            with premium_calculation_output:
                premium_calculation_output.clear_output()
                demonstrate_premium_calculation()
        elif change['new'] == 4:
            with capital_role_output:
                capital_role_output.clear_output()
                demonstrate_capital_role()

    # Register the callback
    tabs.observe(on_tab_selected, names='selected_index')

    # Display the tabs
    display(tabs)

    # Initialize first tab
    with law_of_large_numbers_output:
        demonstrate_law_of_large_numbers()