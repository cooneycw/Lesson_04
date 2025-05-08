
# balance_sheet.py module contents here
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import FloatSlider, interact
from IPython.display import display

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