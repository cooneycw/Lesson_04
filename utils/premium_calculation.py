# premium_calculation.py - Demonstrating how insurance premiums are calculated

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, IntSlider, interact
import pandas as pd


def demonstrate_premium_calculation():
    """
    Demonstrates how insurance premiums are calculated

    Shows the components that make up an insurance premium:
    - Expected Loss (Frequency × Severity)
    - Expenses
    - Risk Margin
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

        # Check our math
        premium_check = expected_loss + expenses + risk_margin

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

