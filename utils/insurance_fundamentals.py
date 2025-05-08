# insurance_fundamentals.py - Main script for insurance fundamentals demonstrations

import ipywidgets as widgets
from ipywidgets import VBox, HBox

# Import the demonstration modules
from law_of_large_numbers import run_law_of_large_numbers_demo
from risk_pooling import run_risk_pooling_demo
from balance_sheet import run_balance_sheet_demo
from premium_calculation import run_premium_calculation_demo
from capital_role import run_capital_role_demo


def run_insurance_fundamentals():
    """
    Main entry point for the insurance fundamentals demonstrations

    Creates a tab interface to navigate between the different demonstration modules
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
                run_law_of_large_numbers_demo()
        elif change['new'] == 1:
            with risk_pooling_output:
                risk_pooling_output.clear_output()
                run_risk_pooling_demo()
        elif change['new'] == 2:
            with balance_sheet_output:
                balance_sheet_output.clear_output()
                run_balance_sheet_demo()
        elif change['new'] == 3:
            with premium_calculation_output:
                premium_calculation_output.clear_output()
                run_premium_calculation_demo()
        elif change['new'] == 4:
            with capital_role_output:
                capital_role_output.clear_output()
                run_capital_role_demo()

    # Register the callback
    tabs.observe(on_tab_selected, names='selected_index')

    # Display the tabs
    display(tabs)

    # Initialize first tab
    with law_of_large_numbers_output:
        run_law_of_large_numbers_demo()

