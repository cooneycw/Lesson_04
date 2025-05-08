from ipywidgets import widgets
from IPython.display import display

# Import the demonstration functions
from utils.law_of_large_numbers import demonstrate_law_of_large_numbers
from utils.risk_pooling import demonstrate_risk_pooling
from utils.balance_sheet import demonstrate_balance_sheet
from utils.premium_calculation import demonstrate_premium_calculation
from utils.capital_role import demonstrate_capital_role

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