from ipywidgets import widgets
from IPython.display import display
import os
import importlib.machinery


def load_module(file_path):
    """Load a Python module from a file path."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    loader = importlib.machinery.SourceFileLoader(module_name, file_path)
    return loader.load_module()


def run_insurance_fundamentals():
    """
    Main entry point for the insurance fundamentals demonstrations
    """
    # Get the current working directory
    cwd = os.getcwd()

    # Load each demonstration module directly
    law_module_path = os.path.join(cwd, "utils", "law_of_large_numbers.py")
    risk_module_path = os.path.join(cwd, "utils", "risk_pooling.py")
    balance_module_path = os.path.join(cwd, "utils", "balance_sheet.py")
    premium_module_path = os.path.join(cwd, "utils", "premium_calculation.py")
    capital_module_path = os.path.join(cwd, "utils", "capital_role.py")

    # Load the modules
    law_module = load_module(law_module_path)
    risk_module = load_module(risk_module_path)
    balance_module = load_module(balance_module_path)
    premium_module = load_module(premium_module_path)
    capital_module = load_module(capital_module_path)

    # Get the demonstration functions
    demonstrate_law_of_large_numbers = getattr(law_module, "demonstrate_law_of_large_numbers")
    demonstrate_risk_pooling = getattr(risk_module, "demonstrate_risk_pooling")
    demonstrate_balance_sheet = getattr(balance_module, "demonstrate_balance_sheet")
    demonstrate_premium_calculation = getattr(premium_module, "demonstrate_premium_calculation")
    demonstrate_capital_role = getattr(capital_module, "demonstrate_capital_role")

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
    tabs.set_title(0, "1. Law of Lrg No's")
    tabs.set_title(1, "2. Risk Pooling")
    tabs.set_title(2, "3. Balance Sheet")
    tabs.set_title(3, "4. Premium Calc")
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