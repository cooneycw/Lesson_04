
# insurance_fundamentals.py module contents here
import importlib.machinery
import os

# Get the current working directory
cwd = os.getcwd()
print(cwd)

# Load the module using the existing loader
module_name = "utils"
file_path = os.path.join(cwd, "utils", f"{module_name}.py")

try:
    loader = importlib.machinery.SourceFileLoader(module_name, file_path)
    module = loader.load_module()

    # Run the insurance fundamentals demo
    run_insurance_fundamentals = getattr(module, "run_insurance_fundamentals")
    run_insurance_fundamentals()
except Exception as e:
    print(f"Error loading module: {e}")