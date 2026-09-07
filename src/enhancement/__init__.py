import importlib
import pkgutil
import os

# Import default packages
from enhancement import baseline as baseline
from enhancement import passthrough as passthrough

# Auto import enhancment options from ECHIPLUGINS
if "ECHIPLUGINS" in os.environ:
    plugin_path = os.environ["ECHIPLUGINS"]
    plugin_root = plugin_path.split("/")[-1]
    for _, module_name, _ in pkgutil.iter_modules([plugin_path]):
        importlib.import_module(f"{plugin_root}.{module_name}")
