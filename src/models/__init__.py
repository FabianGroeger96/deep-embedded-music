import glob
import importlib
import os

# get list of files
module_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))

# loop over all module files
for mod in module_files:

    # check if file is a real module and exists
    if os.path.isfile(mod) and not mod.endswith('__init__.py') and not os.path.basename(mod).startswith(("_", ".")):
        # extract module name from file name and import it
        mod_name = os.path.basename(os.path.splitext(mod)[0])
        # import the module
        mod = importlib.import_module(name="src.models.{}".format(mod_name))
