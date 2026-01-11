"""
Utility functions for environment initialization and state management.
"""
import types
from copy import deepcopy

def init_env_class(env_class_code: str, env_class_name: str):
    """
    Initialize an environment class from source code string.
    
    :param env_class_code: Python source code string of the environment class
    :param env_class_name: Name of the class (must be defined in the code)
    :return: Environment class object
    """
    module = types.ModuleType("dynamic_env")
    exec(env_class_code, module.__dict__)
    
    if not hasattr(module, env_class_name):
        raise ValueError(f"Class '{env_class_name}' not found in provided env_class_code.")
    
    return getattr(module, env_class_name)


def init_env_instance(env_class, init_config=None):
    """
    Create an environment instance from class and apply initial config.
    
    :param env_class: Environment class object
    :param init_config: Optional dict for initial attribute configuration
    :return: Environment instance object
    """
    init_config = deepcopy(init_config)
    try:
        # Try constructor with dict argument
        if init_config and isinstance(init_config, dict):
            env_instance = env_class(init_config)
        else:
            env_instance = env_class({})
    except TypeError:
        # Constructor with no arguments
        env_instance = env_class()
    
    if init_config:
        for key, value in init_config.items():
            setattr(env_instance, key, value)
    
    return env_instance



def get_state_diff(old_state: dict, new_state: dict, ignore_keys: list = []) -> dict:
    """
    Compare two state dictionaries and return differences:
    - Added keys
    - Removed keys
    - Changed values
    
    Recursively compares dict values.
    """
    old_state = deepcopy(old_state)
    new_state = deepcopy(new_state)
    diff_result = {}

    # Find union of all keys
    all_keys = set(old_state.keys()) | set(new_state.keys())

    for key in all_keys:
        old_val = old_state.get(key)
        new_val = new_state.get(key)

        if key not in old_state: # Added key
            diff_result[key] = {"added": new_val}
        elif key not in new_state: # Removed key
            diff_result[key] = {"removed": old_val}
        else: # Both exist, compare values
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Recursive comparison for dicts
                sub_diff = get_state_diff(old_val, new_val)
                if sub_diff:  # Record only if there are changes
                    diff_result[key] = sub_diff
            else:
                # Simple type comparison
                if old_val != new_val:
                    diff_result[key] = {"changed": {"old":old_val, "new":new_val}}
                    
    # Remove ignored keys
    for key in ignore_keys:
        if key in diff_result:
            del diff_result[key]

    return deepcopy(diff_result)


def get_state_info(env_instance):
    """Return state dictionary of environment instance (excluding built-in attributes)."""
    return deepcopy({
        k: v for k, v in vars(env_instance).items()
        if not (k.startswith("__") and k.endswith("__"))
    })


def run_check_function(func_code: str, init_state: dict, final_state: dict):
    """
    Dynamically execute a verification function defined in func_code.
    """
    safe_globals = {
        '__builtins__': __builtins__,
    }
    safe_globals.update({"initial_state": deepcopy(init_state)})

    try:
        # Execute in safe_globals, function will retain this global scope
        exec(func_code, safe_globals)

        if 'check_func' not in safe_globals:
            return False, None, "Function 'check_func' not found."

        result = safe_globals['check_func'](final_state)

        if not isinstance(result, bool):
            print("Function did not return a boolean. Result: {result}")
            return False, None, "Function did not return a boolean."

        return True, result, None
    except Exception as e:
        print("Error:", e)
        return False, None, str(e)