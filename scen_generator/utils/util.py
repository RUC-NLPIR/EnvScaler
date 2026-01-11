from datetime import datetime
from copy import deepcopy

def generate_timestamp() -> str:
    """Generate timestamp string: MMDD-HHMMSS"""
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H%M%S")
    return timestamp


def exec_check_func(check_func, state_diff) -> bool:
    """Execute check_func"""
    state_diff = deepcopy(state_diff)
    namespace = {}
    exec(check_func, namespace)
    try:
        result = namespace["check_state"](state_diff)
    except Exception as e:
        import traceback
        print("Function execution failed:", e)
        # print("Traceback:\n", traceback.format_exc())
        return False
    return result