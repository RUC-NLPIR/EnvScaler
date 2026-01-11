"""
step5: 
Assemble environment description and operation code into complete Python environment class code.
Performs AST syntax validation and return statement checks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import ast
from typing import List, Optional
from copy import deepcopy

from utils.process_file import read_file, save_file


def assemble_env_class(class_def: str, methods: List[str]) -> str:
    """
    Assemble a class definition string and a list of method definitions into one complete Python class code.
    
    Args:
        class_def (str): Python code string defining the class and __init__.
        methods (List[str]): A list of Python method code strings (should start with "def ...").
        output_path (Optional[str]): If given, the result will also be written to this file.
    
    Returns:
        str: The assembled full Python code.
    """
    lines = class_def.splitlines()
    
    # Find insertion point (after __init__ method)
    insertion_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def __init__"):
            insertion_index = i
    if insertion_index is None:
        raise ValueError("No __init__ method found in class_def")

    # Determine indentation level (usually 4 spaces inside class)
    base_indent = " " * 4
    
    # Prepare indented methods
    indented_methods = []
    for method in methods:
        method_lines = method.strip().splitlines()
        indented_methods.extend([base_indent + l if l.strip() else l for l in method_lines])
        indented_methods.append("")  # blank line between methods
    
    # Find end of __init__ method (detect next method definition or end of class)
    end_init_idx = insertion_index
    for i in range(insertion_index + 1, len(lines)):
        if lines[i].startswith("    def ") and not lines[i].lstrip().startswith("__init__"):
            end_init_idx = i - 1
            break
    else:
        end_init_idx = len(lines) - 1

    assembled = []
    assembled.extend(lines[:end_init_idx+1])
    assembled.append("")  # ensure one blank line
    assembled.extend(indented_methods)
    assembled.extend(lines[end_init_idx+1:])

    result = "\n".join(assembled)

    return result



def process_import(result: str) -> str:
    """
    Post-process assembled Python code:
    - Extract all import statements (both `import ...` and `from ... import ...`),
      even if they appear indented inside class/methods.
    - Move them to the file top.
    - Deduplicate and sort.
    - Remove duplicates from original places.
    
    Args:
        result (str): Python code string.
    
    Returns:
        str: Cleaned Python code with all imports at the top.
    """
    lines = result.splitlines()

    import_lines = []
    cleaned_lines = []

    # Regex patterns for import detection
    import_pattern = re.compile(r'^\s*(import .+|from .+ import .+)')

    for line in lines:
        if import_pattern.match(line.strip()):
            # Normalize by stripping leading/trailing spaces
            stmt = line.strip()
            import_lines.append(stmt)
            # remove from inline position
            continue
        cleaned_lines.append(line)

    # Deduplicate + preserve relative order (use dict.fromkeys)
    unique_imports = list(dict.fromkeys(import_lines))

    # Compose new file content
    final_lines = []
    if unique_imports:
        final_lines.extend(unique_imports)
        final_lines.append("")  # blank line after imports

    # Add back the rest (cleaned)
    final_lines.extend(cleaned_lines)

    return "\n".join(final_lines).strip() + "\n"


def check_ast(source: str) -> bool:
    """
    Check whether a Python source code string is syntactically valid.
    
    Args:
        source (str): Python source code as a string.
    
    Returns:
        bool: True if syntax is valid, False otherwise.
    
    Raises:
        SyntaxError: with detailed info if invalid.
    """
    try:
        ast.parse(source)
        return True
    except SyntaxError as e:
        print("❌ Syntax error detected:")
        print(f"  Line {e.lineno}, Offset {e.offset}: {e.text.strip() if e.text else ''}")
        print(f"  Details: {e.msg}")
        return False

def check_returns(source: str, strict: bool = False) -> list:
    """
    Check all function return statements for dictionary style output.
    
    Args:
        source (str): Python source code string.
        strict (bool): 
            If True, enforce that return dict must contain "success" and at least one of ("data","message","error").
    
    Returns:
        list of (function_name, lineno, status, message)
    """
    results = []
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    ret = child.value
                    # Check return dictionary structure
                    if isinstance(ret, ast.Dict):
                        keys = []
                        for k in ret.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                keys.append(k.value)
                        if "success" not in keys:
                            results.append((func_name, child.lineno, "FAIL", "Missing 'success' key in return dict"))
                        elif strict and not (("data" in keys) or ("message" in keys) or ("error" in keys)):
                            results.append((func_name, child.lineno, "WARN", "Return dict has 'success' but missing data/message/error"))
                        else:
                            results.append((func_name, child.lineno, "PASS", "Return dict keys: " + ",".join(keys)))
                    elif isinstance(ret, ast.Name):
                        # Variable return - cannot check statically
                        results.append((func_name, child.lineno, "WARN", f"Returns variable '{ret.id}', cannot check keys statically"))
                    elif ret is None:
                        results.append((func_name, child.lineno, "FAIL", "Return None detected"))
                    else:
                        # Other return types
                        results.append((func_name, child.lineno, "WARN", f"Return type {type(ret).__name__}, not directly checkable"))
    return results


def write_py_code(py_code: str, output_path: str):
    """Write Python code to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(py_code)

def process_env_item(env_item):
    """Assemble class code from definition and methods, validate syntax and returns."""
    class_def = env_item["class_definition"]
    methods = [operation["code"] for operation in env_item["operation_list"]]
    passed = True
    try:
        py_code = assemble_env_class(class_def, methods)
        py_code = process_import(py_code)
    except Exception as e:
        print(f"❌ Error assembling class: {e}")
        passed = False
    if check_ast(py_code):
        print("✅ Syntax is valid")
    else:
        print("❌ Syntax is invalid")
        passed = False
    # Check return statements for proper dictionary structure
    if check_returns(py_code):
        print("✅ Returns are valid")
    else:
        print("❌ Returns are invalid")
        passed = False
    new_item = deepcopy(env_item)
    new_item["env_class_code"] = py_code
    return passed, new_item
        
        
def main(read_file_path, save_file_path):
    """Process all environment items, assemble code, validate, and save valid ones."""
    raw_data = read_file(read_file_path)
    new_data = []
    for env_item in raw_data:
        print(f"Processing env item: {env_item['environment_summary']}")
        try:
            passed, env_item = process_env_item(env_item)
        except Exception as e:
            print(f"❌ Error processing env item: {e}")
            continue
        # Only keep items that passed all validations
        if passed:
            new_data.append(env_item)
    save_file(save_file_path, new_data)



if __name__ == "__main__":
    read_file_path = "stage2_syn_env/temp_result/step4_infer_func_code.json"
    save_file_path = "stage2_syn_env/temp_result/step5_env_class_code.json"
    main(read_file_path,save_file_path)
