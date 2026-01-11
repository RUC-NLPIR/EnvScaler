"""
Parse environment class name and definition
"""
import re

def parse_env_class_name(str):
    """Parse environment class name"""
    # last class
    content = str.split("def __init__")[0]
    pattern = r"class\s+(\w+)\s*:"
    matches = re.findall(pattern, content)
    if matches:
        return matches[-1].strip()
    else:
        return str.split("def __init__")[0].split("class ")[-1].split(":")[0].strip()


def parse_env_class_def(src_str: str, class_name: str) -> dict:
    """Parse environment class definition"""
    part1=src_str.split(f"class {class_name}")[0]
    part2=src_str.split(f"class {class_name}")[1].split("def __init__")[0]
    part3=src_str.split(f"class {class_name}")[1].split("def __init__")[1].split("def ")[0]
    content=part1+f"class {class_name}"+part2+"def __init__"+part3
    return content



