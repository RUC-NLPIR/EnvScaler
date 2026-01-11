"""
Extract tool information (name, description, parameters) from environment function details.
"""
def get_tool_info(env_item):
    """Extract tool information (name, description, parameters) from environment function details."""
    tool_info_list = []
    for func_name, func_detail in env_item["env_func_details"].items():
        tool_info = {
                     "name": func_name,
                     "description": func_detail["doc"],
                     "parameters": {param["name"]: param["type"] for param in func_detail["signature"]["parameters"]}
                    }
        tool_info_list.append(tool_info)
    return tool_info_list

def parse_type(type_str: str):
    """Parse Python type annotation string to JSON schema format."""
    type_str = type_str.strip()
    # Handle Optional[T]
    if type_str.startswith("Optional[") and type_str.endswith("]"):
        inner = parse_type(type_str[len("Optional["):-1])
        if "type" in inner:
            if isinstance(inner["type"], list):
                tlist = inner["type"]
            else:
                tlist = [inner["type"]]
            if "null" not in tlist:
                tlist.append("null")
            inner["type"] = tlist
        else:
            inner["type"] = ["null"]
        inner["_optional"] = True  # Custom marker for optional fields
        return inner
    
    # Handle List[T]
    if type_str.startswith("List[") and type_str.endswith("]"):
        inner = parse_type(type_str[len("List["):-1])
        return {"type": "array", "items": inner}
    
    # Handle Dict[...] - simplified as object
    if type_str.startswith("Dict["):
        return {"type": "object"}
    
    # Basic type mappings
    type_map = {
        "str": "string",
        "int": "integer",
        "list": "array",
        "dict": "object",
        "float": "number",
        "bool": "boolean",
        "Any": {},  # Any type
        "object": "object",
    }
    if type_str in type_map:
        t = type_map[type_str]
        return {"type": t} if t else {}
    
    # Unknown custom class name - fallback to object
    print(f"Unknown type: {type_str}")
    return {"type": "object"}  # Fallback


def convert_tool_schema(simple_def):
    """Convert simple tool definition to OpenAI-compatible function calling schema format."""
    props = {}
    required = []
    for pname, tstr in simple_def.get("parameters", {}).items():
        schema = parse_type(tstr)
        
        # Fallback: if array type has no items, add empty object
        if schema.get("type") == "array" and "items" not in schema:
            schema["items"] = {}
            
        # Filter out internal markers (keys starting with "_")
        props[pname] = {k:v for k,v in schema.items() if not k.startswith("_")}
        # Add to required list if not optional
        if not schema.get("_optional"):
            required.append(pname)
            
    return {
        "type": "function",
        "function": {
            "name": simple_def["name"],
            "description": simple_def.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required
            }
        }
    }