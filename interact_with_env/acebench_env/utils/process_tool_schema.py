"""
Tool schema processing utilities for OpenAI API JSON Schema format conversion.
"""
def process_tool_schema(func_def: dict) -> dict:
    """
    Convert function definition to OpenAI API JSON Schema format.
    Supports automatic type mapping and recursive normalization.
    """
    # Type mapping table
    type_map = {
        "dict": "object",
        "list": "array",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "boolean": "boolean",  # May already be boolean from JSON
        "str": "string",
    }

    def fix_schema(schema: dict) -> dict:
        """Recursively fix schema types and structure."""
        if not isinstance(schema, dict):
            return schema

        # Fix type
        t = schema.get("type")
        if t in type_map:
            schema["type"] = type_map[t]

        # Recursively process properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            schema["properties"] = {
                k: fix_schema(v) if isinstance(v, dict) else v
                for k, v in schema["properties"].items()
            }

        # Recursively process items (array element definition)
        if "items" in schema:
            schema["items"] = fix_schema(schema["items"]) if isinstance(schema["items"], dict) else schema["items"]

        return schema

    # Deep copy original object (prevent modifying original data)
    import copy
    new_func = copy.deepcopy(func_def)

    # Fix parameters section
    if "parameters" in new_func and isinstance(new_func["parameters"], dict):
        new_func["parameters"] = fix_schema(new_func["parameters"])

    return new_func