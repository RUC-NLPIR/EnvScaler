"""
File I/O utilities with support for special Python types in JSON serialization.
"""
import json
import datetime
import decimal
import jsonpickle
import os


class UnsupportedType(Exception):
    pass


def convert_for_save(obj):
    """Recursively convert Python special types to JSON-serializable structures."""
    if isinstance(obj, set):
        return {"__type__": "set", "items": [convert_for_save(i) for i in obj]}
    elif isinstance(obj, tuple):
        return {"__type__": "tuple", "items": [convert_for_save(i) for i in obj]}
    elif isinstance(obj, datetime.datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    elif isinstance(obj, decimal.Decimal):
        return {"__type__": "decimal", "value": str(obj)}
    elif isinstance(obj, dict):
        # JSON-supported key types
        ALLOWED_KEY_TYPES = (str, int, float, bool, type(None))
        if all(isinstance(k, ALLOWED_KEY_TYPES) for k in obj.keys()):
            # Normal keys -> save directly, but values may need recursive processing
            return {k: convert_for_save(v) for k, v in obj.items()}
        else:
            # Unsupported key types (e.g., tuple) -> save with type markers
            items = []
            for k, v in obj.items():
                if not isinstance(k, ALLOWED_KEY_TYPES):
                    k_conv = {"__type__": "key", "value": convert_for_save(k)}
                else:
                    k_conv = k
                items.append((k_conv, convert_for_save(v)))
            return {"__type__": "dict", "items": items}
    elif isinstance(obj, list):
        return [convert_for_save(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Unknown type -> raise exception
        raise UnsupportedType(f"Unsupported type: {type(obj).__name__}")


def restore_after_load(obj):
    """Recursively restore structures with type markers to original Python types."""
    if isinstance(obj, dict) and "__type__" in obj:
        t = obj["__type__"]
        if t == "set":
            return set(restore_after_load(i) for i in obj["items"])
        elif t == "tuple":
            return tuple(restore_after_load(i) for i in obj["items"])
        elif t == "datetime":
            return datetime.datetime.fromisoformat(obj["value"])
        elif t == "decimal":
            return decimal.Decimal(obj["value"])
        elif t == "key":  # Dict key marker
            return restore_after_load(obj["value"])
        elif t == "dict":  # Dict structure with special keys
            restored = {}
            for k, v in obj["items"]:
                if isinstance(k, dict) and "__type__" in k and k["__type__"] == "key":
                    k_restored = restore_after_load(k["value"])
                else:
                    k_restored = k
                restored[k_restored] = restore_after_load(v)
            return restored
        else:
            return obj
    elif isinstance(obj, dict):
        # Normal dict -> recursively process values
        return {k: restore_after_load(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [restore_after_load(i) for i in obj]
    else:
        return obj


def save_file(file_path, data):
    """Save data to JSON/TXT file. Prefer standard JSON, fallback to jsonpickle."""
    # Create directory if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith('.json'):
        try:
            json_obj = convert_for_save(data)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
        except UnsupportedType as e:
            print(f"[Save Warning] Unsupported type detected {e}, using jsonpickle for full save")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(data, indent=2))
    elif file_path.endswith('.txt'):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def read_file(file_path):
    """Read JSON/TXT file. Auto-detect standard JSON vs jsonpickle format."""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # If contains jsonpickle markers, use jsonpickle directly
        if any(tag in content for tag in ('"py/object"', '"py/tuple"', '"py/set"')):
            return jsonpickle.decode(content)

        # Otherwise parse as standard JSON + restore types
        data = json.loads(content)
        return restore_after_load(data)

    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")