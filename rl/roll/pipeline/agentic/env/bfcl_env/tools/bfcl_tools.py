from typing import List, Dict
import json
from pathlib import Path


INVOLVED_CLASS_TO_FUNC_DOC_PATH = {
    "GorillaFileSystem": str(Path(__file__).parent / "bfcl_tools" / "gorilla_file_system.jsonl"),
    "MathAPI": str(Path(__file__).parent / "bfcl_tools" / "math_api.jsonl"),
    "MessageAPI": str(Path(__file__).parent / "bfcl_tools" / "message_api.jsonl"),
    "TwitterAPI": str(Path(__file__).parent / "bfcl_tools" / "posting_api.jsonl"),
    "TicketAPI": str(Path(__file__).parent / "bfcl_tools" / "ticket_api.jsonl"),
    "TradingBot": str(Path(__file__).parent / "bfcl_tools" / "trading_bot.jsonl"),
    "TravelAPI": str(Path(__file__).parent / "bfcl_tools" / "travel_booking.jsonl"),
    "VehicleControlAPI": str(Path(__file__).parent / "bfcl_tools" / "vehicle_control.jsonl"),
}

def construct_tools_from_involved_classes(involved_classes: List[str]) -> str:
    tools = []
    for class_name in involved_classes:
        with open(INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name], "r") as f:
            func_doc = json.load(f)
        for func in func_doc:
            func["description"] = func["description"].split("Tool description: ")[1]
        func_doc = [json.dumps(func) for func in func_doc]
        tools.extend(func_doc)
    return "\n".join(tools)

def mean(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate the mean of a list of numbers.

    Args:
        numbers (List[float]): List of numbers to calculate the mean of.

    Returns:
        result (float): Mean of the numbers.
    """
    if not numbers:
        return {"error": "Cannot calculate mean of an empty list"}
    try:
        return {"result": sum(numbers) / len(numbers)}
    except TypeError:
        return {"error": "All elements in the list must be numbers"}