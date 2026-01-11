import abc
from typing import Any


class Tool(abc.ABC):
    """
    Tool base class.
    """

    @staticmethod
    def invoke(*args, **kwargs):
        """ The entry point for the tool (parameters are defined as needed, return the tool execution result)"""
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        """ Return the metadata of the tool (name, description, parameter types, etc.)"""
        raise NotImplementedError