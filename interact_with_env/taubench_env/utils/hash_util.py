"""
Utility functions for hashing data.
"""
from hashlib import sha256
from typing import Dict, List, Set, Union, Tuple

# To allow hashing of arbitrary data structures, define type aliases here
ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]

def to_hashable(item: ToHashable) -> Hashable:
    """Convert arbitrary nested list/dict/set to hashable tuple structure (for consistent hashing)"""
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(value: Hashable) -> str:
    """Calculate a SHA-256 digest for any hashable value (return as 64-bit hexadecimal string) for "content fingerprint" comparison."""
    return sha256(str(value).encode("utf-8")).hexdigest()



def get_data_hash(data) -> str:
    """Return the consistent hash of the current environment data"""
    return consistent_hash(to_hashable(data))



