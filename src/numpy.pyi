from typing import Any

ndarray = Any
uint8: Any
float32: Any
float64: Any

def __getattr__(name: str) -> Any: ...
