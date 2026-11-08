from typing import Any

DataFrame = Any
Series = Any

def __getattr__(name: str) -> Any: ...
