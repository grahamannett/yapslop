from typing import Callable, Type

from pydantic import TypeAdapter

"""
Was aiming to avoid using other frameworks as they all have so many bugs/issues/etc

Options are using pydantic/openai-agent/similar, pydantic is the most lightweight for now

Pydantic option:
- https://docs.pydantic.dev/latest/concepts/types/#typeadapter

"""


def get_type_schema(cls: Type):
    return TypeAdapter(cls).json_schema()


def _agents_get_schema(func: Callable) -> dict[str, str | dict]:
    """
    may be worthwhile trying out openai-agents, and would be done like this
    """
    from agents import function_tool

    return function_tool(func).params_json_schema
