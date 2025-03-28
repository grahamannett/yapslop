import json
from functools import partial

import pytest

from yapslop.providers.parsers import ReasoningParser

pytestmark = pytest.mark.anyio


class Block:
    @staticmethod
    def tag(text: str, tag: str) -> str:
        return f"<{tag}>{text}</{tag}>"

    think = partial(tag, tag="think")

    @staticmethod
    def fence(text: str, lang: str) -> str:
        return f"```{lang}\n{text}\n```"

    json = partial(fence, lang="json")


@pytest.fixture
def json_content() -> str:
    """Generate sample JSON content for testing."""
    return json.dumps({"name": "testuser", "description": "testdesc"})


@pytest.mark.parametrize(
    "input_text,expected_reasoning,expected_content",
    [
        # Single thinking block
        (
            f"{Block.think('First block')}\n\ncontent",
            ["First block"],
            ["content"],
        ),
        # No thinking blocks
        (
            "just content",
            [],
            ["just content"],
        ),
        # Multiple thinking blocks is tested in test_multiple_thinking_blocks
    ],
)
async def test_reasoning_parser(
    input_text: str,
    expected_reasoning: list[str],
    expected_content: list[str],
) -> None:
    """Test the reasoning parser with simple test cases."""
    parser = ReasoningParser()
    reasoning_content, content = parser.extract(input_text)

    assert reasoning_content == expected_reasoning
    assert content == expected_content


async def test_multiple_thinking_blocks():
    """
    Test multiple thinking blocks.

    Note: By default, the parser only matches the first block starting at the beginning
    of the string. To match multiple blocks, we need to initialize it with re_start=True.
    """
    # Configure parser to match multiple blocks
    parser = ReasoningParser()
    input_text = f"{Block.think('First block')}\n\n{Block.think('Second block')}content"

    reasoning_content, content = parser.extract(input_text)
    assert len(reasoning_content) == 2
    assert "First block" in reasoning_content
    assert "Second block" in reasoning_content
    assert content == ["content"]

    # Default behavior - only matches first block
    default_parser = ReasoningParser(re_start=True)
    reasoning_content, content = default_parser.extract(input_text)
    assert len(reasoning_content) == 1
    assert reasoning_content == ["First block"]
    assert "content" in content[0]
    assert "<think>Second block</think>" in content[0]
