import re
import warnings

# is there a way to get this info from the server?
_REQUIRE_PARSERS = [
    "qwq:latest",
    "deepseek-r1:7b",
]


def remove_thinking_block(content: str):
    # if parsing json explicitly, remove thinking tags, should keep this info if not just generating characters
    if all(x in content for x in ["<think>", "</think>"]):
        warnings.warn("Switch to using ReasoningParser")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content


class ReasoningParser:
    """
    Used when models output reasoning blocks, e.g. `<think>yapyap yap</think>model content slop`

    The vLLM parser:
    - https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py

    But only matches on the first block, aso requires tokenizer which I want to avoid
    """

    def __init__(self, tags: tuple[str, str] = ("<think>", "</think>"), re_start: bool = False):
        # if requiring
        self.tags = tags
        self.only_start = r""  # or  r"^"
        start = r"^" if re_start else r""
        self.reasoning_regex = re.compile(rf"{start}{self.tags[0]}(.*?){self.tags[1]}", re.DOTALL)

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)

    def extract_streaming(self, *args, **kwargs):
        raise NotImplementedError("Streaming extraction not implemented")

    def extract(self, text: str):
        reasoning_content = [block.strip() for block in self.reasoning_regex.findall(text)]

        # Remove reasoning blocks from the original text
        content = self.reasoning_regex.sub("", text).strip()

        return reasoning_content, [content] if content else []
