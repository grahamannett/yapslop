from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.status import Status
from rich import box

from rich.text import Text
from rich.panel import Panel
from rich.align import AlignMethod


console = Console()


def print(msg: str, **kwargs):
    console.print(msg, **kwargs)


def info(msg: str, **kwargs):
    print(f"[cyan]Info: {msg}[/cyan]", **kwargs)


def debug(msg: str, **kwargs):
    print(f"[purple]Debug: {msg}[/purple]", **kwargs)


def rule(msg: str, **kwargs):
    console.rule(msg, **kwargs)


def generated_message_panel(
    *renderable: RenderableType,
    title: str = "Message",
    border_style: str = "cyan",
    box: box.Box = box.HEAVY,
    expand: bool = True,
    padding: tuple[int, int] = (1, 1),
    title_align: AlignMethod = "left",
):
    """
    e.g. `generate_message_panel(Text(msg, style="green"))`
    """
    return Panel(
        *renderable,
        title=title,
        border_style=border_style,
        title_align=title_align,
        box=box,
        expand=expand,
        padding=padding,
    )


def model_generating_status(
    status: str,
    spinner: str = "aesthetic",
    speed: float = 0.4,
    refresh_per_second: int = 5,
):
    """
    For use when the model is generating a response.,

    e.g. "Generating audio...
    """
    return Status(status, spinner=spinner, speed=speed, refresh_per_second=refresh_per_second)


class ResponseDisplay:
    def __init__(self, console_: Console | None = None):
        self.console = console_ or Console()

    def show_response(self, *msg: str):
        """
        ResponseDisplay().show_response(generation_stream)
        """
        panels = [generated_message_panel(Text(m, style="green")) for m in msg]

        with Live(console=self.console) as live:
            status = Status("Generating...", spinner="aesthetic", speed=0.4, refresh_per_second=10)
            live.update(status)
            live.update(Group(*panels))
