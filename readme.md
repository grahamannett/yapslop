# YapSlop

## The slop must flow.

Generate a continuous stream of conversational audio using CSM 1B model: [CSM](https://github.com/SesameAILabs/csm) and another model (via ollama).

This repo contains both a CLI way to generate yap and a server that allows you to continuously generate a stream of slop.

# Setup

1. Clone this repository:
```bash
git clone https://github.com/grahamannett/yapslop.git
cd yapslop
# if subtree needed:
git subtree add --prefix=packages/csm https://github.com/SesameAILabs/csm.git main --squash
```


2. Make editable install and install dependencies using `uv`:
```bash
uv pip install -e .
uv pip install -r csm/requirements.txt
```

3. Additionally need `ffmpeg` installed and local files from `csm` as importable (if this is possible from pyproject.toml using uv, I am unaware of how. instead i am using mise and setting PYTHONPATH to the csm directory such that there is ideally no changes to the original csm repo).


## Server Usage

To run a server that generates a continuous stream of slop, run:

```bash
uv run python yapslop/server.py
```

# Notes

- Originally thought this could be <500 LoC and one file, but doesnt seem like that would be a clean solution so will be splitting out the `yap` into more files.
- Right now, using websockets for streaming the audio, had tested both websocket and `StreamingResponse` from fastapi and like the `StreamingResponse` way better but seemed less consistent in various ways.  Will reincorporate that for later.

