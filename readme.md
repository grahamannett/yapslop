# YapSlop

A simple wrapper around [CSM](https://github.com/facebookresearch/csm) to make it easier to use.

## Setup

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

3. Additionally need `ffmpeg` installed and local files from `csm` as importable (if this is possible from pyproject.toml using uv please let me know, i am using mise with PYTHONPATH for the time being)


## Server Usage

To run a server that generates a continuous stream of slop, run:

```bash
uv run python src/yapslop/server.py
```

# Notes

- Originally thought this could be <500 LoC and one file, but doesnt seem like that would be a clean solution so will be splitting out the `yap` into more files.
- Right now, using websockets for streaming the audio, had tested both websocket and `StreamingResponse` from fastapi and like the `StreamingResponse` way better but seemed less consistent in various ways.  Will reincorporate that for later.

