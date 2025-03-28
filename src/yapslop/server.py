import asyncio
import io
import traceback
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from yapslop.convo_dto import Speaker
from yapslop.providers.yaproviders import ProvidersSetup
from yapslop.yap import ConvoManager, HTTPConfig
from yapslop.yap_common import initial_speakers

# avoid mounting static dir unless i add more html/js
STATIC_DIR = Path(__file__).parent / "static"

# Use a typed dictionary to store app state
app_state: dict[str, Any] = {
    "initial_text": "Did you hear about that new conversational AI model that just came out?",
}


# im using 3.10 for some reason so cant use StrEnum... should see if 3.11 works with the csm repo
# using Enum is too verbose
class MsgTypes:
    initial = "initial:"
    info = "slopinfo:"
    speaking = "yap:"
    complete = "complete:"
    error = "bruh:"


def audio_tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """
    Convert audio tensor to WAV bytes for streaming.

    Args:
        audio_tensor: The audio tensor to convert
        sample_rate: The sample rate of the audio

    Returns:
        Byte representation of the audio in WAV format
    """
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), sample_rate, format="wav")
    buffer.seek(0)
    return buffer.read()


async def generate_conversation_turns(
    convo_manager: ConvoManager,
    initial_text: str,
    initial_speaker: Speaker,
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    buffer_size: int = 2,
):
    """
    Generate conversation turns and put them in the queue.


    Args:
        convo_manager: The conversation manager
        initial_text: Initial prompt for the conversation
        initial_speaker: Initial speaker for the conversation
        audio_queue: Queue to put generated turns into
        stop_event: Event to signal when to stop generation
        buffer_size: Maximum number of turns to buffer
    """
    try:
        async for turn in convo_manager.generate_convo_stream(
            num_turns=-1,  # -1 means infinite
            initial_text=initial_text,
            initial_speaker=initial_speaker,
            save_audio=False,
        ):
            if stop_event.is_set():
                break

            await audio_queue.put(turn)

            # Prevent overwhelming the queue with generated content
            if audio_queue.qsize() >= buffer_size:
                await asyncio.sleep(0.1)

    except Exception as e:
        error_msg = f"Generator error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        await audio_queue.put(error_msg)
    finally:
        # Signal the end of generation
        await audio_queue.put(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.

    Sets up the conversation manager and providers that will be used throughout
    the application's lifetime.

    Args:
        app: The FastAPI application instance
    """

    async with httpx.AsyncClient(base_url=HTTPConfig.base_url) as client:
        text_provider, audio_provider = ProvidersSetup(
            configs={
                "text": {"client": client},
                "audio": {},
            }
        )

        # setup the speakers and convo manager ahead of time since it adds to startup time,
        # later would want this to be args on the creation of the websocket
        convo_manager = ConvoManager(
            n_speakers=2,
            speakers=initial_speakers,
            text_provider=text_provider,
            audio_provider=audio_provider,
            audio_output_dir="audio_output",
        )
        await convo_manager.setup_speakers()
        print(f"Speakers: {[s.name for s in convo_manager.speakers]}")

        app_state["convo_manager"] = convo_manager
        app_state["initial_speaker"] = convo_manager.speakers[0]
        yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/stream")
async def stream_audio(websocket: WebSocket, n_buffer_turns: int = 2):
    """
    Stream generated conversation audio over WebSocket connection.

    Handles the WebSocket connection for streaming AI-generated conversation,
    including both text and audio. The client can provide an initial prompt
    and receives turn-by-turn updates of the conversation.

    Args:
        websocket: The WebSocket connection to the client
        n_buffer_turns: Number of turns to buffer ahead
    """
    # Setup communication mechanisms

    # Get state from app

    convo_manager: ConvoManager = app_state["convo_manager"]
    initial_speaker: Speaker = app_state["initial_speaker"]
    msg: str = app_state["initial_text"]
    sample_rate: int = convo_manager.audio_provider.sample_rate
    buffer = []

    # Accept the WebSocket connection
    await websocket.accept()

    try:
        # Wait for the initial message from the client

        if (initial_msg := await websocket.receive_text()).startswith(MsgTypes.initial):
            msg = initial_msg.lstrip(MsgTypes.initial)

        await websocket.send_text(f"{MsgTypes.info} Generating {n_buffer_turns} buffer turns...")

        convo_stream = convo_manager.generate_convo_stream(
            num_turns=-1,  # -1 means infinite
            initial_text=msg,
            initial_speaker=initial_speaker,
            save_audio=False,
        )

        # generate the buffer
        print("Generating buffer")
        async for turn in convo_stream:
            buffer.append(turn)
            if len(buffer) >= n_buffer_turns:
                break

        # send the buffer
        print("Sending buffer")
        for turn in buffer:
            await websocket.send_text(f"{MsgTypes.speaking}{turn.speaker.name}:{turn.text}")
            audio_bytes = audio_tensor_to_wav_bytes(turn.audio, sample_rate)
            await websocket.send_bytes(audio_bytes)
            # Wait for client to confirm receipt before continuing
            msg = await websocket.receive_text()
            if msg == "stop":
                print("Client requested to stop")
                break
            elif msg != "next":
                print(f"Unexpected message from client: {msg}")

        print("Buffer sent")
        async for turn in convo_stream:
            try:
                await websocket.send_text(f"{MsgTypes.speaking}{turn.speaker.name}:{turn.text}")

                if turn.audio is not None:
                    audio_bytes = audio_tensor_to_wav_bytes(turn.audio, sample_rate)
                    await websocket.send_bytes(audio_bytes)
                    # Wait for client to confirm receipt before continuing
                    msg = await websocket.receive_text()
                    if msg == "stop":
                        print("Client requested to stop")
                        break
                    elif msg != "next":
                        print(f"Unexpected message from client: {msg}")

            except Exception as e:
                print(f"Error streaming turn: {str(e)}")
                traceback.print_exc()
                await websocket.send_text(f"{MsgTypes.error}{str(e)}")
                await asyncio.sleep(0.1)

        await websocket.send_text(f"{MsgTypes.complete}Conversation finished")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        await websocket.send_text(f"{MsgTypes.error}{str(e)}")


@app.get("/")
async def serve_index():
    """
    Serve the main index.html page for the web interface.

    Returns:
        The index.html file from the static directory
    """
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
