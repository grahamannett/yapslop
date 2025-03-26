import asyncio
import io
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from yapslop.yap import (
    ConvoManager,
    HTTPConfig,
    ProvidersSetup,
)
from yapslop.convo_dto import Speaker
from yapslop.yap_common import initial_speakers

# avoid mounting static dir unless i add more html/js
STATIC_DIR = Path(__file__).parent / "static"

# Use a typed dictionary to store app state
app_state: dict[str, Any] = {
    "initial_text": "Did you hear about that new conversational AI model that just came out?",
    "initial_speakers": [Speaker(**s) for s in initial_speakers],
}


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

        convo_manager = ConvoManager(
            n_speakers=2,
            speakers=app_state["initial_speakers"],
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
    """

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

    await websocket.accept()

    try:
        msg = app_state["initial_text"]
        convo_manager: ConvoManager = app_state["convo_manager"]
        initial_speaker: Speaker = app_state["initial_speaker"]

        # Wait for the initial message from the client
        if (_msg := await websocket.receive_text()).startswith("initial:"):
            msg = _msg[8:].strip()

        await websocket.send_text("Starting conversation stream...")
        await websocket.send_text(f"slopinfo: Generating {n_buffer_turns} buffer turns...")

        async for turn in convo_manager.generate_convo_stream(
            num_turns=-1,  # -1 means infinite
            initial_text=msg,
            initial_speaker=initial_speaker,
            save_audio=False,
        ):
            try:
                await websocket.send_text(f"speaking:{turn.speaker.name}:{turn.text}")

                # Convert audio tensor to WAV bytes and stream
                if turn.audio is not None:
                    audio_bytes = audio_tensor_to_wav_bytes(
                        turn.audio,
                        convo_manager.audio_provider.sample_rate,
                    )
                    await websocket.send_bytes(audio_bytes)

                    # Wait for client to confirm receipt before continuing
                    msg = await websocket.receive_text()
                    if msg == "stop":
                        break
                    elif msg != "next":
                        print(f"Unexpected message from client: {msg}")

            except Exception as e:
                error_msg = f"Error streaming turn: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                await websocket.send_text(f"error:{error_msg}")
                await asyncio.sleep(1)

        await websocket.send_text("complete:Conversation finished")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        try:
            await websocket.send_text(f"error:{error_msg}")
        except Exception:
            # Use a bare except only for this final error handling attempt
            pass


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
