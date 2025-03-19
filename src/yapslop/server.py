import asyncio
import io
import traceback
from contextlib import asynccontextmanager

import httpx
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from yapslop.yap import AudioProvider, ConvoManager, Speaker, TextProvider

app_lifespan = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(base_url="http://localhost:11434") as client:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_provider = AudioProvider(device=device)
        text_provider = TextProvider(client=client, model_name="gemma3:latest")

        speakers = [
            Speaker(
                name="Seraphina",
                description="Tech entrepreneur. Uses technical jargon, speaks confidently",
            ),
        ]

        convo_manager = ConvoManager(
            n_speakers=2,
            speakers=speakers,
            text_provider=text_provider,
            audio_provider=audio_provider,
            audio_output_dir="audio_output",
        )

        await convo_manager.setup_speakers()

        initial_speaker = convo_manager.speakers[0]
        app_lifespan["convo_manager"] = convo_manager
        app_lifespan["initial_speaker"] = initial_speaker
        yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


async def audio_tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to WAV bytes for streaming"""
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), sample_rate, format="wav")
    buffer.seek(0)
    return buffer.read()


@app.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    """Stream generated conversation audio over WebSocket connection."""
    await websocket.accept()

    try:
        convo_manager = app_lifespan["convo_manager"]
        initial_speaker = app_lifespan["initial_speaker"]

        await websocket.send_text("Starting conversation stream...")

        # Start conversation with initial phrase
        initial_phrase = "Did you hear about that new conversational AI model that just came out?"

        async for turn in convo_manager.generate_convo_text_stream(
            num_turns=-1,
            initial_phrase=initial_phrase,
            initial_speaker=initial_speaker,
            save_audio=False,
        ):
            try:
                # Send the speaker and text first
                print(f"speaking:{turn.speaker.name}:{turn.text}")
                await websocket.send_text(f"speaking:{turn.speaker.name}:{turn.text}")

                # Convert audio tensor to WAV bytes and stream
                if turn.audio is not None:
                    audio_bytes = await audio_tensor_to_wav_bytes(
                        turn.audio, convo_manager.audio_provider.sample_rate
                    )
                    await websocket.send_bytes(audio_bytes)

                    # Wait for client to confirm receipt before continuing
                    msg = await websocket.receive_text()
                    if msg != "next":
                        print(f"Unexpected message from client: {msg}")
                        if msg == "stop":
                            break

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
        except:
            pass


@app.get("/")
async def serve_index():
    return FileResponse("src/yapslop/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
