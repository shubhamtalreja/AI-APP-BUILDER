import json
import uuid
import os
from fastapi import FastAPI
from pydantic import BaseModel
from llm import intent_chain, app_chain, chat_chain
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = "../sandbox"
os.makedirs(PROJECT_ROOT, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_app(data: GenerateRequest):
    async def stream_response():
        # Intent Classification
        intent_response = await intent_chain.ainvoke({"prompt": data.prompt})
        intent_data = json.loads(intent_response.content)
        intent = intent_data["intent"]

        yield f"data: {json.dumps({'event': 'intent', 'intent': intent})}\n\n"

        # General Query
        if intent == "GENERAL_QUERY":
            async for chunk in chat_chain.astream(data.prompt):
                if chunk.content:
                    yield f"data: {json.dumps({'event': 'msg', 'content': chunk.content})}\n\n"

            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            return
        
        # Clarification
        if intent == "CLARIFY":

            message = intent_data.get(
                "message",
                "Could you tell me a bit more about what you want?"
            )
            yield f"data: {json.dumps({'event': 'msg', 'message': message})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            return
        
        # App Generation
        if intent == "APP_REQUEST":

            yield f"data: {json.dumps({'event': 'progress', 'message': 'Generating app structure...'})}\n\n"

            app_response = await app_chain.ainvoke({"prompt": data.prompt})
            project = json.loads(app_response.content)

            yield f"data: {json.dumps({'event': 'progress', 'message': 'Writing files...'})}\n\n"

            project_id = str(uuid.uuid4())[:8]
            project_path = os.path.join(PROJECT_ROOT, project_id)
            os.makedirs(project_path, exist_ok=True)

            for file, content in project["files"].items():
                full_path = os.path.join(project_path, file)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

                yield f"data: {json.dumps({'event': 'file_written', 'file': file})}\n\n"
                await asyncio.sleep(0.05)

            yield f"data: {json.dumps({
                'event': 'app',
                'projectId': project_id,
                'files': project['files']
            })}\n\n"

            yield f"data: {json.dumps({'event': 'done'})}\n\n"
    return StreamingResponse(stream_response(), media_type="text/event-stream")
