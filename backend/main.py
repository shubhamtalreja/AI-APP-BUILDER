import json
import uuid
import os
from fastapi import FastAPI
from pydantic import BaseModel
from llm import intent_chain, app_chain
from fastapi.middleware.cors import CORSMiddleware


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
    # NEW LangChain invocation
    intent_response = await intent_chain.ainvoke({"prompt": data.prompt})
    intent_data = json.loads(intent_response.content)
    intent = intent_data["intent"]

    if intent == "GENERAL_QUERY":
        return {"type": "text", "response": intent_data.get("message", "")}

    if intent == "CLARIFY":
        return {
            "type": "clarify",
            "response": intent_data.get(
                "message",
                "Could you tell me a bit more about what you want?"
            ),
        }


    app_response = await app_chain.ainvoke({"prompt": data.prompt})
    project = json.loads(app_response.content)

    project_id = str(uuid.uuid4())[:8]
    project_path = os.path.join(PROJECT_ROOT, project_id)
    os.makedirs(project_path, exist_ok=True)

    for file, content in project["files"].items():
        full_path = os.path.join(project_path, file)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    return {"response":"App created successfully","type": "app", "projectId": project_id, "files": project["files"]}
