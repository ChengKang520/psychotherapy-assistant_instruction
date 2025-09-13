import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from model import ModelServe
from pydantic import BaseModel

app = FastAPI(
    title="Falcon Psychotherapy API",
    description="Run Falcon with API",
    version="0.0.1",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model1 = ModelServe1(load_8bit=False, device_map='auto')
model2 = ModelServe2(load_8bit=False, device_map='auto')

@app.get("/")
async def read_root():
    return {"message": "Alpaca API is ready"}


@app.post("/completion/")
async def completion1(
    instruction: str = Query(description="給模型的角色設定或指令", example="為用戶生成三種可能的投資標的建議", required=True), 
    input: str = Query(description="給模型的 context 或輸入", example="用戶是剛畢業的碩士生", default=None), 
):
    res = model1.generate(instruction=instruction, input=input)
    return res

@app.post("/completion/")
async def completion2(
    instruction: str = Query(description="給模型的角色設定或指令", example="為用戶生成三種可能的投資標的建議", required=True),
    input: str = Query(description="給模型的 context 或輸入", example="用戶是剛畢業的碩士生", default=None),
):
    res = model2.generate(instruction=instruction, input=input)
    return res


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=9889)
    # uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
