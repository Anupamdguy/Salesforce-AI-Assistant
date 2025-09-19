
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline

app = FastAPI()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class SalesforceRecord(BaseModel):
    id: Optional[str] = None
    text: str

class SalesforceData(BaseModel):
    records: List[SalesforceRecord]

@app.post("/salesforce-summarize")
async def salesforce_summarize(data: SalesforceData):
    summaries = []
    for record in data.records:
        summary = summarizer(record.text, max_length=20, min_length=10, do_sample=False)
        summaries.append({
            "id": record.id,
            "summary": summary[0]["summary_text"]
        })
    return {"summaries": summaries}

@app.get("/")
async def root():
    return {
        "message": "Send POST to /salesforce-summarize with JSON: { 'records': [ { 'id': '...', 'text': '...' }, ... ] } }"
    }
