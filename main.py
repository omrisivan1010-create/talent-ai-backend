from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os

app = FastAPI(title="AI Candidate Search API", version="1.0.0")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class CandidateExtraction(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    current_title: Optional[str] = None
    experience_years: Optional[float] = None
    skills: List[str] = []
    military_or_security_background: Optional[str] = None
    seniority_level: Optional[str] = None

class CandidatePostRequest(BaseModel):
    raw_text: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/candidates")
def create_candidate(request: CandidatePostRequest):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "אתה עוזר חכם למערכת גיוס. תפקידך לחלץ מתוך טקסט גולמי שנאסף מהרשת על מועמד את כל הפרטים הנדרשים בצורה מדויקת."
                },
                {"role": "user", "content": request.raw_text},
            ],
            response_format=CandidateExtraction,
        )
        
        extracted_data = completion.choices[0].message.parsed
        
        return {
            "message": "Candidate parsed successfully",
            "data": extracted_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
