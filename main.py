from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os

app = FastAPI(title="AI Candidate Search API", version="1.0.0")

# אתחול הלקוח של OpenAI (מוודא שלוקח את המפתח מתוך משתני הסביבה ב-Render)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# הגדרת מבנה הנתונים המעודכן לסריקת מועמדים ברשת עבור בעלי עסקים
class CandidateExtraction(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None                          # גיל המועמד
    city: Optional[str] = None                          # עיר מגורים / מיקום גאוגרפי
    current_title: Optional[str] = None                 # תפקיד נוכחי ברשת
    experience_years: Optional[float] = None            # שנות ניסיון משוערות
    skills: List[str] = []                              # כישורים טכנולוגיים או מקצועיים
    military_or_security_background: Optional[str] = None # רקע ביטחוני/צבאי
    seniority_level: Optional[str] = None               # רמת בכירות (Junior, Mid, Senior, Lead)

class CandidatePostRequest(BaseModel):
    raw_text: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/candidates")
def create_candidate(request: CandidatePostRequest):
    try:
        # שימוש ב-Structured Outputs של OpenAI לחילוץ מובנה לפי הסכמה שהגדרנו
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
        
        extracted_data = completion.choices.message.parsed
        
        # כאן בהמשך נחבר את השמירה ל-Supabase עם כל השדות החדשים
        return {
            "message": "Candidate parsed successfully",
            "data": extracted_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
