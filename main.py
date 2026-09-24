from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from supabase import create_client, Client
import os

app = FastAPI(title="AI Candidate Intelligence API", version="2.1.0")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# הגדרת חיבור ל-Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 1. ניסיון תעסוקתי והיסטוריית קריירה
class WorkHistoryDetails(BaseModel):
    companies_worked_at: List[str] = []
    tenure_per_company: List[str] = []
    current_title: Optional[str] = None
    time_in_current_role: Optional[str] = None
    career_progression_pace: Optional[str] = None
    career_path: Optional[str] = None

# 2. ניסיון ניהולי ופרויקטים
class ExperienceAndProjects(BaseModel):
    relevant_experience_years: Optional[float] = None
    similar_roles_experience: Optional[str] = None
    seniority_level: Optional[str] = None
    management_experience: Optional[bool] = None
    number_of_employees_managed: Optional[int] = None
    relevant_industry_experience: Optional[str] = None
    projects_executed: List[str] = []
    project_scale_and_scope: Optional[str] = None
    measurable_professional_achievements: List[str] = []

# 3. כישורים, השכלה וטכנולוגיות
class SkillsAndEducation(BaseModel):
    professional_skills: List[str] = []
    technologies_and_tools: List[str] = []
    academic_education: Optional[str] = None
    degrees_and_certifications: List[str] = []
    languages: List[str] = []
    english_level: Optional[str] = None

# 4. נוכחות דיגיטלית ופעילות מקצועית
class DigitalFootprint(BaseModel):
    professional_interests: List[str] = []
    online_professional_activity: Optional[str] = None
    publications_and_content: List[str] = []
    talks_conferences_communities: List[str] = []
    personal_projects: List[str] = []
    github_portfolio_sites: List[str] = []

# 5. זמינות, התאמה ונתוני שוק
class MarketAndFit(BaseModel):
    geographic_location: Optional[str] = None
    remote_hybrid_onsite_preference: Optional[str] = None
    estimated_availability_for_transition: Optional[str] = None
    occupational_stability: Optional[str] = None
    job_change_frequency: Optional[str] = None
    cultural_and_org_fit: Optional[str] = None
    market_demand_level: Optional[str] = None
    skill_rarity: Optional[str] = None
    demand_in_similar_roles: Optional[str] = None
    current_role_responsibility_level: Optional[str] = None
    current_company_size: Optional[str] = None
    current_company_type: Optional[str] = None
    role_fit_potential: Optional[str] = None
    gaps_vs_requirements: List[str] = []
    strengths_vs_requirements: List[str] = []

# 6. אלגוריתם תגמול ושכר
class CompensationAndPrediction(BaseModel):
    probability_interested_in_role: Optional[str] = None
    probability_open_to_offer: Optional[str] = None
    market_salary_range: Optional[str] = None
    estimated_current_salary: Optional[float] = None
    probability_to_switch_for_salary: Optional[str] = None
    possible_compensation_components: List[str] = []
    minimum_estimated_salary_threshold: Optional[float] = None
    attractive_salary_range: Optional[str] = None
    salary_estimation_certainty_level: Optional[str] = None
    data_quality_and_freshness: Optional[str] = None

class CandidateExtraction(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    work_history: Optional[WorkHistoryDetails] = None
    experience_and_projects: Optional[ExperienceAndProjects] = None
    skills_and_education: Optional[SkillsAndEducation] = None
    digital_footprint: Optional[DigitalFootprint] = None
    market_and_fit: Optional[MarketAndFit] = None
    compensation_prediction: Optional[CompensationAndPrediction] = None

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
                    "content": "אתה מנתח בכיר למערכת גיוס ובינה עסקית. תפקידך לחלץ ולנתח מתוך טקסט גולמי ברשת את כל 52 הפרמטרים המקצועיים, התעסוקתיים ותחזיות השכר של המועמד בדיוק מירבי."
                },
                {"role": "user", "content": request.raw_text},
            ],
            response_format=CandidateExtraction,
        )
        
        extracted_data = completion.choices[0].message.parsed
        
        # שמירת הנתונים בטבלת Supabase
        db_response = supabase.table("candidates").insert({
            "full_name": extracted_data.full_name,
            "candidate_data": extracted_data.model_dump()
        }).execute()
        
        return {
            "message": "Candidate parsed and saved to Supabase successfully",
            "data": extracted_data,
            "db_response": db_response.data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
