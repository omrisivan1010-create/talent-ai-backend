from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os

app = FastAPI(title="AI Candidate Intelligence API", version="2.0.0")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 1. ניסיון תעסוקתי והיסטוריית קריירה
class WorkHistoryDetails(BaseModel):
    companies_worked_at: List[str] = []                  # 14. חברות שבהן עבד בעבר
    tenure_per_company: List[str] = []                   # 15. ותק בכל חברה
    current_title: Optional[str] = None                  # 17. תפקיד נוכחי
    time_in_current_role: Optional[str] = None           # 18. משך הזמן בתפקיד הנוכחי
    career_progression_pace: Optional[str] = None        # 16. קצב התקדמות בקריירה
    career_path: Optional[str] = None                    # 39. מסלול הקריירה

# 2. ניסיון ניהולי ופרויקטים
class ExperienceAndProjects(BaseModel):
    relevant_experience_years: Optional[float] = None    # 1. שנות ניסיון רלוונטי
    similar_roles_experience: Optional[str] = None       # 2. ניסיון בתפקידים דומים
    seniority_level: Optional[str] = None                # 3. רמת Seniority
    management_experience: Optional[bool] = None         # 4. ניסיון ניהולי
    number_of_employees_managed: Optional[int] = None    # 5. מספר עובדים שניהל
    relevant_industry_experience: Optional[str] = None   # 6. ניסיון בתחום התעשייה הרלוונטי
    projects_executed: List[str] = []                    # 11. פרויקטים שביצע
    project_scale_and_scope: Optional[str] = None        # 12. גודל והיקף הפרויקטים
    measurable_professional_achievements: List[str] = [] # 13. הישגים מקצועיים מדידים

# 3. כישורים, השכלה וטכנולוגיות
class SkillsAndEducation(BaseModel):
    professional_skills: List[str] = []                  # 7. כישורים מקצועיים
    technologies_and_tools: List[str] = []             # 8. טכנולוגיות וכלים שבהם השתמש
    academic_education: Optional[str] = None             # 9. השכלה אקדמית
    degrees_and_certifications: List[str] = []           # 10. תארים והסמכות מקצועיות
    languages: List[str] = []                            # 25. שפות
    english_level: Optional[str] = None                  # 26. רמת אנגלית

# 4. נוכחות דיגיטלית ופעילות מקצועית
class DigitalFootprint(BaseModel):
    professional_interests: List[str] = []               # 19. תחומי עניין מקצועיים
    online_professional_activity: Optional[str] = None   # 20. פעילות מקצועית ברשת
    publications_and_content: List[str] = []             # 21. פרסומים ותוכן מקצועי
    talks_conferences_communities: List[str] = []        # 22. הרצאות, כנסים וקהילות מקצועיות
    personal_projects: List[str] = []                    # 23. פרויקטים אישיים
    github_portfolio_sites: List[str] = []               # 24. פעילות ב-GitHub / Portfolio / אתרים

# 5. זמינות, התאמה ונתוני שוק
class MarketAndFit(BaseModel):
    geographic_location: Optional[str] = None            # 27. מיקום גיאוגרפי
    remote_hybrid_onsite_preference: Optional[str] = None# 28. נכונות לעבודה מרחוק / היברידית / מהמשרד
    estimated_availability_for_transition: Optional[str] = None # 29. זמינות משוערת למעבר תפקיד
    occupational_stability: Optional[str] = None         # 30. יציבות תעסוקתית
    job_change_frequency: Optional[str] = None           # 31. תדירות החלפת מקומות עבודה
    cultural_and_org_fit: Optional[str] = None           # 32. התאמה תרבותית לתפקיד ולארגון
    market_demand_level: Optional[str] = None            # 33. רמת ביקוש בשוק לכישורים
    skill_rarity: Optional[str] = None                   # 34. נדירות הכישורים שלו בשוק
    demand_in_similar_roles: Optional[str] = None        # 35. ביקוש למועמד בתפקידים דומים
    current_role_responsibility_level: Optional[str] = None # 36. רמת האחריות בתפקיד הנוכחי
    current_company_size: Optional[str] = None           # 37. גודל החברה הנוכחית
    current_company_type: Optional[str] = None           # 38. סוג החברה — סטארטאפ / תאגיד / וכו'
    role_fit_potential: Optional[str] = None             # 40. פוטנציאל התאמה לתפקיד המבוקש
    gaps_vs_requirements: List[str] = []                 # 41. פערים בין דרישות המשרה לפרופיל
    strengths_vs_requirements: List[str] = []            # 42. חוזקות ביחס לדרישות המשרה

# 6. אלגוריתם תגמול ושכר (Salary & Compensation Engine)
class CompensationAndPrediction(BaseModel):
    probability_interested_in_role: Optional[str] = None # 43. סבירות שהמועמד יהיה מעוניין במשרה
    probability_open_to_offer: Optional[str] = None      # 44. סבירות שהמועמד יהיה פתוח להצעה
    market_salary_range: Optional[str] = None            # 45. טווח שכר שוק לתפקיד ולניסיון
    estimated_current_salary: Optional[float] = None     # 46. רמת השכר המשוערת בתפקיד הנוכחי
    probability_to_switch_for_salary: Optional[str] = None # 47. סבירות לשינוי מקום עבודה תמורת שכר
    possible_compensation_components: List[str] = []     # 48. רכיבי תגמול אפשריים — בונוס, מניות וכו'
    minimum_estimated_salary_threshold: Optional[float] = None # 49. טווח שכר מינימלי משוער שיגרום לו לשקול הצעה
    attractive_salary_range: Optional[str] = None        # 50. טווח שכר אטרקטיבי משוער
    salary_estimation_certainty_level: Optional[str] = None # 51. רמת ודאות של הערכת השכר
    data_quality_and_freshness: Optional[str] = None     # 52. איכות ועדכניות המידע שעליו מבוססת ההערכה

# המודל המרכזי שמאגד את כל הקטגוריות
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
        
        return {
            "message": "Candidate comprehensive profile parsed successfully",
            "data": extracted_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
