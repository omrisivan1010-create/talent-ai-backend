import streamlit as st
import requests
import os

# הגדרת כתובת ה-API שלך (אפשר לשנות לכתובת של Render או להריץ לוקאלית)
API_URL = os.environ.get("API_URL", "https://talent-ai-backend-q6r8.onrender.com")

st.set_page_title("AI Candidate Intelligence Platform", layout="wide")

st.title("🤖 מערכת בינה עסקית וגיוס מועמדים חכמה")
st.markdown("מערכת אוטומטית לניתוח חכם של קורות חיים לפי 52 פרמטרים וניהול מאגר מועמדים.")

# תפריט ניווט צדדי
menu = st.sidebar.selectbox("תפריט ניווט", ["הוספת מועמד חדש", "מאגר מועמדים"])

if menu == "הוספת מועמד חדש":
    st.header("📄 הוספה וניתוח מועמד חדש")
    st.markdown("הדבק כאן את טקסט קורות החיים או המידע הגולמי על המועמד:")
    
    raw_text_input = st.text_area("טקסט גולמי / קורות חיים", height=250)
    
    if st.button("נתח ושמור מועמד", type="primary"):
        if not raw_text_input.strip():
            st.warning("אנא הזן טקסט לפני שליחה.")
        else:
            with st.spinner("מנתח בעזרת בינה מלאכותית ושומר במסד הנתונים... ⏳"):
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/candidates",
                        json={"raw_text": raw_text_input}
                    )
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success("המועמד נותח ונשמר בהצלחה במסד הנתונים! 🎉")
                        st.json(res_data.get("data"))
                    else:
                        st.error(f"שגיאה בשמירה: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"שגיאת תקשורת מול ה-API: {str(e)}")

elif menu == "מאגר מועמדים":
    st.header("👥 מאגר המועמדים השמורים")
    
    if st.button("רענן רשימה"):
        st.rerun()
        
    try:
        response = requests.get(f"{API_URL}/api/v1/candidates?limit=50")
        if response.status_code == 200:
            candidates = response.json()
            
            if not candidates:
                st.info("עדיין אין מועמדים שמורים במערכת.")
            else:
                # יצירת בחירת מועמד מתוך רשימה
                candidate_options = {f"{c['id']} - {c.get('full_name', 'ללא שם')} (נוצר ב: {c['created_at'][:10]})": c['id'] for c in candidates}
                selected_label = st.selectbox("בחר מועמד לצפייה בפרופיל המלא:", list(candidate_options.keys()))
                
                if selected_label:
                    selected_id = candidate_options[selected_label]
                    
                    # שליפת פרטי המועמד המלאים לפי ID
                    detail_response = requests.get(f"{API_URL}/api/v1/candidates/{selected_id}")
                    if detail_response.status_code == 200:
                        candidate_full = detail_response.json()
                        
                        st.divider()
                        st.subheader(f"פרופיל מועמד: {candidate_full.get('full_name')}")
                        st.text(f"מספר מזהה במערכת (ID): {candidate_full.get('id')}")
                        st.text(f"נוצר בתאריך: {candidate_full.get('created_at')}")
                        
                        # הצגת ה-JSON המלא של 52 הפרמטרים בצורה מסודרת
                        st.markdown("### 📊 נתונים מנותחים (52 פרמטרים)")
                        st.json(candidate_full.get('candidate_data'))
                    else:
                        st.error("שגיאה בשליפת פרטי המועמד הספציפי.")
        else:
            st.error("שגיאה בטעינת רשימת המועמדים מהשרת.")
    except Exception as e:
        st.error(f"שגיאת תקשורת מול ה-API: {str(e)}")
