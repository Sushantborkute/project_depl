import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from google import genai as genai_v2 # Standardizing name for the client
from google import genai
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------
# 1. PAGE CONFIG & GLOBAL SETTINGS
# ------------------------------------------
st.set_page_config(
    page_title="💼 Job Intelligence System",
    page_icon="🤖",
    layout="wide"
)

# Sidebar Controls
st.sidebar.title("⚙️ Global Explorer Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
top_n = st.sidebar.number_input("Top N records for Map", min_value=1, max_value=100, value=10)
plotly_template = "plotly_dark" if dark_mode else "plotly"

# ------------------------------------------
# SESSION STATE INITIALIZATION  ✅ ADDED
# ------------------------------------------
if "resume_ready" not in st.session_state:
    st.session_state.resume_ready = False

if "resume_data" not in st.session_state:
    st.session_state.resume_data = {}

# ------------------------------------------
# CUSTOM CSS
# ------------------------------------------
st.markdown("""
<style>
.chat-box { padding: 14px; border-radius: 10px; margin-bottom: 10px; font-size: 15px; line-height: 1.6; }
.user { background-color: #FFF1DC; color: #2B2B2B; border-left: 5px solid #FF9800; }
.bot { background-color: #E6F2FF; color: #1A1A1A; border-left: 5px solid #2196F3; }
.chat-box b { display: block; margin-bottom: 4px; }
.big-title { font-size: 32px; font-weight: bold; margin-bottom: 10px; }
.section-header { font-size: 20px; font-weight: bold; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# API & DATA INITIALIZATION
# ------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Google API Key not found.")
    st.stop()

client = genai_v2.Client(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY

@st.cache_data
def load_job_data():
    return pd.read_parquet("s3://rag-job-data-bucket-xyz/ai_job_dataset.parquet")

@st.cache_resource
def load_salary_model():
    with open("salary_model.pkl", "rb") as f:
        return pickle.load(f)

df = load_job_data()
model = load_salary_model()

# ------------------------------------------
# TABS
# ------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Predict Your Salary", "🤖 AI Powered Chatbot", "🌍 Global Search"])

# ===================================================================================
# TAB 1 — SALARY PREDICTION
# ===================================================================================
with tab1:
    st.markdown("<div class='big-title'>💼 Salary Prediction Assistant</div>", unsafe_allow_html=True)

    employment_type_map = {
        "Full Time": "FT",
        "Part Time": "PT",
        "Contract": "CT",
        "Freelance": "FL"
    }

    company_size_map = {
        "Small": "S",
        "Medium": "M",
        "Large": "L"
    }

    remote_ratio_map = {
        "Onsite (0%)": "0",
        "Hybrid (50%)": "50",
        "Remote (100%)": "100"
    }

    col1, col2 = st.columns(2)

    # ------------------ JOB INFO ------------------
    with col1:
        st.markdown("<div class='section-header'>📌 Job Information</div>", unsafe_allow_html=True)

        employment_type = employment_type_map[
            st.selectbox("Employment Type", list(employment_type_map.keys()))
        ]

        company_location = st.text_input("Company Country")
        company_size = company_size_map[
            st.selectbox("Company Size", list(company_size_map.keys()))
        ]

        industry = st.text_input("Industry")
        remote_ratio = remote_ratio_map[
            st.selectbox("Remote Ratio", list(remote_ratio_map.keys()))
        ]

    # ------------------ RESUME UPLOAD ------------------
    with col2:
        st.markdown("<div class='section-header'>📄 Upload Resume</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        if uploaded_file:
            try:
                pdf_bytes = uploaded_file.read()
                prompt = """Extract the following JSON fields from the provided resume. Output ONLY valid JSON.
                {
                  "job_title": "",
                  "experience_level": "EN | MI | SE | EX",
                  "education_required": "Associate | Bachelor | Master | PhD",
                  "num_required_skills": 0,
                  "years_experience": 0
                }"""

                pdf_part = genai.types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, pdf_part]
                )

                cleaned = response.text.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned)

                st.session_state.resume_data = {
                    "job_title": result.get("job_title", "N/A"),
                    "experience_level": result.get("experience_level", "N/A"),
                    "education_required": result.get("education_required", "N/A"),
                    "num_required_skills": int(result.get("num_required_skills", 0)),
                    "years_experience": int(result.get("years_experience", 0))
                }

                st.session_state.resume_ready = True
                st.success("✅ Resume processed successfully!")

            except Exception as e:
                st.session_state.resume_ready = False
                st.error(f"❌ Error reading resume: {e}")
        else:
            st.session_state.resume_ready = False

    company_ok = bool(company_location.strip())
    industry_ok = bool(industry.strip())
    # ------------------ PREDICT BUTTON (SAFE) ------------------
    if not st.session_state.resume_ready:
        st.info("📄 Upload a resume,company country, industry to enable salary prediction.")

    if st.session_state.resume_ready and company_ok and industry_ok:
        if st.button("🔮 Predict Salary", key="salary_btn"):
            d = st.session_state.resume_data

            user_df = pd.DataFrame([{
                "job_title": d["job_title"],
                "experience_level": d["experience_level"],
                "employment_type": employment_type,
                "company_location": company_location,
                "company_size": company_size,
                "education_required": d["education_required"],
                "industry": industry,
                "years_experience": d["years_experience"],
                "num_required_skills": d["num_required_skills"],
                "remote_ratio": str(remote_ratio)
            }])

            salary = np.expm1(model.predict(user_df)[0])

            st.markdown(
                f"""
                <div style='padding:20px; background:#f0f2f6; border-radius:10px;'>
                    <h3 style='color:#2E7D32;'>💲 Predicted Salary (USD)</h3>
                    <h1 style='color:#2E7D32;'>${salary:,.2f}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

# ===================================================================================
# TAB 2 — CHATBOT (UNCHANGED)
# ===================================================================================
with tab2:
    st.markdown("<div class='big-title'>🤖 RAG AI Chatbot</div>", unsafe_allow_html=True)

    embedding_function = embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    chroma = Chroma(persist_directory="./chroma", collection_name="job_dataset",
                    embedding_function=embedding_function)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask job or salary-related questions:")
    if st.button("Send 🚀"):
        docs = chroma.similarity_search(user_query, k=15)
        context = "\n\n".join(d.page_content for d in docs)
        answer = llm.invoke(
            f"Answer ONLY using context.\nQuery:{user_query}\nContext:{context}"
        ).content

        st.session_state.chat_history.extend([
            ("You", user_query),
            ("Bot", answer)
        ])

    for role, msg in st.session_state.chat_history:
        st.markdown(
            f"<div class='chat-box {'user' if role=='You' else 'bot'}'><b>{role}:</b> {msg}</div>",
            unsafe_allow_html=True
        )
# -----------------------------------------------------------------------------------
# TAB 3 — GLOBAL TOP JOBS EXPLORER
# -----------------------------------------------------------------------------------
with tab3:
    st.markdown("<div class='big-title'>🌍 Highest Paying Jobs Explorer</div>", unsafe_allow_html=True)

    # Country latitude, longitude & map scope
    COUNTRY_DATA = {
        "United States": {"lat": 37.0902, "lon": -95.7129, "scope": "north america"},
        "Canada": {"lat": 56.1304, "lon": -106.3468, "scope": "north america"},
        "India": {"lat": 20.5937, "lon": 78.9629, "scope": "asia"},
        "Germany": {"lat": 51.1657, "lon": 10.4515, "scope": "europe"},
        "United Kingdom": {"lat": 55.3781, "lon": -3.4360, "scope": "europe"},
        "France": {"lat": 46.2276, "lon": 2.2137, "scope": "europe"},
        "Australia": {"lat": -25.2744, "lon": 133.7751, "scope": "world"},
        "Switzerland": {"lat": 46.8182, "lon": 8.2275, "scope": "europe"}
    }

    # ------------------- Filters -------------------
    c1, c2 = st.columns(2)

    with c1:
        country_list = ["World"] + sorted(df["company_location"].dropna().unique().tolist())
        selected_country = st.selectbox("Select Country (World = Global View)", country_list)

    with c2:
        job_list = ["All Jobs"] + sorted(df["job_title"].dropna().unique().tolist())
        selected_job = st.selectbox("Select Job Title", job_list)

    # ------------------- Filter Logic -------------------
    filtered_df = df.copy()

    if selected_country != "World":
        filtered_df = filtered_df[filtered_df["company_location"] == selected_country]

    if selected_job != "All Jobs":
        filtered_df = filtered_df[filtered_df["job_title"] == selected_job]

    top_jobs = filtered_df.sort_values("salary_usd", ascending=False).head(top_n)

    # ------------------- Map Visualization -------------------
    if not top_jobs.empty:
        map_df = top_jobs.copy()

        map_df["lat"] = map_df["company_location"].map(
            lambda c: COUNTRY_DATA.get(c, {}).get("lat")
        )
        map_df["lon"] = map_df["company_location"].map(
            lambda c: COUNTRY_DATA.get(c, {}).get("lon")
        )

        map_df = map_df.dropna(subset=["lat", "lon"])

        if not map_df.empty:
            # Add jitter so overlapping points are visible
            map_df["lat"] += np.random.uniform(-1.5, 1.5, len(map_df))
            map_df["lon"] += np.random.uniform(-1.5, 1.5, len(map_df))

            map_scope = (
                COUNTRY_DATA.get(selected_country, {}).get("scope", "world")
                if selected_country != "World"
                else "world"
            )

            proj = "natural earth" if map_scope == "world" else None

            fig = px.scatter_geo(
                map_df,
                lat="lat",
                lon="lon",
                size="salary_usd",
                color="salary_usd",
                hover_name="job_title",
                scope=map_scope,
                projection=proj,
                template=plotly_template,
                title=f"Mapping Top {len(top_jobs)} Results",
                hover_data={
                    "company_location": True,
                    "years_experience": True,   # ✅ Experience shown
                    "salary_usd": ":,.0f",
                    "lat": False,
                    "lon": False
                },
                labels={
                    "company_location": "Country",
                    "experience_level": "Experience Level",
                    "salary_usd": "Salary (USD)"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

        # ------------------- Ranked Table -------------------
        st.markdown("### 📋 Ranked Results")

        st.dataframe(
            top_jobs[["job_title", "company_location", "experience_level", "salary_usd"]]
            .rename(columns={
                "job_title": "Job Title",
                "company_location": "Country",
                "experience_level": "Experience Level",
                "salary_usd": "Salary (USD)"
            })
            .style.format({"Salary (USD)": "${:,.0f}"}),
            use_container_width=True
        )

    else:
        st.warning("No data found for the selected filters.")

