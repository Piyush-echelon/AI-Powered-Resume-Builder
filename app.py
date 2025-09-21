import os
import json
import streamlit as st
from typing import List, Dict
from dataclasses import dataclass, asdict, field
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv

load_dotenv()

# --- LangChain (LLM optional) ---
USE_LLM = True
LLM_PROVIDER = None
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    # Prefer GROQ if available
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        LLM_PROVIDER = "groq"
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        LLM_PROVIDER = "openai"
    else:
        USE_LLM = False
except Exception:
    USE_LLM = False
    LLM_PROVIDER = None

# --------------------------- Data Models ---------------------------
@dataclass
class PersonalInfo:
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    job_title: str = ""
    links: List[str] = field(default_factory=list)

@dataclass
class EducationEntry:
    institution: str = ""
    degree: str = ""
    location: str = ""
    start: str = ""
    end: str = ""
    percentage_or_gpa: str = ""

@dataclass
class ExperienceEntry:
    company: str = ""
    role: str = ""
    location: str = ""
    start: str = ""
    end: str = ""
    bullets: List[str] = field(default_factory=list)

@dataclass
class ProjectEntry:
    name: str = ""
    link: str = ""
    tech: str = ""
    bullets: List[str] = field(default_factory=list)

@dataclass
class CertificationEntry:
    name: str = ""
    issuer: str = ""
    year: str = ""

@dataclass
class ResumeData:
    personal: PersonalInfo = field(default_factory=PersonalInfo)
    education: List[EducationEntry] = field(default_factory=list)
    experience: List[ExperienceEntry] = field(default_factory=list)
    skillsets: Dict[str, List[str]] = field(default_factory=dict)
    projects: List[ProjectEntry] = field(default_factory=list)
    certifications: List[CertificationEntry] = field(default_factory=list)
    additional: Dict[str, List[str]] = field(default_factory=dict)

# --------------------------- Helpers ---------------------------
def get_env():
    return Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["html"])
    )

def render_html(data: ResumeData) -> str:
    env = get_env()
    template = env.get_template("resume.html")
    return template.render(d=asdict(data))

def init_state():
    if "resume" not in st.session_state:
        st.session_state.resume = ResumeData()
    if "download_html" not in st.session_state:
        st.session_state.download_html = ""

def ensure_llm():
    if not USE_LLM:
        return None
    if LLM_PROVIDER == "groq":
        return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return None

def polish_bullets(llm, role: str, bullets: List[str]) -> List[str]:
    if not llm:
        return bullets
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise resume editor. Rewrite the bullets to be action-oriented, quantified, and ATS-friendly. Keep each under 24 words."),
        ("user", "Role: {role}\nBullets:\n{bullets}\nReturn only the improved bullets as a numbered list.")
    ])
    chain = prompt | llm | StrOutputParser()
    improved = chain.invoke({"role": role, "bullets": "\n".join(f"- {b}" for b in bullets)})
    out = []
    for line in improved.splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        if line[0].isdigit():
            line = line.split(".", 1)[-1].strip()
        out.append(line)
    return out or bullets

# --------------------------- AI Generation ---------------------------
def extract_json_from_text(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        for i in range(len(parts)):
            if i + 1 < len(parts) and ("json" in parts[i].lower()):
                return parts[i + 1].strip()
        blocks = [p.strip() for p in parts if "{" in p and "}" in p]
        if blocks:
            return blocks[0]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text

def ai_generate_resume(llm, job_profile: str, extras: dict = None) -> ResumeData:
    if not llm:
        raise RuntimeError("LLM not configured.")
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    schema_hint = {
        "personal": {"first_name": "", "last_name": "", "email": "", "phone": "", "address": "", "job_title": "", "links": []},
        "education": [{"institution": "", "degree": "", "location": "", "start": "", "end": "", "percentage_or_gpa": ""}],
        "experience": [{"company": "", "role": "", "location": "", "start": "", "end": "", "bullets": [""]}],
        "skillsets": {"Programming Languages": [""], "Frameworks": [""]},
        "projects": [{"name": "", "link": "", "tech": "", "bullets": [""]}],
        "certifications": [{"name": "", "issuer": "", "year": ""}],
        "additional": {"Honors & Awards": [""]}
    }

    # escape braces so LangChain doesn't treat JSON keys as variables
    schema_text = json.dumps(schema_hint, indent=2).replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a resume-writing assistant. Create a strong, ATS-friendly resume draft. "
         "Return STRICT JSON ONLY matching the schema below. Do not add markdown or commentary."),
        ("user",
         "Job Profile: {job_profile}\n"
         "Extras: {extras}\n\n"
         f"Schema (keys/structure only):\n{schema_text}")
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"job_profile": job_profile, "extras": json.dumps(extras or {}, ensure_ascii=False)})

    clean = extract_json_from_text(raw)
    data = json.loads(clean)

    rd = ResumeData()
    p = data.get("personal", {})
    rd.personal = PersonalInfo(**{**asdict(PersonalInfo()), **p})
    rd.education = [EducationEntry(**e) for e in data.get("education", [])]
    rd.experience = [ExperienceEntry(**ex) for ex in data.get("experience", [])]
    rd.skillsets = data.get("skillsets", {})
    rd.projects = [ProjectEntry(**pr) for pr in data.get("projects", [])]
    rd.certifications = [CertificationEntry(**c) for c in data.get("certifications", [])]
    rd.additional = data.get("additional", {})
    return rd



# --------------------------- Sidebar ---------------------------
st.sidebar.title("🔑 Settings")

# Helpful link for new users
st.sidebar.markdown(
    "Don't have a Groq API key yet? [Get one here](https://console.groq.com/keys) 🔗"
)

# Input for API key
api_key_input = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password",
    value=st.session_state.get("groq_api_key", "")
)

# Save to session state if provided
if api_key_input:
    st.session_state.groq_api_key = api_key_input
    os.environ["GROQ_API_KEY"] = api_key_input  # make it visible for LangChain
    st.sidebar.success("Groq API key saved ✅")
else:
    st.sidebar.warning("⚠️ No Groq API key set. Enter one to enable AI features.")


# --------------------------- UI ---------------------------
st.set_page_config(page_title="AI Resume Builder", page_icon="📄", layout="wide")
st.title("📄 AI Resume Builder")
st.caption("Powered by Groq + LangChain🦜🔗")
st.caption("Made by Piyush 💗")
init_state()
resume: ResumeData = st.session_state.resume
llm = ensure_llm()

# 🚀 Generate with AI
with st.expander("🚀 Generate with AI", expanded=True):
    st.caption("Enter the job profile and let AI draft your resume.")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        job_profile = st.text_input("Target Job Profile", "", key="gen_job_profile")
        seniority = st.selectbox("Seniority", ["Junior", "Mid-level", "Senior", "Lead"], 1, key="gen_seniority")
        years = st.slider("Years of Experience", 0, 20, 2, key="gen_years")
    with col2:
        primary_stack = st.text_input("Primary Tech/Domain (optional)", "", key="gen_stack")
        industry = st.text_input("Industry (optional)", "", key="gen_industry")
        location_pref = st.text_input("Location (optional)", "", key="gen_location")
    extras = {
        "seniority": seniority,
        "years_experience": years,
        "primary_stack": primary_stack,
        "industry": industry,
        "location": location_pref
    }
    if st.button("✨ Generate Resume Draft", key="gen_button"):
        if not llm:
            st.error("LLM not configured.")
        elif not job_profile.strip():
            st.warning("Enter a job profile.")
        else:
            with st.spinner("Thinking..."):
                try:
                    generated = ai_generate_resume(llm, job_profile.strip(), extras)
                    st.session_state.resume = generated
                    resume = generated
                    st.success("Draft generated! Scroll down to review.")
                except Exception as e:
                    st.error(f"Failed: {e}")

# --------------------------- Personal Info ---------------------------
with st.expander("Personal Info", expanded=True):
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        resume.personal.first_name = st.text_input("First Name", value=resume.personal.first_name, key="pi_first")
        resume.personal.email = st.text_input("Email", value=resume.personal.email, key="pi_email")
        resume.personal.job_title = st.text_input("Job Title", value=resume.personal.job_title, key="pi_title")
    with c2:
        resume.personal.last_name = st.text_input("Last Name", value=resume.personal.last_name, key="pi_last")
        resume.personal.phone = st.text_input("Phone", value=resume.personal.phone, key="pi_phone")
        resume.personal.address = st.text_input("Address", value=resume.personal.address, key="pi_address")
    with c3:
        links = resume.personal.links or []
        n_links = st.number_input("Links (0-5)", 0, 5, len(links), key="pi_links_count")
        links = links[:n_links] + [""] * (n_links - len(links))
        for i in range(n_links):
            links[i] = st.text_input(f"Link {i+1}", value=links[i], key=f"pi_link_{i}")
        resume.personal.links = [l for l in links if l.strip()]

# --------------------------- Education ---------------------------
with st.expander("Education", expanded=False):
    count = st.number_input("How many entries?", 0, 6, len(resume.education), key="edu_count")
    while len(resume.education) < count:
        resume.education.append(EducationEntry())
    while len(resume.education) > count:
        resume.education.pop()
    for idx, e in enumerate(resume.education):
        st.subheader(f"Education {idx+1}")
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            e.institution = st.text_input(f"Institution {idx+1}", value=e.institution, key=f"edu_inst_{idx}")
            e.location = st.text_input(f"Location {idx+1}", value=e.location, key=f"edu_loc_{idx}")
        with c2:
            e.degree = st.text_input(f"Degree {idx+1}", value=e.degree, key=f"edu_deg_{idx}")
            e.start = st.text_input(f"Start {idx+1}", value=e.start, key=f"edu_start_{idx}")
        with c3:
            e.end = st.text_input(f"End {idx+1}", value=e.end, key=f"edu_end_{idx}")
            e.percentage_or_gpa = st.text_input(f"% / GPA {idx+1}", value=e.percentage_or_gpa, key=f"edu_gpa_{idx}")

# --------------------------- Experience ---------------------------
with st.expander("Experience", expanded=False):
    count = st.number_input("How many roles?", 0, 10, len(resume.experience), key="exp_count")
    while len(resume.experience) < count:
        resume.experience.append(ExperienceEntry())
    while len(resume.experience) > count:
        resume.experience.pop()
    for idx, ex in enumerate(resume.experience):
        st.subheader(f"Role {idx+1}")
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            ex.company = st.text_input(f"Company {idx+1}", value=ex.company, key=f"exp_company_{idx}")
            ex.location = st.text_input(f"Location {idx+1}", value=ex.location, key=f"exp_location_{idx}")
        with c2:
            ex.role = st.text_input(f"Title {idx+1}", value=ex.role, key=f"exp_role_{idx}")
            ex.start = st.text_input(f"Start {idx+1}", value=ex.start, key=f"exp_start_{idx}")
        with c3:
            ex.end = st.text_input(f"End {idx+1}", value=ex.end, key=f"exp_end_{idx}")

        bullets = ex.bullets or []
        n_bul = st.number_input(f"Bullets for role {idx+1}", 0, 10, len(bullets), key=f"exp_bul_count_{idx}")
        bullets = bullets[:n_bul] + [""] * (n_bul - len(bullets))
        for i in range(n_bul):
            bullets[i] = st.text_input(f"• Bullet {i+1} (Role {idx+1})", value=bullets[i], key=f"exp_bul_{idx}_{i}")
        if llm and st.button(f"✨ AI Polish Bullets (Role {idx+1})", key=f"exp_polish_{idx}"):
            bullets = polish_bullets(llm, ex.role or "Professional", bullets)
        ex.bullets = [b for b in bullets if b.strip()]

# --------------------------- Skillsets ---------------------------
with st.expander("Skillsets", expanded=False):
    st.caption("Add skill categories and items (e.g., Programming → Python, SQL).")
    raw = st.text_area(
        "Skillsets JSON (dict of lists)",
        value=json.dumps(resume.skillsets or {"Programming Languages": ["Python", "SQL"], "Frameworks": ["LangChain", "Django"]}, indent=2),
        height=180,
        key="skills_json",
    )
    try:
        resume.skillsets = json.loads(raw)
        if not isinstance(resume.skillsets, dict):
            st.error("Skillsets must be a dict of category -> list[str].")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

# --------------------------- Projects ---------------------------
with st.expander("Projects", expanded=False):
    count = st.number_input("How many projects?", 0, 10, len(resume.projects), key="proj_count")
    while len(resume.projects) < count:
        resume.projects.append(ProjectEntry())
    while len(resume.projects) > count:
        resume.projects.pop()
    for idx, p in enumerate(resume.projects):
        st.subheader(f"Project {idx+1}")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            p.name = st.text_input(f"Name {idx+1}", value=p.name, key=f"proj_name_{idx}")
            p.bullets = [b for b in st.text_area(f"Bullets (one per line) {idx+1}", value="\n".join(p.bullets), key=f"proj_bul_{idx}").splitlines() if b.strip()]
        with c2:
            p.link = st.text_input(f"Link {idx+1}", value=p.link, key=f"proj_link_{idx}")
            p.tech = st.text_input(f"Tech/Tools {idx+1}", value=p.tech, key=f"proj_tech_{idx}")

# --------------------------- Certifications ---------------------------
with st.expander("Certifications", expanded=False):
    count = st.number_input("How many certifications?", 0, 12, len(resume.certifications), key="cert_count")
    while len(resume.certifications) < count:
        resume.certifications.append(CertificationEntry())
    while len(resume.certifications) > count:
        resume.certifications.pop()
    for idx, c in enumerate(resume.certifications):
        st.subheader(f"Certification {idx+1}")
        c.name = st.text_input(f"Name {idx+1}", value=c.name, key=f"cert_name_{idx}")
        c.issuer = st.text_input(f"Issuer {idx+1}", value=c.issuer, key=f"cert_issuer_{idx}")
        c.year = st.text_input(f"Year {idx+1}", value=c.year, key=f"cert_year_{idx}")

# --------------------------- Additional ---------------------------
with st.expander("Additional (e.g., Honors & Awards, Interests)", expanded=False):
    raw = st.text_area(
        "Additional JSON (dict of lists)",
        value=json.dumps(resume.additional or {"Honors & Awards": ["Tech Star Spotlight"]}, indent=2),
        height=160,
        key="additional_json",
    )
    try:
        resume.additional = json.loads(raw)
        if not isinstance(resume.additional, dict):
            st.error("Additional must be a dict of section -> list[str].")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

# --------------------------- Preview & Export ---------------------------
st.divider()
left, right = st.columns([1, 1])
with left:
    if st.button("🔧 Build Preview", key="build_preview"):
        html = render_html(resume)
        st.session_state.download_html = html
        st.success("Preview ready! See the right panel.")

with right:
    if st.session_state.download_html:
        st.markdown("### Preview")
        st.components.v1.html(st.session_state.download_html, height=1200, scrolling=True)
        st.download_button("⬇️ Download HTML", st.session_state.download_html, "resume.html", "text/html", key="dl_html")

        # Try PDF export
        pdf_bytes = None
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=st.session_state.download_html, base_url=".").write_pdf()
        except Exception:
            try:
                import pdfkit, tempfile
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
                    tmp.write(st.session_state.download_html)
                    tmp_path = tmp.name
                pdf_bytes = pdfkit.from_file(tmp_path, False)
            except Exception:
                st.info("PDF engine not available. Use your browser 'Print → Save as PDF'.")
        if pdf_bytes:
            st.download_button("⬇️ Download PDF", pdf_bytes, "resume.pdf", "application/pdf", key="dl_pdf")

st.caption("Tip: If PDF generation isn't available, open the HTML and print to PDF from your browser for identical layout.")
