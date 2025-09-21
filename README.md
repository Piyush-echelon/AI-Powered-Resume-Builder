# 📄 AI Resume Builder

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://airesumebuilde-nf3kjw8iy6mawkvt588hwz.streamlit.app/)  
Create professional, ATS-friendly resumes in minutes with the power of **LangChain** and **Groq AI**.

---

## 🚀 Hosted on Streamlit
👉 [Try it here](https://airesumebuilde-nf3kjw8iy6mawkvt588hwz.streamlit.app/)

<img width="1910" height="909" alt="image" src="https://github.com/user-attachments/assets/32b66320-75da-419d-94e9-9a2504b3e90b" />

---

## ✨ Features
- **AI-Generated Drafts** – Enter your job profile and let AI generate a complete resume (Education, Experience, Skills, Projects, Certifications, etc.).
- **Manual Editing** – Fine-tune each section with an intuitive editor.
- **Smart Bullet Polishing** – AI rewrites your experience bullets to be action-oriented and recruiter-friendly.
- **Real-time Preview** – Instantly see your resume in a clean template.
- **Export Options** – Download as **HTML** or **PDF**.
- **🔑 API Key Management** – Enter your **Groq API Key** from the sidebar (with a link for new users).

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – Interactive web app framework  
- [LangChain](https://www.langchain.com/) – Prompt orchestration  
- [Groq](https://console.groq.com/keys) – LLM provider  
- [Jinja2](https://palletsprojects.com/p/jinja/) – HTML templating  
- [WeasyPrint](https://weasyprint.org/) / [pdfkit](https://pypi.org/project/pdfkit/) – PDF export  

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/your-username/ai-resume-builder.git
cd ai-resume-builder
```

---

## Create a virtual environment and activate it:
```bash
python -m venv .venv
# On macOS/Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

---

## Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup

- Get a free API key from Groq: https://console.groq.com/keys

- Run the app and paste your key into the sidebar input.

- Alternatively, create a .env file in the project root:
```bash
GROQ_API_KEY=your_api_key_here
```

---

## ▶️Run Locally
```bash
streamlit run app.py
```

---

## 📋 Requirements

- Your requirements.txt should include:

```bash
streamlit
jinja2
python-dotenv

# LangChain & LLM providers
langchain
langchain-core
langchain-community
langchain-groq
langchain-openai

# PDF export
weasyprint
pdfkit
```

---

## ⚠️ Windows users:

- Installing weasyprint may be tricky; you can remove it and just use pdfkit.

- If using pdfkit, install wkhtmltopdf
 and add it to your PATH.


---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome.
Fork the repo, create a branch, and open a pull request 🚀
