# AI Resume Builder (LangChain + Streamlit)

A clean resume builder that asks for:
- Personal Info (First/Last Name, Email, Phone, Address, Job Title, Links)
- Education
- Experience (with bullets + optional "AI polish")
- Skillsets (category → items)
- Projects
- Certifications
- Additional sections (e.g., Honors & Awards)

Then it renders a PDF-ready HTML that mimics a formal single-column style.
If WeasyPrint is available, you can download a PDF directly.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# set one of the following:
export OPENAI_API_KEY=sk-...      # for OpenAI
# or
export GROQ_API_KEY=...           # for Groq

streamlit run app.py
```

If you can't install WeasyPrint, use your browser's **Print → Save as PDF** from the HTML preview.

## Notes
- The template lives in `templates/resume.html`. Tweak the CSS for your exact visual.
- The "AI Polish" feature uses LangChain and whichever LLM you configured.