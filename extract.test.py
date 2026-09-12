import pdfplumber
import os

resume_folder = "resumes"  # we'll put resume PDFs in this subfolder

for filename in os.listdir(resume_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(resume_folder, filename)
        with pdfplumber.open(filepath) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        print(f"--- {filename} ---")
        print(text[:200])  
        print()