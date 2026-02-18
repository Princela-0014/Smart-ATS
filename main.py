# To run this app:
# uvicorn main:app --reload
# Install dependencies:
# pip install fastapi uvicorn pypdf python-dotenv python-multipart openai

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
from dotenv import load_dotenv
import io
import json
import os
import openai

load_dotenv()  # load .env file
# Demo mode - set to True to use mock data instead of OpenAI API
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client only when not in demo mode
client = None
if not DEMO_MODE:
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")
    client = openai.OpenAI(api_key=api_key)

# Skills database for keyword matching
HARD_SKILLS_DB = {
    "Python", "Java", "JavaScript", "C++", "C#", "PHP", "Ruby", "Go", "Rust", "Swift",
    "TypeScript", "Kotlin", "Scala", "R", "MATLAB", "SQL", "NoSQL", "MongoDB", "PostgreSQL",
    "MySQL", "Oracle", "Redis", "Elasticsearch", "React", "Vue", "Angular", "Node.js",
    "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "Express", "ASP.NET", "Laravel",
    "Docker", "Kubernetes", "AWS", "Azure", "Google Cloud", "GCP", "Git", "GitHub",
    "GitLab", "Jenkins", "CI/CD", "DevOps", "Apache", "Nginx", "REST API", "GraphQL",
    "AWS Lambda", "DynamoDB", "S3", "EC2", "RDS", "Kafka", "RabbitMQ", "Linux",
    "Windows", "Mac", "Unix", "HTML", "CSS", "XML", "JSON", "YAML", "Terraform",
    "Ansible", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn",
    "Pandas", "NumPy", "Data Science", "Analytics", "Tableau", "Power BI", "Hadoop",
    "Spark", "Hive", "Pig", "HDFS", "MapReduce", "Agile", "Scrum", "JIRA",
    "Confluence", "AWS EC2", "OpenAI", "LLM", "BERT", "GPT", "Regex", "API",
    "Microservices", "Monolith", "Architecture", "Database", "Queue", "Cache",
    "Load Balancing", "SSL", "TLS", "OAuth", "JWT", "Encryption", "Security"
}

SOFT_SKILLS_DB = {
    "Communication", "Leadership", "Teamwork", "Collaboration", "Problem-solving",
    "Critical thinking", "Time management", "Project management", "Organization",
    "Adaptability", "Flexibility", "Creative thinking", "Decision making", "Analysis",
    "Analytical thinking", "Research", "Documentation", "Presentation", "Public speaking",
    "Negotiation", "Conflict resolution", "Empathy", "Patience", "Motivation",
    "Initiative", "Self-learning", "Mentoring", "Feedback", "Willing to learn"
}

# Skill normalization dictionary for variations
SKILL_NORMALIZATION = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "py": "Python",
    "python": "Python",
    "c#": "C#",
    "csharp": "C#",
    "c++": "C++",
    "cpp": "C++",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "ml": "Machine Learning",
    "ai": "Machine Learning",
    "dl": "Deep Learning",
    "devops": "DevOps",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud",
    "git": "Git",
    "sql": "SQL",
    "nosql": "NoSQL",
    "html": "HTML",
    "css": "CSS",
    "xml": "XML",
    "json": "JSON",
    "yaml": "YAML",
    "api": "API",
    "rest": "REST API",
    "graphql": "GraphQL",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "linux": "Linux",
    "windows": "Windows",
    "mac": "Mac",
    "unix": "Unix",
    "agile": "Agile",
    "scrum": "Scrum",
    "jira": "JIRA",
    "confluence": "Confluence",
    "llm": "LLM",
    "bert": "BERT",
    "gpt": "GPT",
    "regex": "Regex",
    "microservices": "Microservices",
    "monolith": "Monolith",
    "architecture": "Architecture",
    "database": "Database",
    "queue": "Queue",
    "cache": "Cache",
    "ssl": "SSL",
    "tls": "TLS",
    "oauth": "OAuth",
    "jwt": "JWT",
    "encryption": "Encryption",
    "security": "Security",
    "communication": "Communication",
    "leadership": "Leadership",
    "teamwork": "Teamwork",
    "collaboration": "Collaboration",
    "problem-solving": "Problem-solving",
    "critical thinking": "Critical thinking",
    "time management": "Time management",
    "project management": "Project management",
    "organization": "Organization",
    "adaptability": "Adaptability",
    "flexibility": "Flexibility",
    "creative thinking": "Creative thinking",
    "decision making": "Decision making",
    "analysis": "Analysis",
    "analytical thinking": "Analytical thinking",
    "research": "Research",
    "documentation": "Documentation",
    "presentation": "Presentation",
    "public speaking": "Public speaking",
    "negotiation": "Negotiation",
    "conflict resolution": "Conflict resolution",
    "empathy": "Empathy",
    "patience": "Patience",
    "motivation": "Motivation",
    "initiative": "Initiative",
    "self-learning": "Self-learning",
    "mentoring": "Mentoring",
    "feedback": "Feedback",
    "willing to learn": "Willing to learn"
}

def normalize_text(text):
    """Normalize text by replacing skill variations with standard names"""
    words = text.lower().split()
    normalized_words = []
    for word in words:
        normalized_words.append(SKILL_NORMALIZATION.get(word, word))
    return ' '.join(normalized_words)

def extract_skills_from_text(text):
    """Extract skills from text using keyword matching with normalization"""
    normalized_text = normalize_text(text)
    hard_skills = [skill for skill in HARD_SKILLS_DB if skill.lower() in normalized_text]
    soft_skills = [skill for skill in SOFT_SKILLS_DB if skill.lower() in normalized_text]
    return hard_skills, soft_skills

def calculate_match_score(resume_text, jd_text):
    """Calculate JD match based on skill overlap"""
    resume_hard, resume_soft = extract_skills_from_text(resume_text)
    jd_hard, jd_soft = extract_skills_from_text(jd_text)
    
    total_jd_skills = len(jd_hard) + len(jd_soft)
    if total_jd_skills == 0:
        return 50
    
    matched_hard = len(set(resume_hard) & set(jd_hard))
    matched_soft = len(set(resume_soft) & set(jd_soft))
    total_matched = matched_hard + matched_soft
    
    percentage = min(100, int((total_matched / total_jd_skills) * 100))
    return percentage

def find_missing_keywords(resume_text, jd_text):
    """Find keywords in JD that are missing from resume"""
    resume_hard, resume_soft = extract_skills_from_text(resume_text)
    jd_hard, jd_soft = extract_skills_from_text(jd_text)
    
    missing_hard = [skill for skill in jd_hard if skill not in resume_hard]
    missing_soft = [skill for skill in jd_soft if skill not in resume_soft]
    
    return missing_hard, missing_soft

def generate_suggestions(resume_text, jd_text, missing_hard, missing_soft):
    """Generate actionable suggestions to improve match"""
    suggestions = []
    
    # Suggest adding missing hard skills
    if missing_hard:
        top_missing = missing_hard[:3]
        suggestions.append(f"📌 Add technical skills: Consider highlighting or developing expertise in {', '.join(top_missing)}. These are specifically mentioned in the job description.")
    
    # Suggest adding soft skills
    if missing_soft:
        top_missing_soft = missing_soft[:2]
        suggestions.append(f"💡 Emphasize soft skills: Make sure to highlight your experience with {', '.join(top_missing_soft)} in your resume, as the employer values these qualities.")
    
    # Suggest adding certifications or projects
    if "certification" in jd_text.lower() or "certified" in jd_text.lower():
        suggestions.append("🏆 Add certifications: The job posting emphasizes certifications. Include any relevant professional certifications you hold.")
    
    if "project" in jd_text.lower():
        suggestions.append("📂 Add project experience: Include 2-3 relevant projects in your resume that demonstrate the required skills in practical scenarios.")
    
    if "year" in jd_text.lower() or "experience" in jd_text.lower():
        suggestions.append("⏱️ Highlight relevant experience: Structure your experience section to clearly show years of experience with the technologies mentioned in the job description.")
    
    # Suggest improving summary
    if len(resume_text.split()) < 200:
        suggestions.append("📝 Expand your profile: Add a professional summary that highlights your key skills and experience relevant to this role.")
    
    return suggestions[:4]  # Return top 4 suggestions

def generate_polite_summary(match_score, resume_text, jd_text, missing_hard, missing_soft):
    """Generate a polite and helpful summary"""
    resume_hard, _ = extract_skills_from_text(resume_text)
    jd_hard, _ = extract_skills_from_text(jd_text)
    
    total_missing = len(missing_hard) + len(missing_soft)
    
    if match_score >= 85:
        summary = f"🌟 Excellent Match! ({match_score}%) Your resume aligns very well with this job posting. You possess {len(resume_hard)} of the key technical skills, which is fantastic! With just a few minor additions like {', '.join(missing_hard[:2]) if missing_hard else 'enhancing soft skills'}, you'd be a perfect fit. You're clearly well-prepared for this role—keep up the great work!"
    elif match_score >= 70:
        summary = f"✓ Good Match! ({match_score}%) Your resume shows solid alignment with the job requirements. You have {len(resume_hard)} relevant technical skills, which is impressive! By incorporating {total_missing} additional skills and experiences, you can significantly strengthen your application. This is a great foundation to build upon."
    elif match_score >= 50:
        summary = f"⚠️ Moderate Match ({match_score}%) - Your resume has potential! You currently have {len(resume_hard)} relevant technical skills, and that's a strong start. To improve your candidacy, consider adding experience with {', '.join(missing_hard[:3])}. This presents an exciting opportunity to develop new skills and enhance your profile."
    else:
        summary = f"📈 Room for Growth ({match_score}%) - Every great career starts somewhere, and you have a wonderful opportunity here! Focus on highlighting {', '.join(missing_hard[:3])} to better align with this opportunity. Our suggestions will help you become an even stronger candidate—embrace this as a chance to grow!"
    
    return summary

app = FastAPI(docs_url=None, redoc_url=None)
# -------------------------------
# Home route
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse("<h2>ATS Resume Analyzer Running</h2>", status_code=200)

# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_pdf_text(file_stream):
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"  # Add line breaks for readability
        if not text.strip():
            raise ValueError("No extractable text found in PDF.")
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

# -------------------------------
# Analyze resume with LLM
# -------------------------------
def analyze_with_llm(resume_text, jd_text):
    # Demo mode - return detailed, helpful analysis
    if DEMO_MODE:
        hard_resume, soft_resume = extract_skills_from_text(resume_text)
        hard_jd, soft_jd = extract_skills_from_text(jd_text)
        
        match_score = calculate_match_score(resume_text, jd_text)
        missing_hard, missing_soft = find_missing_keywords(resume_text, jd_text)
        
        summary = generate_polite_summary(match_score, resume_text, jd_text, missing_hard, missing_soft)
        suggestions = generate_suggestions(resume_text, jd_text, missing_hard, missing_soft)
        
        # Create detailed response
        return {
            "JD Match": f"{match_score}%",
            "Missing Keywords": missing_hard if missing_hard else ["No critical missing skills"],
            "Missing Soft Skills": missing_soft if missing_soft else ["Your soft skills are well-aligned"],
            "Hard Skills": hard_resume if hard_resume else ["Please add technical skills to your resume"],
            "Soft Skills": soft_resume if soft_resume else ["Emphasize soft skills in your resume"],
            "Profile Summary": summary,
            "Suggestions for Improvement": suggestions,
            "Matched Skills": list(set(hard_resume + soft_resume) & set(hard_jd + soft_jd))
        }
    
    prompt = f"""
    Act as a professional career advisor and advanced ATS system expert. Thoroughly analyze the provided resume against the job description, identifying all relevant skills, experiences, qualifications, and requirements. Provide a comprehensive, polite, encouraging, and confidence-building assessment that highlights the user's strengths and frames any gaps as opportunities for growth.

    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Return ONLY valid JSON in this exact format (no extra text):
    {{
      "JD Match": "percentage",
      "Hard Skills": [],
      "Soft Skills": [],
      "Missing Keywords": [],
      "Missing Soft Skills": [],
      "Matched Skills": [],
      "Profile Summary": "A polite, encouraging summary that appreciates the user's existing skills and experiences, builds confidence, and motivates improvement. Highlight positives first, then gently suggest enhancements.",
      "Suggestions for Improvement": ["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4"]
    }}

    Guidelines:
    - Be extremely polite, positive, and supportive throughout the analysis.
    - Always start the summary by appreciating and highlighting the user's strengths, matched skills, and relevant experiences.
    - Frame any missing skills or gaps as exciting opportunities for learning and growth, not as failures.
    - Use encouraging language like "You have a great foundation with...", "Building on your existing skills...", "This is a wonderful chance to develop...".
    - Provide specific, actionable, and motivating suggestions for improvement.
    - Include emojis for visual appeal and to make the feedback more engaging.
    - Ensure the match percentage accurately reflects the overlap, but emphasize potential and progress.
    - Cover all aspects of the job description thoroughly in the analysis.
    """

    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client is not configured. Set OPENAI_API_KEY or enable DEMO_MODE=true.")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        # Attempt to clean and parse JSON
        if not content.startswith("{"):
            content = content[content.find("{"):]  # Remove leading non-JSON text
        return json.loads(content)
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key. Please check your .env file or environment variable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

# -------------------------------
# Analyze resume endpoint
# -------------------------------
@app.post("/analyze/")
async def analyze(jd: str = Form(...), resume: UploadFile = File(...)):
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await resume.read()
    resume_text = extract_pdf_text(io.BytesIO(file_bytes))

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text could not be extracted.")

    # Check PDF parse quality
    word_count = len(resume_text.split())
    if word_count < 100:
        parse_quality = "Hard to parse - The PDF may be image-based, scanned, or contain minimal text. Consider using a text-based PDF for better analysis."
    else:
        parse_quality = "Easy to parse - The PDF contains sufficient text for accurate analysis."

    llm_output = analyze_with_llm(resume_text, jd)
    llm_output["Parse Quality"] = parse_quality

    return llm_output

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
