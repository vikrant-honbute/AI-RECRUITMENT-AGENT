import re
import PyPDF2
import io
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import os
import json

class ResumeAnalysisAgent:
    def __init__(self, api_key, cutoff_score=75):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.rag_vectorstore = None
        self.analysis_result = None
        self.jd_text = None
        self.extracted_skill = None
        self.resume_weakness = []
        self.resume_strength = []
        self.improvement_suggestions = {}

    def _retry_with_backoff(self, func, *args, max_retries=5):
        """Retry a function call with exponential backoff on rate limit errors"""
        for attempt in range(max_retries):
            try:
                return func(*args)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = min(2 ** attempt * 5, 60)
                    time.sleep(wait)
                else:
                    raise
        return func(*args)


    def extract_text_from_pdf(self, pdf_file):
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                reader = PyPDF2.PdfReader(pdf_file_like)
            else:
                reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_text(self, txt_file):
        # Remove extra whitespace and newlines
        try:
            if hasattr(txt_file, 'getvalue'):
                return txt_file.getvalue().decode('utf-8')
            else:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""

    def extract_text_from_file(self, file):
        if hasattr(file, 'name'):
            file_extension = file.name.split('.')[-1].lower()
        else:
            file_extension = file.split('.')[-1].lower()

        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'txt':
            return self.extract_text_from_text(file)
        else:
            print(f"Unsupported file type: {file_extension}")
            return ""

    def create_rag_vectorstore(self, text):
        """create a vector store for RAG"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore

    def analyze_skill(self, qa_chain, skill):
        """analyze a skill in resume"""
        query = f"On the scale of 0 to 10, how clearly does the candidate mention proficiency in {skill}? Provide a numeric rating first, followed by reasoning for the rating. Be specific and provide examples from the resume to support your analysis."

        response = qa_chain.run(query)
        match = re.search(r"(\d{1,2})", response)
        score = int(match.group(1)) if match else 0

        reasoning = response.split('.', 1)[1].strip() if '.' in response and len(response.split('.', 1)) > 1 else "No reasoning provided."

        return skill, min(score, 10), reasoning

    def analyze_skills_batch(self, resume_text, skills_batch):
        """Analyze ALL skills in one LLM call to minimize API usage"""
        skills_list = ", ".join(skills_batch)
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=self.api_key)

        prompt = f"""Rate the candidate's proficiency in each skill: {skills_list}

Resume:
{resume_text[:2000]}

For EACH skill, give a score 0-10 and ONE sentence reasoning.
Return ONLY valid JSON:
{{
    "skill_name": {{"score": 7, "reasoning": "brief explanation"}}
}}
"""
        response = self._retry_with_backoff(llm.invoke, prompt)
        content = response.content

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # Fix common LLM issues: single quotes, trailing commas
                fixed = json_match.group(0).replace("'", '"')
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    return {}
        return {}
    
    def analyze_resume_weakness(self):
        """analyze weaknesses for ALL missing skills in a single LLM call"""
        if not self.resume_text or not self.extracted_skill or not self.analysis_result:
            return []

        missing_skills = self.analysis_result.get("missing_skills", [])
        if not missing_skills:
            return []

        skills_list = ", ".join(missing_skills)
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=self.api_key)
        prompt = f"""For each of these skills the candidate is weak in: {skills_list}

Resume:
{self.resume_text[:2000]}

For EACH skill, provide:
- weakness: what's missing (1 sentence)
- improvement_suggestions: 2 actionable suggestions
- example_addition: 1 bullet point to add

Return ONLY valid JSON array:
[
  {{"skill": "skill_name", "weakness": "...", "improvement_suggestions": ["...", "..."], "example_addition": "..."}}
]
"""

        try:
            response = self._retry_with_backoff(llm.invoke, prompt)
            content = response.content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                weaknesses = json.loads(json_match.group(0))
                self.resume_weakness = weaknesses
                return weaknesses
        except Exception:
            pass

        # Fallback: simple weakness entries without LLM
        weaknesses = [{"skill": s, "weakness": f"Low proficiency in {s}", "improvement_suggestions": [], "example_addition": ""} for s in missing_skills]
        self.resume_weakness = weaknesses
        return weaknesses
        


    
    def extract_skills_from_jd(self,jd_text):
        """extract skills from job description"""

        try:
            llm = ChatGroq(model="llama-3.1-8b-instant", api_key=self.api_key)
            prompt = f"""Extract the technical skills, technologies, and competencies from this job description.
Return ONLY a JSON array of strings, nothing else. Use double quotes.
Example: ["Python", "Machine Learning", "AWS"]

Job Description:
{jd_text[:2000]}"""

            response = self._retry_with_backoff(llm.invoke, prompt)
            skills_text = response.content

            match = re.search(r"\[.*\]", skills_text, re.DOTALL)
            if match:
                skills_list_str = match.group(0)
                # Handle single quotes (Python list format) by replacing them
                try:
                    skills_list = json.loads(skills_list_str)
                except json.JSONDecodeError:
                    import ast
                    skills_list = ast.literal_eval(skills_list_str)
                self.extracted_skill = [str(s) for s in skills_list if isinstance(s, str)]
                return self.extracted_skill
            else:
                # Fallback: extract quoted strings from the response
                fallback = re.findall(r'["\']([A-Za-z][\w\s/\-\+\.#]*)["\']', skills_text)
                if fallback:
                    self.extracted_skill = fallback
                    return fallback
                print("No list found in the response.")
                return []
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return []
        

    def semantic_skill_analysis(self, resume_text, skills):
        """analyze skill semantically"""

        if not skills:
            return {
                "overall_score": 0,
                "selected": False,
                "skill_scores": {},
                "skill_reasoning": {},
                "missing_skills": [],
                "reasoning": "No skills provided for analysis.",
                "strengths": [],
                "improvement_areas": []
            }

        vectorstore = self.create_rag_vectorstore(resume_text)

        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        # Analyze ALL skills in a single LLM call to minimize token usage
        batch_results = self.analyze_skills_batch(resume_text, skills)

        for skill in skills:
            # Find matching key (case-insensitive)
            matched = None
            for key in batch_results:
                if key.lower().strip() == skill.lower().strip():
                    matched = key
                    break
            if not matched:
                for key in batch_results:
                    if skill.lower() in key.lower() or key.lower() in skill.lower():
                        matched = key
                        break

            if matched and isinstance(batch_results[matched], dict):
                score = min(int(batch_results[matched].get("score", 0)), 10)
                reasoning_text = batch_results[matched].get("reasoning", "No reasoning provided.")
            else:
                score = 0
                reasoning_text = "Could not analyze this skill."

            skill_scores[skill] = score
            skill_reasoning[skill] = reasoning_text
            total_score += score
            if score <= 5:
                missing_skills.append(skill)

        overall_score = int((total_score / (10 * len(skills)) * 100))
        selected = overall_score >= self.cutoff_score

        reasoning = "Candidate evaluted based on explicit resume content using semantic similarity and clear numeric scoring."
        strengths = [skill for skill, score in skill_scores.items() if score >= 7]
        improvement_areas = missing_skills if not selected else []

        
        self.analysis_strengths = strengths

        return {
            "overall_score": overall_score,
            "selected": selected,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "missing_skills": missing_skills,
            "reasoning": reasoning,
            "strengths": strengths,
            "improvement_areas": improvement_areas
        }

    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None):
        """Analyze a resume againt role requirements or custom JD"""

        self.resume_text = self.extract_text_from_file(resume_file)

        if not self.resume_text or not self.resume_text.strip():
            raise ValueError("Could not extract text from the resume. Please ensure the PDF is not scanned/image-based and contains selectable text.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding='utf-8') as tmp:
            tmp.write(self.resume_text)
            self.resume_file_path = tmp.name

        self.rag_vectorstore = self.create_rag_vectorstore(self.resume_text)

        if custom_jd:
            self.jd_text = self.extract_text_from_file(custom_jd)

            if not self.jd_text or not self.jd_text.strip():
                raise ValueError("Could not extract text from the Job Description PDF. Please ensure it contains selectable text, or try a .txt file instead.")

            self.extracted_skill = self.extract_skills_from_jd(self.jd_text)

            if not self.extracted_skill:
                raise ValueError("Could not extract any skills from the Job Description. Please check if the JD file has valid content.")

            self.analysis_result = self.semantic_skill_analysis(self.resume_text, self.extracted_skill)

        elif role_requirements:
            self.extracted_skill = role_requirements
            self.analysis_result = self.semantic_skill_analysis(self.resume_text, role_requirements)

        else:
            raise ValueError("No role requirements or job description provided for analysis.")

        
        if self.analysis_result and "missing_skills" in self.analysis_result:
            if self.analysis_result["missing_skills"]:
                self.analyze_resume_weakness()

                self.analysis_result["detailed_weaknesses"] = self.resume_weakness

        return self.analysis_result

    def ask_question(self, question):
        """Ask a question about the resume analysis result and get an answer based on the RAG vectorstore and analysis data"""
        if not self.rag_vectorstore:
            return "Please analyze a resume first."
        retriever = self.rag_vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(model="llama-3.1-8b-instant", api_key=self.api_key),
            retriever=retriever,
            return_source_documents=False,
        )
        response = qa_chain.run(question)
        return response

    def generate_interview_questions(self, question_types, difficulty, num_questions, target_role=""):
        """Generate role-based interview questions using prompt engineering"""
        if not self.resume_text or not self.extracted_skill:
            return []

        try:
            llm = ChatGroq(model="llama-3.1-8b-instant", api_key=self.api_key)

            # Role-based prompt engineering
            role_context = ""
            if target_role:
                role_context = f"""
    Target Role: {target_role}
    You are an expert interviewer for the {target_role} position.
    Tailor all questions specifically to what a {target_role} would need to know and demonstrate.
    Focus on role-specific scenarios, tools, and industry best practices relevant to {target_role}.
    """

            difficulty_guidance = {
                "Easy": "Ask foundational questions that test basic understanding of concepts. Suitable for junior-level candidates.",
                "Medium": "Ask intermediate questions that require practical experience and understanding of trade-offs. Suitable for mid-level candidates.",
                "Hard": "Ask advanced questions that require deep expertise, system design thinking, and real-world problem solving. Suitable for senior-level candidates.",
            }.get(difficulty, "Ask a mix of difficulty levels.")

            context = f"""
Resume Content:
{self.resume_text[:2000]}

Skills to focus on: {', '.join(self.extracted_skill)}
Strengths: {', '.join(self.analysis_strengths)}
Areas for improvement: {', '.join(self.analysis_result.get("improvement_areas", []))}
{role_context}
"""

            prompt = f"""
You are a senior technical interviewer. {difficulty_guidance}

{context}

Generate exactly {num_questions} interview questions that are:
1. Focused on assessing the candidate's proficiency in the identified skills.
2. Aligned with the candidate's strengths to further explore those areas.
3. Targeted at understanding how the candidate can improve in their weaker areas.
4. A mix of the following question types: {', '.join(question_types)}.

Difficulty level: {difficulty}

Provide the questions in a numbered list format.
"""

            response = self._retry_with_backoff(llm.invoke, prompt)
            questions_text = response.content

            questions = re.findall(r"\d+\.\s*(.*)", questions_text)
            return questions

        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []

    def improve_resume(self, improvement_areas, target_role=""):
        """Generate specific suggestions to improve the resume based on identified improvement areas and target role"""

        if not self.resume_text or not improvement_areas:
            return {}

        try:
            llm = ChatGroq(model="llama-3.1-8b-instant", api_key=self.api_key)

            prompt = f"""
            Based on the resume content and identified improvement areas, provide specific and actionable suggestions to enhance the resume for better alignment with the target role.

            Resume Content:
            {self.resume_text[:2000]}

            Identified Improvement Areas: {', '.join(improvement_areas)}

            Target Role: {target_role}

            For each improvement area, suggest:
            1. What specific content or skills should be added or emphasized in the resume?
            2. How can the candidate demonstrate proficiency in this area more effectively?
            3. Provide an example bullet point that could be added to the resume to showcase improvement in this area.

            Format your response as a JSON object where each key is an improvement area and the value is an object containing the suggestions.
            """

            response = self._retry_with_backoff(llm.invoke, prompt)
            suggestions_text = response.content

            json_match = re.search(r'\{.*\}', suggestions_text, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group(0))
            else:
                suggestions = json.loads(suggestions_text)
            self.improvement_suggestions = suggestions
            return suggestions

        except Exception as e:
            print(f"Error generating resume improvement suggestions: {e}")
            return {}

    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'resume_file_path') and os.path.exists(self.resume_file_path):
            os.remove(self.resume_file_path)
