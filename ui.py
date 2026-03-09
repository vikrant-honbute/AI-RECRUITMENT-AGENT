import streamlit as st


def setup_page():
    """Configure page settings and custom CSS"""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .score-card {
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .score-high {
            background-color: #d4edda;
            border: 2px solid #28a745;
        }
        .score-medium {
            background-color: #fff3cd;
            border: 2px solid #ffc107;
        }
        .score-low {
            background-color: #f8d7da;
            border: 2px solid #dc3545;
        }
        .skill-tag {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .skill-strong {
            background-color: #d4edda;
            color: #155724;
        }
        .skill-weak {
            background-color: #f8d7da;
            color: #721c24;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the main application header"""
    st.markdown("""
    <div class="main-header">
        <p>Upload a resume, select a target role, and get AI-powered analysis, Q&A, interview prep, and improvement suggestions.</p>
    </div>
    """, unsafe_allow_html=True)


def setup_sidebar():
    """Setup the sidebar with API key input and app settings"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        api_key = st.text_input(
            "🔑 GROQ API Key",
            type="password",
            placeholder="Enter your GROQ API key...",
            help="Get your free API key from https://console.groq.com"
        )

        st.markdown("---")
        st.header("📋 About")
        st.markdown("""
        **AI Recruitment Agent** uses LLMs and RAG to:
        - Analyze resumes against job roles
        - Answer questions about resumes
        - Generate interview questions
        - Suggest resume improvements
        """)

        if api_key:
            st.success("✅ API Key provided")
        else:
            st.warning("⚠️ Please enter your API key")

    return {"api_key": api_key}


def create_tabs():
    """Create the main application tabs"""
    tabs = st.tabs([
        "📄 Resume Analysis",
        "❓ Resume Q&A",
        "🎤 Interview Questions",
        "✨ Resume Improvements",
    ])
    return tabs


def role_selection_section(role_requirements):
    """Display role selection and optional custom JD upload"""
    st.subheader("🎯 Select Target Role")

    col1, col2 = st.columns(2)

    with col1:
        role = st.selectbox(
            "Choose a predefined role",
            options=list(role_requirements.keys()),
            help="Select a role to match the resume against its required skills."
        )

        if role:
            with st.expander(f"Skills for {role}", expanded=False):
                skills = role_requirements[role]
                cols = st.columns(3)
                for i, skill in enumerate(skills):
                    cols[i % 3].markdown(f"• {skill}")

    with col2:
        st.markdown("**Or upload a custom Job Description**")
        custom_jd = st.file_uploader(
            "Upload JD (PDF or TXT)",
            type=["pdf", "txt"],
            key="jd_upload",
            help="Upload a custom job description to extract skills from."
        )
        if custom_jd:
            st.info(f"📎 Custom JD uploaded: {custom_jd.name}")

    return role, custom_jd


def resume_upload_section():
    """Display the resume upload section"""
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or TXT)",
        type=["pdf", "txt"],
        key="resume_upload",
        help="Supported formats: PDF, TXT"
    )
    if uploaded_file:
        st.success(f"✅ Resume uploaded: {uploaded_file.name}")
    return uploaded_file


def _build_analysis_report(results):
    """Build a text report from analysis results"""
    lines = ["RESUME ANALYSIS REPORT", "=" * 40, ""]
    lines.append(f"Overall Score: {results.get('overall_score', 0)}%")
    lines.append(f"Status: {'Selected' if results.get('selected') else 'Not Selected'}")
    lines.append("")

    lines.append("SKILL SCORES:")
    lines.append("-" * 30)
    for skill, score_val in results.get("skill_scores", {}).items():
        reasoning = results.get("skill_reasoning", {}).get(skill, "")
        lines.append(f"  {skill}: {score_val}/10 - {reasoning}")
    lines.append("")

    strengths = results.get("strengths", [])
    if strengths:
        lines.append("STRENGTHS: " + ", ".join(strengths))
        lines.append("")

    missing = results.get("missing_skills", [])
    if missing:
        lines.append("WEAK/MISSING SKILLS: " + ", ".join(missing))
        lines.append("")

    detailed = results.get("detailed_weaknesses", [])
    if detailed:
        lines.append("DETAILED WEAKNESS ANALYSIS:")
        lines.append("-" * 30)
        for w in detailed:
            lines.append(f"  Skill: {w.get('skill', '')}")
            lines.append(f"  Issue: {w.get('weakness', '')}")
            for sug in w.get("improvement_suggestions", []):
                lines.append(f"    - {sug}")
            example = w.get("example_addition", "")
            if example:
                lines.append(f"  Example: {example}")
            lines.append("")

    return "\n".join(lines)


def display_analysis_results(results):
    """Display the full analysis results"""
    if not results:
        return

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Overall score card
    score = results.get("overall_score", 0)
    selected = results.get("selected", False)

    if score >= 75:
        score_class = "score-high"
        emoji = "🟢"
    elif score >= 50:
        score_class = "score-medium"
        emoji = "🟡"
    else:
        score_class = "score-low"
        emoji = "🔴"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="score-card {score_class}">
            <h1>{emoji} {score}%</h1>
            <p><b>Overall Score</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        status = "✅ Selected" if selected else "❌ Not Selected"
        status_color = "score-high" if selected else "score-low"
        st.markdown(f"""
        <div class="score-card {status_color}">
            <h2>{status}</h2>
            <p><b>Candidate Status</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        num_strengths = len(results.get("strengths", []))
        num_missing = len(results.get("missing_skills", []))
        st.markdown(f"""
        <div class="score-card score-medium">
            <h2>{num_strengths} 💪 / {num_missing} ⚠️</h2>
            <p><b>Strengths / Gaps</b></p>
        </div>
        """, unsafe_allow_html=True)

    # Reasoning
    st.markdown(f"**Reasoning:** {results.get('reasoning', 'N/A')}")

    # Skill scores breakdown
    st.markdown("### 📈 Skill-by-Skill Breakdown")
    skill_scores = results.get("skill_scores", {})
    skill_reasoning = results.get("skill_reasoning", {})

    if skill_scores:
        for skill, score_val in skill_scores.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(min(score_val / 10, 1.0), text=f"**{skill}**")
            with col2:
                st.markdown(f"**{score_val}/10**")

            reasoning_text = skill_reasoning.get(skill, "")
            if reasoning_text:
                with st.expander(f"Reasoning for {skill}", expanded=False):
                    st.write(reasoning_text)

    # Strengths and weaknesses
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown("### 💪 Strengths")
        strengths = results.get("strengths", [])
        if strengths:
            for s in strengths:
                st.markdown(f'<span class="skill-tag skill-strong">✅ {s}</span>', unsafe_allow_html=True)
        else:
            st.info("No strong skills identified.")

    with col_w:
        st.markdown("### ⚠️ Missing / Weak Skills")
        missing = results.get("missing_skills", [])
        if missing:
            for m in missing:
                st.markdown(f'<span class="skill-tag skill-weak">❌ {m}</span>', unsafe_allow_html=True)
        else:
            st.success("No major skill gaps found!")

    # Detailed weaknesses
    detailed = results.get("detailed_weaknesses", [])
    if detailed:
        st.markdown("### 🔍 Detailed Weakness Analysis")
        for w in detailed:
            with st.expander(f"📌 {w.get('skill', 'Unknown Skill')}", expanded=False):
                st.markdown(f"**Issue:** {w.get('weakness', 'N/A')}")

                suggestions = w.get("improvement_suggestions", [])
                if suggestions:
                    st.markdown("**Suggestions:**")
                    for sug in suggestions:
                        st.markdown(f"- {sug}")

                example = w.get("example_addition", "")
                if example:
                    st.markdown("**Example bullet point to add:**")
                    st.code(example, language=None)

    # Download Analysis Report
    st.markdown("---")
    report = _build_analysis_report(results)
    st.download_button(
        label="📥 Download Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        key="download_analysis_report"
    )


def resume_qa_section(has_resume, ask_questions_fn):
    """Display the resume Q&A section"""
    st.subheader("❓ Ask Questions About the Resume")

    if not has_resume:
        st.warning("Please analyze a resume first to enable the Q&A section.")
        return

    st.markdown("Ask any question about the analyzed resume and get AI-powered answers.")

    # Chat history
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    question = st.text_input(
        "Your question",
        placeholder="e.g., What is the candidate's experience with Python?",
        key="qa_input"
    )

    if st.button("Ask", type="primary", key="qa_ask_btn"):
        if question:
            response = ask_questions_fn(question)
            if response:
                st.session_state.qa_history.append({"question": question, "answer": response})

    # Display Q&A history
    if st.session_state.qa_history:
        st.markdown("### 💬 Q&A History")
        for i, item in enumerate(reversed(st.session_state.qa_history)):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                st.markdown("---")

    # Suggested questions
    with st.expander("💡 Suggested Questions", expanded=False):
        suggested = [
            "What is the candidate's strongest technical skill?",
            "Does the candidate have leadership experience?",
            "What projects has the candidate worked on?",
            "How many years of experience does the candidate have?",
            "Does the candidate have cloud computing experience?",
        ]
        for sq in suggested:
            st.markdown(f"• {sq}")


def interview_question_generation_section(has_resume, generate_questions_func, role_list=None):
    """Display the interview question generation section with role-based options"""
    st.subheader("🎤 Generate Interview Questions")

    if not has_resume:
        st.warning("Please analyze a resume first to enable interview question generation.")
        return

    st.markdown("Generate role-specific interview questions based on the resume analysis.")

    col1, col2 = st.columns(2)
    with col1:
        question_types = st.multiselect(
            "Question Types",
            options=["Technical", "Behavioral", "Situational", "Problem-Solving", "System Design", "Cultural Fit"],
            default=["Technical", "Behavioral"],
            help="Select the types of questions to generate."
        )

    with col2:
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            value="Medium",
            help="Set the difficulty level of the questions."
        )

    col3, col4 = st.columns(2)
    with col3:
        num_questions = st.slider(
            "Number of Questions",
            min_value=3,
            max_value=15,
            value=5,
            help="How many questions to generate."
        )
    with col4:
        target_role = ""
        if role_list:
            target_role = st.selectbox(
                "Target Role for Questions",
                options=[""] + role_list,
                help="Optionally select a role to tailor questions."
            )

    if st.button("🎯 Generate Questions", type="primary", key="gen_q_btn"):
        if question_types:
            questions = generate_questions_func(question_types, difficulty, num_questions, target_role)
            if questions:
                st.session_state.generated_questions = questions
            else:
                st.error("Failed to generate questions. Please try again.")
        else:
            st.warning("Please select at least one question type.")

    # Display generated questions
    if "generated_questions" in st.session_state and st.session_state.generated_questions:
        st.markdown("### 📝 Generated Interview Questions")
        qa_items = st.session_state.generated_questions
        for i, item in enumerate(qa_items, 1):
            if isinstance(item, dict):
                st.markdown(f"**Q{i}.** {item.get('question', '')}")
                answer = item.get('answer', '')
                if answer:
                    with st.expander(f"💡 Answer for Q{i}", expanded=True):
                        st.markdown(answer)
            else:
                st.markdown(f"**Q{i}.** {item}")

        # Download interview questions report
        report_lines = []
        for i, item in enumerate(qa_items, 1):
            if isinstance(item, dict):
                report_lines.append(f"Q{i}: {item.get('question', '')}")
                ans = item.get('answer', '')
                if ans:
                    report_lines.append(f"Answer: {ans}")
            else:
                report_lines.append(f"Q{i}: {item}")
            report_lines.append("")
        interview_report = "\n".join(report_lines)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            with st.expander("📋 Copy All Q&A"):
                st.text_area("Questions & Answers (copy from here)", value=interview_report, height=300, key="copy_questions")
        with col_dl2:
            st.download_button(
                label="📥 Download Interview Q&A",
                data=interview_report,
                file_name="interview_questions.txt",
                mime="text/plain",
                key="download_interview_qa"
            )


def resume_improvement_section(has_resume, improvement_fn, analysis_results=None):
    """Display the resume improvement section"""
    st.subheader("✨ Resume Improvement Suggestions")

    if not has_resume:
        st.warning("Please analyze a resume first to enable resume improvement suggestions.")
        return

    st.markdown("Get specific, actionable suggestions to strengthen your resume.")

    # Pre-populate improvement areas from analysis
    default_areas = []
    if analysis_results:
        default_areas = analysis_results.get("missing_skills", []) + analysis_results.get("improvement_areas", [])
        # Deduplicate
        default_areas = list(dict.fromkeys(default_areas))

    improvement_areas = st.multiselect(
        "Select areas to improve",
        options=default_areas if default_areas else ["Add areas manually below"],
        default=default_areas[:5] if default_areas else [],
        help="Select the skills/areas you want improvement suggestions for."
    )

    custom_areas = st.text_input(
        "Or add custom areas (comma-separated)",
        placeholder="e.g., Docker, System Design, Leadership",
        key="custom_improvement_areas"
    )

    if custom_areas:
        extra = [a.strip() for a in custom_areas.split(",") if a.strip()]
        improvement_areas = list(dict.fromkeys(improvement_areas + extra))

    target_role = st.text_input(
        "Target role (optional)",
        placeholder="e.g., Senior Backend Engineer",
        key="improvement_target_role"
    )

    if st.button("🚀 Get Improvement Suggestions", type="primary", key="improve_btn"):
        if improvement_areas:
            suggestions = improvement_fn(improvement_areas, target_role)
            if suggestions:
                st.session_state.improvement_suggestions = suggestions
            else:
                st.error("Failed to generate suggestions. Please try again.")
        else:
            st.warning("Please select or enter at least one area to improve.")

    # Display improvement suggestions
    if "improvement_suggestions" in st.session_state and st.session_state.improvement_suggestions:
        st.markdown("### 📋 Improvement Suggestions")
        suggestions = st.session_state.improvement_suggestions

        for area, details in suggestions.items():
            with st.expander(f"📌 {area}", expanded=True):
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, list):
                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                            for item in value:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.write(details)

        # Download improvement report
        report_lines = ["RESUME IMPROVEMENT REPORT", "=" * 40, ""]
        for area, details in suggestions.items():
            report_lines.append(f"Area: {area}")
            report_lines.append("-" * 30)
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, list):
                        report_lines.append(f"  {key.replace('_', ' ').title()}:")
                        for item in value:
                            report_lines.append(f"    - {item}")
                    else:
                        report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                report_lines.append(f"  {details}")
            report_lines.append("")
        improvement_report = "\n".join(report_lines)

        st.download_button(
            label="📥 Download Improvement Report",
            data=improvement_report,
            file_name="resume_improvement_report.txt",
            mime="text/plain",
            key="download_improvement_report"
        )
