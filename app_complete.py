import os
import re
import time
import pdfplumber
import docx
import spacy
import streamlit as st
from typing import TypedDict, Annotated, List, Dict
import operator
from spacy.matcher import Matcher
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD1PwlK3OPsLE38mziLavR9ocMtDEYqPn4"

def safe_read_file(file_path: str) -> str:
    """
    Safely read a file with multiple encoding attempts and error handling.
    """
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'latin1', 'ascii']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding}: {e}")
            continue
    
    # Final fallback: read as binary and decode with error handling
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            return raw_data.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading file as binary: {e}")
        return ""

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class ParsedResume(BaseModel):
    """Schema for the data extracted from a resume."""
    name: str
    filename: str
    email: str
    mobile_number: str
    skills: List[str]
    education: str
    work_experience_years: float
    

class SimulatedResumeParser:
    def __init__(self, resume_path: str, skills_file: str = None, custom_regex: str = None):
        self.resume_path = resume_path
        self.__skills_file = skills_file
        self.__custom_regex = custom_regex
        self.__text = self.__extract_text(resume_path)

        # Load spaCy NLP
        self.__nlp = spacy.load("en_core_web_sm")
        self.__doc = self.__nlp(self.__text)
        self.__matcher = Matcher(self.__nlp.vocab)

        # Extract details
        self.__details = self.__get_basic_details()

    def __extract_text(self, resume_path: str) -> str:
        """Extract text from TXT, PDF, or DOCX files."""
        ext = os.path.splitext(resume_path)[1].lower()

        if ext == ".pdf":
            text = ""
            try:
                with pdfplumber.open(resume_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text
            except Exception as e:
                print(f"Error reading PDF {resume_path}: {e}")
                return ""

        elif ext == ".docx":
            text = ""
            try:
                doc = docx.Document(resume_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
                return text
            except Exception as e:
                print(f"Error reading DOCX {resume_path}: {e}")
                return ""

        else:  # fallback for plain text
            try:
                return safe_read_file(resume_path)
            except Exception as e:
                print(f"Error reading text file {resume_path}: {e}")
                return ""

    def __extract_name(self, text: str, doc) -> str:
        """Extract name using multiple strategies for better accuracy."""
        # Strategy 1: Look for name patterns at the beginning of the document
        lines = text.split('\n')
        first_few_lines = [line.strip() for line in lines[:5] if line.strip()]
        
        # Strategy 2: Use filename if it contains a name pattern
        filename = os.path.basename(self.resume_path)
        filename_name = self.__extract_name_from_filename(filename)
        if filename_name:
            return filename_name
        
        # Strategy 3: Look for name patterns in first few lines
        for line in first_few_lines:
            # Skip lines that are likely headers/titles
            if any(word.lower() in line.lower() for word in ['resume', 'curriculum', 'cv', 'vitae', 'profile']):
                continue
            
            # Look for name patterns (2-4 words, proper case, no numbers)
            name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$)'
            match = re.match(name_pattern, line.strip())
            if match:
                potential_name = match.group(1).strip()
                # Validate it's not a common header word
                if not any(word.lower() in potential_name.lower() for word in ['address', 'phone', 'email', 'objective', 'summary']):
                    return potential_name
        
        # Strategy 4: Use spaCy NER but with better filtering
        person_names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Filter out common false positives
                name_text = ent.text.strip()
                if len(name_text.split()) >= 2 and len(name_text.split()) <= 4:
                    # Check if it's not a file extension or random text
                    if not any(ext in name_text.lower() for ext in ['.png', '.jpg', '.pdf', '.doc', '.txt']):
                        if re.match(r'^[A-Za-z\s\.\-\']+$', name_text):  # Only letters, spaces, dots, hyphens, apostrophes
                            person_names.append(name_text)
        
        # Return the first reasonable person name found
        if person_names:
            return person_names[0]
        
        return None
    
    def __extract_name_from_filename(self, filename: str) -> str:
        """Extract name from filename if it follows common naming patterns."""
        # Remove extension
        name_part = os.path.splitext(filename)[0]
        
        # Common patterns: "FirstName_LastName", "FirstName LastName", "LastName_FirstName"
        # Clean up underscores and common resume words
        cleaned = re.sub(r'[_\-]', ' ', name_part)
        cleaned = re.sub(r'(?i)\b(resume|cv|curriculum|vitae)\b', '', cleaned).strip()
        
        # Check if it looks like a name (2-4 words, proper case or can be title-cased)
        words = cleaned.split()
        if 2 <= len(words) <= 4:
            # Title case the words
            name_candidate = ' '.join(word.capitalize() for word in words if word.isalpha())
            if len(name_candidate.split()) >= 2:
                return name_candidate
        
        return None

    def __get_basic_details(self) -> Dict:
        """Extract details using spaCy NLP + regex (like original ResumeParser)."""
        text = self.__text
        doc = self.__doc
        details = {}

        name = self.__extract_name(text, doc)
        details["name"] = name if name else "Unknown Candidate"

        email = re.search(r"[\w\.-]+@[\w\.-]+", text)
        details["email"] = email.group(0) if email else "N/A"

        mobile = re.search(self.__custom_regex or r"(\+?\d[\d\-\s]{8,}\d)", text)
        details["mobile_number"] = mobile.group(0) if mobile else "N/A"

        skill_keywords = []
        if self.__skills_file and os.path.exists(self.__skills_file):
            with open(self.__skills_file, "r", encoding="utf-8") as f:
                skill_keywords = [line.strip() for line in f if line.strip()]
        else:
            skill_keywords = ["Python", "SQL", "Spark", "AWS", "Kubernetes",
                              "SEO", "Marketing", "JavaScript", "React", "Node.js"]

        found_skills = [s for s in skill_keywords if re.search(r"\b" + re.escape(s) + r"\b", text, re.IGNORECASE)]
        details["skills"] = found_skills

        edu_keywords = ["B.S.", "B.Sc", "M.S.", "M.Sc", "PhD", "Bachelor", "Master", "MBA", "B.Tech", "M.Tech"]
        education = None
        for token in doc:
            for kw in edu_keywords:
                if kw.lower() in token.text.lower():
                    education = token.sent.text
                    break
        if not education:
            education = "Degree Placeholder"
        details["education"] = education

        years_exp = 0.0
        exp_match = re.search(r"(\d+)\s+years", text, re.IGNORECASE)
        if exp_match:
            years_exp = float(exp_match.group(1))
        else:
            years_exp = 1.0
        details["work_experience_years"] = years_exp

        details["filename"] = os.path.basename(self.resume_path)
        return details

    def get_extracted_data(self) -> ParsedResume:
        """Return extracted resume data as a Pydantic model."""
        return ParsedResume(**self.__details)

@tool
def generate_onboarding_plan(candidate_data: dict, role: str) -> str:
    """
    Uses an LLM to create a structured 30-60-90 day onboarding plan
    based on the candidate's skills and the specific job role.
    """
    prompt = f"""
    Create a detailed, actionable 30-60-90 day onboarding plan for the new hire:
    - Name: {candidate_data.get('name')}
    - Skills: {', '.join(candidate_data.get('skills', []))}
    - Target Role: {role}

    Structure the output clearly with the 30, 60, and 90-day milestones.
    """

    response = llm.invoke(prompt)
    return response.content

@tool
def process_and_score_resume(resume_filepath: str, job_requirements: str) -> dict:
    """
    Parses a single resume file, extracts key details using the parser,
    and then calculates a similarity score against the job_requirements using LLM.
    """
    print(f"  > Processing {os.path.basename(resume_filepath)}...")

    # 1. Use the simulated parser
    parser = SimulatedResumeParser(resume_filepath)
    parsed_data = parser.get_extracted_data().dict()

    # 2. Use LLM to score the relevance
    prompt = f"""
    You are a professional HR screener. Score the candidate's relevance for the job
    based on their extracted skills and experience.

    JOB REQUIREMENTS: {job_requirements}
    CANDIDATE PROFILE:
    - Skills: {', '.join(parsed_data['skills'])}
    - Experience: {parsed_data['work_experience_years']} years.
    - Education: {parsed_data['education']}

    Provide ONLY a single float score between 0.0 (not relevant) and 1.0 (perfect fit)
    in your response. Do not include any other text or explanation.
    """

    score = 0.5 # Default score if LLM extraction fails

    try:
        response = llm.invoke(prompt)
        # Attempt to find the score float in the response content
        score_match = re.search(r'(\d\.\d+)', response.content.strip())
        if score_match:
            score = float(score_match.group(1))
    except Exception as e:
        print(f"LLM scoring failed, using default: {e}")

    time.sleep(0.1) # Simulate complex operation time

    return {
        "candidate_data": parsed_data,
        "relevance_score": score,
        "filename": os.path.basename(resume_filepath)
    }

@tool
def retrieve_policy_answer(question: str, policy_file_path: str) -> str:
    """
    Simulates a RAG tool by searching a local policy file for an answer.
    """
    print(f"  > Searching policy documents for: '{question}'")

    try:
        policy_text = safe_read_file(policy_file_path)
        if not policy_text:
            return f"Error: Could not read policy documents ({policy_file_path}). The file may be corrupted or in an unsupported format."
        print(f"  > Successfully read policy file")
            
    except FileNotFoundError:
        return f"Error: Policy documents ({policy_file_path}) were not found. Cannot answer policy questions."
    except Exception as e:
        return f"Error reading policy file: {str(e)}. Please check the file format and try again."

    # Use LLM to perform Q/A over the retrieved text (simulated RAG context)
    prompt = f"""
    Using ONLY the following policy text, answer the user's question concisely.
    If the answer is not available in the text, state that.

    POLICY TEXT:
    ---
    {policy_text}
    ---

    QUESTION: {question}
    """

    response = llm.invoke(prompt)
    return response.content

class SimpleMemory:
    """Simple short-term memory management for the workflow."""
    
    @staticmethod
    def initialize_memory() -> dict:
        """Initialize a simple memory structure."""
        return {
            "session_memory": {
                "session_id": f"session_{int(time.time())}",
                "session_start": time.time(),
                "actions_performed": []
            },
            "workflow_history": []
        }
    
    @staticmethod
    def add_action(state: dict, action: str, success: bool, execution_time: float):
        """Add an action to the session memory."""
        action_record = {
            "action": action,
            "timestamp": time.time(),
            "success": success,
            "execution_time": execution_time
        }
        
        # Add to session memory
        if "session_memory" not in state:
            state["session_memory"] = {"actions_performed": []}
        
        state["session_memory"]["actions_performed"].append(action_record)
        
        # Keep only last 10 actions in session
        if len(state["session_memory"]["actions_performed"]) > 10:
            state["session_memory"]["actions_performed"] = state["session_memory"]["actions_performed"][-10:]
        
        # Add to workflow history (basic)
        if "workflow_history" not in state:
            state["workflow_history"] = []
        
        state["workflow_history"].append(action_record)
        
        # Keep only last 20 in history
        if len(state["workflow_history"]) > 20:
            state["workflow_history"] = state["workflow_history"][-20:]

class AgentState(TypedDict):
    """Represents the state of our multi-agent system with simple short-term memory."""
    input: str
    chat_history: Annotated[List[HumanMessage], operator.add]
    job_requirements: str
    processed_resumes: List[dict]
    best_candidate: dict
    onboarding_plan: str
    final_output: str
    next_action: str
    resume_directory: str
    policy_file_path: str
    
    # Simple Memory System
    session_memory: dict  # Current session memory with recent actions
    workflow_history: List[dict]  # Basic history of workflow executions (last 10)

class TalentScout:
    """Agent for parsing, scoring, and selecting the best resume."""

    def run_screening(self, state: AgentState):
        """Scans the directory and runs the scoring tool for all files."""
        print(f"\n[TalentScout] Starting screening in directory: {state['resume_directory']}")

        # Filter files to only include common resume types
        resume_files = [os.path.join(state['resume_directory'], f)
                        for f in os.listdir(state['resume_directory'])
                        if f.lower().endswith(('.txt', '.pdf', '.docx'))]

        if not resume_files:
            return {"final_output": f"Error: No .txt, .pdf, or .docx files found in the specified directory: {state['resume_directory']}"}

        all_results = []
        for file_path in resume_files:
            try:
                # Invoke the tool for each resume
                result = process_and_score_resume.invoke({
                    "resume_filepath": file_path,
                    "job_requirements": state['job_requirements']
                })
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Select the best candidate
        best_candidate = max(all_results, key=lambda x: x['relevance_score'])

        # Sort all results by score (highest first)
        sorted_results = sorted(all_results, key=lambda x: x['relevance_score'], reverse=True)

        # Build output showing all candidates
        output_message = f"Screening complete! **{len(all_results)}** resumes processed.\n\n"
        output_message += "**All Candidates (sorted by score):**\n\n"

        for idx, result in enumerate(sorted_results, 1):
            output_message += (
                f"{idx}. **{result['candidate_data']['name']}**\n"
                f"   - File: {result['filename']}\n"
                f"   - Score: {result['relevance_score']:.2f}/1.00\n"
                f"   - Skills: {', '.join(result['candidate_data']['skills'])}\n"
                f"   - Experience: {result['candidate_data']['work_experience_years']} years\n\n"
            )

        output_message += f"\n**Top Candidate Selected:** {best_candidate['candidate_data']['name']}\n"
        output_message += "**Action:** Do you want to proceed with **Onboarding Plan** creation?"

        return {
            "processed_resumes": all_results,
            "best_candidate": best_candidate['candidate_data'],
            "final_output": output_message,
            "next_action": "SCREENING_COMPLETE"
        }

class Onboarder:
    """Generate onboarding plan using LLM"""
    @staticmethod
    def create_plan(state: AgentState):
        candidate = state['best_candidate']
        if not candidate:
            return {"final_output": "No best candidate selected!", "next_action": ""}

        # Read company policies text
        try:
            policies_text = safe_read_file(state['policy_file_path'])
            if not policies_text:
                return {"final_output": f"Error: Could not read policy file. The file may be corrupted or in an unsupported format.", "next_action": ""}
                    
        except Exception as e:
            return {"final_output": f"Error reading policy file: {e}", "next_action": ""}

        # LLM prompt
        prompt = f"""
You are an HR automation assistant. Generate a comprehensive onboarding plan
for a new employee based on the following information.

Candidate Details:
- Name: {candidate['name']}
- Email: {candidate['email']}
- Skills: {', '.join(candidate['skills'])}
- Education: {candidate['education']}
- Work Experience (years): {candidate['work_experience_years']}

Job Requirements:
{state['job_requirements']}

Company Policies (summarized or relevant sections):
{policies_text[:3000]}  # use first 3000 chars to avoid hitting token limits

Requirements:
- Provide a step-by-step onboarding plan.
- Include tasks such as document submission, orientation, setup, team introduction, and first project assignment.
- Reference company policies where relevant.
- Keep it professional and clear.

Output the onboarding plan in readable steps.
"""

        # Call the LLM
        try:
            response = llm.invoke(prompt)
            plan_content = response.content # Access the content attribute
        except Exception as e:
            return {"final_output": f"Error generating onboarding plan with LLM: {e}", "next_action": ""}

        final_message = f"--- ✅ Onboarding Plan Generated ✅ ---\n{plan_content}"

        return {
            "onboarding_plan": plan_content,
            "final_output": final_message,
            "next_action": "PolicyQA"
        }

class PolicyQA:
    """Agent for answering policy questions using RAG."""

    def answer_question(self, state: AgentState):
        """Invokes the RAG tool to get the policy answer."""
        print(f"\n[PolicyQA] Answering question...")

        answer = retrieve_policy_answer.invoke({"question": state['input'], "policy_file_path": state['policy_file_path']})

        final_message = f"--- 📄 **Policy Answer** 📄 ---\n{answer}"

        return {
            "final_output": final_message,
            "next_action": "QA_COMPLETE"
        }

def execute_tool_call(state: AgentState):
    """Executes the specific agent logic based on user selection."""
    action = state['next_action']

    if action == "TalentScout":
        return TalentScout().run_screening(state)
    elif action == "PolicyQA":
        return PolicyQA().answer_question(state)
    elif action == "Onboarder":
        return Onboarder().create_plan(state)

    return {"final_output": f"Error: Unknown tool or missing logic for action: {action}"}

def build_user_driven_app():
    """Builds a workflow that executes a single tool based on user input and then ends."""
    workflow = StateGraph(AgentState)

    workflow.add_node("ExecuteTool", execute_tool_call)
    workflow.set_entry_point("ExecuteTool")
    workflow.add_edge("ExecuteTool", END)

    return workflow.compile()

# --- Streamlit App ---

st.set_page_config(page_title="HR Automation Suite", layout="wide")

st.title("🤖 HR Automation Suite")

# Initialize session state for the workflow and its state
if 'workflow' not in st.session_state:
    st.session_state.workflow = build_user_driven_app()

if 'current_state' not in st.session_state:
    # Initialize memory system
    memory_data = SimpleMemory.initialize_memory()
    
    st.session_state.current_state = AgentState(
        input="",
        chat_history=[],
        job_requirements="",
        processed_resumes=[],
        best_candidate={},
        onboarding_plan="",
        final_output="",
        next_action="ExecuteTool",
        resume_directory="",
        policy_file_path="",
        **memory_data  # Unpack all memory components
    )

# --- Input Fields ---

st.header("Configuration")
policy_file = st.text_input("Enter the path to your company policies file:",
                            value=st.session_state.current_state.get('policy_file_path', ''))
if policy_file and os.path.isfile(policy_file):
    st.session_state.current_state['policy_file_path'] = policy_file
elif policy_file and not os.path.isfile(policy_file):
    st.error("Invalid policy file path.")

# --- Tool Selection and Execution ---

st.header("Run HR Tools")

tool_choice = st.radio("Choose a tool to run:",
                       ('TalentScout (Screen Resumes)',
                        'Onboarder (Generate Onboarding Plan)',
                        'PolicyQA (Answer Policy Questions)'))

# Collect inputs based on tool choice (outside of button click)
resume_directory = ""
job_requirements = ""
policy_question = ""

if tool_choice == 'TalentScout (Screen Resumes)':
    resume_directory = st.text_input("Enter the path to your resume folder:",
                                    value=st.session_state.current_state.get('resume_directory', ''))
    job_requirements = st.text_area("Enter the Job Requirements:",
                                   value=st.session_state.current_state.get('job_requirements', ''))

elif tool_choice == 'PolicyQA (Answer Policy Questions)':
    policy_question = st.text_input("Enter your policy question:",
                                   value=st.session_state.current_state.get('input', ''))

run_button = st.button("Run Selected Tool")

if run_button:
    st.session_state.current_state['final_output'] = ""
    run_tool = True

    if tool_choice == 'TalentScout (Screen Resumes)':
        st.session_state.current_state['next_action'] = "TalentScout"
        st.session_state.current_state['resume_directory'] = resume_directory
        st.session_state.current_state['job_requirements'] = job_requirements

        if not resume_directory or not os.path.isdir(resume_directory):
            st.error("Please enter a valid resume folder path.")
            run_tool = False
        elif not job_requirements:
            st.error("Please enter the job requirements.")
            run_tool = False

    elif tool_choice == 'Onboarder (Generate Onboarding Plan)':
        st.session_state.current_state['next_action'] = "Onboarder"
        if not st.session_state.current_state.get("best_candidate"):
            st.warning("Please run TalentScout first to select a candidate.")
            run_tool = False

    elif tool_choice == 'PolicyQA (Answer Policy Questions)':
        st.session_state.current_state['next_action'] = "PolicyQA"
        st.session_state.current_state['input'] = policy_question
        if not policy_question:
            st.error("Please enter a policy question.")
            run_tool = False

    if run_tool:
        with st.spinner(f"Running {st.session_state.current_state['next_action']}..."):
            # Record execution start time
            start_time = time.time()
            action = st.session_state.current_state['next_action']
            
            # Execute workflow
            updated_state = st.session_state.workflow.invoke(st.session_state.current_state)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Determine success
            success = bool(updated_state.get("final_output") and "Error" not in updated_state.get("final_output", ""))
            
            # Add to simple memory
            SimpleMemory.add_action(
                st.session_state.current_state,
                action,
                success,
                execution_time
            )
            
            # Update the state
            st.session_state.current_state.update(updated_state)

# --- Display Output ---

st.header("Output")

if st.session_state.current_state.get('final_output'):
    st.markdown(st.session_state.current_state['final_output'])

# if st.session_state.current_state.get('best_candidate'):
#     st.subheader("Best Candidate Selected:")
#     st.json(st.session_state.current_state['best_candidate'])

# if st.session_state.current_state.get('processed_resumes'):
#     st.subheader("Processed Resumes:")
#     st.json(st.session_state.current_state['processed_resumes'])

if st.session_state.current_state.get('best_candidate'):
    st.subheader("Best Candidate Selected:")
    st.json(st.session_state.current_state['best_candidate'])

if st.session_state.current_state.get('processed_resumes'):
    st.subheader("Processed Resumes:")
    for idx, resume in enumerate(st.session_state.current_state['processed_resumes'], 1):
        with st.expander(f"📄 {resume['candidate_data']['name']} - Score: {resume['relevance_score']:.2f}"):
            st.write(f"**File:** {resume['filename']}")
            st.write(f"**Skills:** {', '.join(resume['candidate_data']['skills'])}")
            st.write(f"**Experience:** {resume['candidate_data']['work_experience_years']} years")
            st.write(f"**Education:** {resume['candidate_data']['education']}")

if st.session_state.current_state.get('onboarding_plan'):
    st.subheader("Generated Onboarding Plan:")
    st.markdown(st.session_state.current_state['onboarding_plan'])

# --- Simple Memory System ---
if st.session_state.current_state.get('workflow_history'):
    with st.expander("🧠 Recent Activity"):
        recent_actions = st.session_state.current_state['session_memory'].get('actions_performed', [])[-3:]
        if recent_actions:
            st.write("**Last Actions:**")
            for action in recent_actions:
                success_icon = "✅" if action.get('success') else "❌"
                duration = action.get('execution_time', 0)
                st.write(f"{success_icon} {action.get('action', 'Unknown')} ({duration:.1f}s)")
        else:
            st.info("No recent activity")

# Debug section (optional, can be commented out)
if st.checkbox("Show Debug Information"):
    st.subheader("Current State (for debugging):")
    debug_state = dict(st.session_state.current_state)
    # Truncate large data for readability
    if 'workflow_history' in debug_state and len(debug_state['workflow_history']) > 3:
        debug_state['workflow_history'] = debug_state['workflow_history'][-3:] + [{"...": f"and {len(debug_state['workflow_history'])-3} more entries"}]
    st.json(debug_state)