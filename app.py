import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# --------------------------------------------------
# 1. Load the AI Model (using an open-source model)
# --------------------------------------------------
@st.cache_resource
def load_model():
    # Using a Mistral model from Hugging Face as an example
    return pipeline("text-generation", model="facebook/opt-1.3b", device=-1)

model = load_model()

# --------------------------------------------------
# 2. Global Settings and Session State Initialization
# --------------------------------------------------
TOTAL_QUESTIONS = 10
difficulty_levels = ["easy", "medium", "hard"]
scoring = {"easy": 1, "medium": 2, "hard": 3}

if "started" not in st.session_state:
    st.session_state.started = False
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "current_difficulty" not in st.session_state:
    st.session_state.current_difficulty = "medium"
if "questions" not in st.session_state:
    st.session_state.questions = {}  # will hold each question's details

# --------------------------------------------------
# 3. Helper Functions
# --------------------------------------------------

def adjust_difficulty(current_difficulty, correct):
    index = difficulty_levels.index(current_difficulty)
    if correct:
        return difficulty_levels[min(index + 1, len(difficulty_levels) - 1)]
    else:
        return difficulty_levels[max(index - 1, 0)]

def parse_response(response):
    """
    Parses the raw AI output.
    Expected format:
      Question: <text>
      A) <option text>
      B) <option text>
      C) <option text>
      D) <option text>
      E) <option text>
      Correct Answer: <letter>
    If parsing fails, returns a fallback dummy question.
    """
    text = response[0]["generated_text"]
    lines = text.split("\n")
    question = ""
    options = {}
    correct = ""
    for line in lines:
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
        elif line.startswith("A)") or line.startswith("B)") or \
             line.startswith("C)") or line.startswith("D)") or line.startswith("E)"):
            letter = line[0]
            option_text = line[2:].strip()
            options[letter] = option_text
        elif line.startswith("Correct Answer:"):
            correct = line.replace("Correct Answer:", "").strip()
    if not question or len(options) < 5 or not correct:
        # Fallback if model output is unexpected
        question = "What is 2 + 2?"
        options = {"A": "3", "B": "4", "C": "5", "D": "6", "E": "7"}
        correct = "B"
    return {"question": question, "options": options, "correct_answer": correct}

def generate_question(difficulty="medium"):
    prompt = (
        f"Generate a GMAT-style quantitative problem with a {difficulty} difficulty level. "
        "Provide 5 answer choices labeled A) through E) and indicate the correct answer using 'Correct Answer: [letter]'."
    )
    response = model(prompt, max_length=200)
    return parse_response(response)

def plot_score(score):
    fig, ax = plt.subplots()
    scores = [score, 10, 30]
    labels = ["Your Score", "Min Score", "Max Score"]
    ax.bar(labels, scores, color=["blue", "gray", "gray"])
    ax.set_ylabel("Score")
    st.pyplot(fig)

# --------------------------------------------------
# 4. Build the Streamlit UI
# --------------------------------------------------
st.title("AI-Driven GMAT Adaptive Test")

# Start Test Button
if not st.session_state.started:
    if st.button("Start Test"):
        st.session_state.started = True
        st.experimental_rerun()

# Test in Progress
if st.session_state.started:
    if st.session_state.current_question < TOTAL_QUESTIONS:
        st.write(f"**Question {st.session_state.current_question + 1} of {TOTAL_QUESTIONS}**")
        
        # Generate new question if one isn’t already loaded
        if "current_question_data" not in st.session_state:
            st.session_state.current_question_data = generate_question(st.session_state.current_difficulty)
        qdata = st.session_state.current_question_data
        
        st.markdown(f"**{qdata['question']}**")
        # Ensure options are presented in alphabetical order
        option_keys = sorted(qdata["options"].keys())
        user_answer = st.radio("Select your answer:", option_keys, key="user_answer_radio")
        
        if st.button("Submit Answer"):
            correct = qdata["correct_answer"]
            is_correct = (user_answer == correct)
            points = scoring[st.session_state.current_difficulty] if is_correct else 0
            st.session_state.score += points
            
            # Record this question's details
            entry = {
                "Question": qdata["question"],
                "Your Answer": user_answer,
                "Correct Answer": correct,
                "Difficulty": st.session_state.current_difficulty,
                "Points Awarded": points,
                "Result": "Correct" if is_correct else "Incorrect"
            }
            st.session_state.questions[st.session_state.current_question + 1] = entry
            
            # Adjust difficulty based on answer
            st.session_state.current_difficulty = adjust_difficulty(st.session_state.current_difficulty, is_correct)
            st.session_state.current_question += 1
            
            # Remove the current question to load a new one
            del st.session_state.current_question_data
            st.experimental_rerun()
    else:
        # Test Completed: Show Summary and Score Visualization
        st.success("Test Completed!")
        st.write(f"**Final Score: {st.session_state.score}**")
        
        df = pd.DataFrame.from_dict(st.session_state.questions, orient="index")
        st.subheader("Summary of Your Responses")
        st.dataframe(df)
        
        st.subheader("Score Comparison")
        plot_score(st.session_state.score)
        
        if st.button("Restart Test"):
            # Reset all session state values to restart the test
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
