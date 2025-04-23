import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chatbot_data import qa_pairs

# Initialize chatbot logic
class SimpleChatbot:
    def __init__(self, qa_pairs):
        self.qa_pairs = qa_pairs
        self.questions = list(qa_pairs.keys())
        self.answers = list(qa_pairs.values())
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_input):
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.question_vectors)
        best_match_index = np.argmax(similarities)

        if similarities[0, best_match_index] > 0.3:
            return self.answers[best_match_index]
        else:
            return "I'm sorry, I don't understand that question."

# Set up Streamlit UI
st.set_page_config(page_title="Python Chatbot", page_icon="🐍")
st.title("🤖 Python Programming Chatbot")
st.write("Ask me anything about **Python programming**!")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
user_input = st.text_input("You:", key="user_input")

# Respond to user input
if user_input:
    bot = SimpleChatbot(qa_pairs)
    response = bot.get_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# Display chat history
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")
