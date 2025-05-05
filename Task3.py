import nltk
import string
import numpy as np
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
!pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined responses for simplicity
responses = {
    "greeting": ["Hello! How can I help you?", "Hi there! What can I do for you?"],
    "farewell": ["Goodbye! Have a great day!", "Bye! Take care!"],
    "default": ["I'm sorry, I didn't understand that. Can you rephrase?"]
}

# Function to check for greetings
def check_greeting(sentence):
    greetings = ["hello", "hi", "hey", "greetings"]
    tokens = word_tokenize(sentence.lower())
    for word in tokens:
        if word in greetings:
            return True
    return False

#raw data
raw_data = {
"what is machine learning?":"Machine learning is a field of artificial intelligence (AI) that enables computers to learn from data without being explicitly programmed.",
"what are types of machine learning": "There are four main types of machine learning: supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning.",
"what is NLP in machine learning?": "In the field of machine learning, Natural Language Processing (NLP) is a subfield focused on enabling computers to understand, interpret, and generate human language.",
"what is artificial intelligence": "Artificial intelligence (AI) is a technology that enables machines to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and decision-making. It encompasses a wide range of techniques, including machine learning, deep learning, and natural language processing, allowing computers to learn from data and make predictions or decisions without explicit programming.",
"what is deep learning": "Deep learning is a subfield of machine learning that utilizes artificial neural networks with multiple layers (deep neural networks) to analyze and learn from data. These networks, inspired by the human brain, are trained on large datasets to recognize complex patterns, make predictions, and perform tasks like image recognition, natural language processing, and speech recognition."
}
# Preprocess raw_data
questions = list(raw_data.keys())
answers = list(raw_data.values())
# Download NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Main chatbot function
def chatbot_response(user_input):
    if check_greeting(user_input):
        return random.choice(responses["greeting"])

# Use spaCy to parse the input
    doc = nlp(user_input)
    for token in doc:
        if token.text.lower() in ["bye", "goodbye", "see you"]:
            return random.choice(responses["goodbye, takecare!"])
# Preprocess user input
def preprocess_input(user_input):
    doc = nlp(user_input)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Chatbot logic
def chatbot_response(user_input):
    preprocessed_input = preprocess_input(user_input)

# Vectorize questions and user input
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [preprocessed_input])

# Compute cosine similarity
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_index = similarity_scores.argmax()

# Respond with the best match
    if similarity_scores[0, best_match_index] > 0.2:
        return answers[best_match_index]
    else:
        return "I'm sorry, I don't understand that question. Could you rephrase?"


# Function to find synonyms using WordNet
def find_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)
# Tokenize and preprocess the input
    tokens = preprocess(user_input)

# Example: Check for keywords and provide a simple response
    if "weather" in tokens:
        return "I can't give you weather updates, but you can check online!"

    return random.choice(responses["default"])
# Find answer in the raw data
def get_answer(query):
    for key, value in raw_data.items():
        if key in query.lower():
            return value
    return none

 # Check the raw data
    answer = get_answer(user_input)
    if answer:
        return answer

    return random.choice(responses["default"])
# Chat loop
if __name__ == "__main__":
    print("Chatbot: Hello! I'm your knowledge guru. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Chatbot:", random.choice(responses["farewell"]))
            break
        print("Chatbot:", chatbot_response(user_input))
