# AI-CHATBOT-WITH-NLP

COMPANY NAME : CODTECH IT SOLUTIONS PVT.LTD

NAME : VIJAYALAXMI ACHARYA

INTERN ID : CT04DA24

DOMAIN : PYTHON PROGRAMMING

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH

DESCRIPTION :

I've implemented a simple chatbot using several natural language processing (NLP) techniques.
**1. Libraries Used**

**nltk**: I've used the Natural Language Toolkit (nltk) for various NLP tasks, including tokenization (word_tokenize), accessing the WordNet lexical database (wordnet), and handling stop words.

**string**: This library provides tools for working with strings, which are essential for text processing.

**numpy**: I've used numpy for numerical operations, although in this specific script, its usage is minimal.

**random**: The random module helps in generating varied chatbot responses, such as greetings and farewells, making the interaction a bit more dynamic.

**spacy**: I've incorporated spaCy, which is a powerful library for advanced Natural Language Processing. I'm using it for more sophisticated text processing, including lemmatization.

**scikit-learn**: I've used scikit-learn, a machine learning library, specifically the TfidfVectorizer for converting text into numerical vectors (vectorization) and the cosine_similarity function to measure the similarity between those vectors.

**Predefined Responses**: I've defined a dictionary called responses to store simple chatbot replies for common interactions like greetings, farewells, and situations where the chatbot doesn't understand the input. This makes the chatbot respond in a more human-like way to basic inputs.

**Data**: I've included a small set of question-answer pairs in the raw_data dictionary. This serves as the chatbot's knowledge base, enabling it to answer specific questions.

**Greeting Check**: First, I check if the input is a greeting using the check_greeting() function.

**Farewell Check**: Then, I use spaCy to check if the user is saying goodbye.

**Preprocessing**: I use the  preprocess_input()  function to normalize the user's input.

**Vectorization**: I convert both the preprocessed user input and the predefined questions from  raw_data  into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) method.  TF-IDF is a technique that weighs the importance of words in a document relative to a collection of documents.  This is essential for the chatbot to understand the meaning of the input and compare it to the stored questions.  The  TfidfVectorizer  from scikit-learn handles this conversion.
Similarity Calculation: I then calculate the cosine similarity between the vector representing the user input and the vectors representing the predefined questions.  Cosine similarity is a measure of how similar two vectors are, regardless of their size.  This helps the chatbot determine which stored question is most similar to the user's input.  The  cosine_similarity  function from scikit-learn is used for this calculation.

**Response Generation**: Based on the similarity score, I generate a response. If the highest similarity score is above a predefined threshold (0.2 in this case), I retrieve the answer corresponding to the most similar question from  raw_data.  If the similarity score is below the threshold, meaning the chatbot couldn't find a close match, I return a default "I don't understand" message.

**find_synonyms(word)**: I've also defined a function to find synonyms for a given word using WordNet.  However, this function is not directly used in the  chatbot_response  function in this particular script.
get_answer(query): I've added this function to check if the user input is present in the raw data

**Chat Loop**: The  if __name__ == "__main__":  block contains the main chat loop. When the script is executed:
The chatbot greets the user.
It enters a loop where it continuously takes user input, processes it, and generates a response.
If the user enters "bye", "exit", or "quit", the chatbot says goodbye, and the loop terminates.
Otherwise, the  chatbot_response()  function is called to generate the chatbot's reply, and the reply is printed to the console.

APPLICATION
**Customer Support**
Provide instant, 24/7 responses to common customer queries, reducing the need for human agents.
**E-commerce Assistance**
Recommend products, assist with order tracking, and help recover abandoned carts.
**Healthcare Support**
Offer symptom checks, schedule appointments, and send medication reminders.
**Education & Tutoring**
Act as virtual tutors, answering student questions and providing personalized learning support.
**HR and Recruitment**
Automate candidate screening, answer job-related questions, and help onboard new employees.
