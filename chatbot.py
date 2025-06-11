import nltk
import numpy as np
import random
import string  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('wordnet')


corpus = """


sent_tokens = nltk.sent_tokenize(corpus.lower())
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["Hi there!", "Hello!", "Hey!", "Hi, how can I help?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def generate_response(user_input):
    user_input = user_input.lower()
    sent_tokens.append(user_input)

    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = similarity.argsort()[0][-1]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-1]

    sent_tokens.pop()

    if score > 0.2:
        return sent_tokens[idx]
    else:
        return "I'm sorry, I didn't understand that. Can you rephrase?
def chatbot():
    print("BOT: Hello! I’m your NLP chatbot. Type 'bye' to exit.")
    while True:
        user_input = input("YOU: ")
        if user_input.lower() == 'bye':
            print("BOT: Goodbye! Have a great day!")
            break
        elif greeting(user_input) is not None:
            print("BOT:", greeting(user_input))
        else:
            print("BOT:", generate_response(user_input))

if __name__ == "__main__":
    chatbot()
