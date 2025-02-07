# # libraries
# import random
# import numpy as np
# import pickle
# import json
# from flask import Flask, request, jsonify
# import nltk
# from keras.models import load_model
# from nltk.stem import WordNetLemmatizer

# # Initialize WordNetLemmatizer for word lemmatization
# lemmatizer = WordNetLemmatizer()

# # Load pre-trained chatbot model and associated data
# model = load_model("final_chatbot_model.h5")
# intents = json.loads(open("cs_data.json").read())
# words = pickle.load(open("words.pkl", "rb"))
# classes = pickle.load(open("classes.pkl", "rb"))

# # Create a Flask web application
# app = Flask(__name__)

# # Define a route to handle incoming messages and respond with chatbot's response
# @app.route("/chatbot", methods=["POST"])
# def chatbot():
#     # Extract the message from the JSON data
#     data = request.get_json()
#     msg = data.get("msg")

#     if not msg:
#         return jsonify({"error": "No message provided"}), 400

#     # Process the message using the chatbot model
#     try:
#         ints = predict_class(msg, model)
#         res = getResponse(ints, intents)
#     except IndexError:
#         res = "Sorry, I didn't understand that."

#     # Return the response as JSON data
#     return jsonify({"response": res})


# # Tokenize and lemmatize the input sentence
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words


# # Create a bag of words array for the input sentence
# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#                 if show_details:
#                     print("found in bag: %s" % w)
#     return np.array(bag)


# # Predict the intent of the input sentence using the model
# def predict_class(sentence, model):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list


# Generate a response based on the predicted intent
# def getResponse(ints, intents_json):
#     tag = ints[0]["intent"]
#     list_of_intents = intents_json["intents"]
#     for i in list_of_intents:
#         if i["tag"] == tag:
#             result = random.choice(i["responses"])
#             break
#     return result


# # Run the Flask application
# if __name__ == "__main__":
#     app.run()



# Libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for enabling CORS

import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize WordNetLemmatizer for word lemmatization
lemmatizer = WordNetLemmatizer()

# Load pre-trained chatbot model and associated data
model = load_model("final_chatbot_model.h5")
intents = json.loads(open("cs_data.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Create a Flask web application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a route to handle incoming messages and respond with chatbot's response
@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Extract the message from the JSON data
    data = request.get_json()
    msg = data.get("msg")

    if not msg:
        return jsonify({"error": "No messageo provided"}), 400

    # Process the message using the chatbot model
    try:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    except IndexError:
        res = "Sorry, I didn't understand that."

    # Return the response as JSON data
    return jsonify({"response": res})


# Tokenize and lemmatize the input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Create a bag of words array for the input sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


# Predict the intent of the input sentence using the model
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Generate a response based on the predicted intent
def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


# Run the Flask application
if __name__ == "__main__":
    app.run()

