from flask import Flask, render_template, request, jsonify
import random
import json
import torch
import speech_recognition as sr
import pyttsx3
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Nova"

engine = pyttsx3.init()

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

@app.route('/')
def index():
    return render_template('cindex.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    sentence = data['message']
    if sentence:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            response = ""
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
            return jsonify({"response": response})
        else:
            return jsonify({"response": "I do not understand..."})

if __name__ == "__main__":
    app.run(debug=True)
    
