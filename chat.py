import time
import threading
import logging
import random
import json
import torch
import pyttsx3
import speech_recognition as sr
from flask import Flask, request, jsonify
from waitress import serve
from rocketchat_API.rocketchat import RocketChat
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
recognizer = sr.Recognizer()


bot = RocketChat(user_id='Z2HJCanxJxvfThFDm', auth_token='wr7BDFDqJcn9e2cB5qCQxsgnHLreFPE0rNrClTUk1la', server_url='https://chat.globalhealthhelpline.com')
channel_name = 'general'  
channel_info = bot.channels_info(channel=channel_name).json()
room_id = channel_info['channel']['_id']
logger.info(f"Retrieved room_id: {room_id}")

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def generate_response(sentence):
    """Generate a response from the chatbot."""
    sentence = sentence.lower()
    
    if any(word in sentence for word in ["quit", "stop", "goodbye"]):
        return "Goodbye! Feel free to reach out if you need anything else.", True
    
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
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses']), False
    else:
        return "I do not understand...", False

def send_message_to_rocket_chat(message, room_id):
    """Send a message to Rocket.Chat."""
    try:
        bot.chat_post_message(message, room_id=room_id)
        logger.info("Message sent to Rocket.Chat")
    except Exception as e:
        logger.error(f"Failed to send message to Rocket.Chat: {str(e)}")

def send_message_to_brevo(message):
    """Send a message to Brevo using their API."""
    brevo_api_endpoint = "YOUR_BREVO_API_ENDPOINT"  
    payload = {
        "message": message,
    }
    headers = {
        "Authorization": "Bearer YOUR_BREVO_API_KEY",  
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(brevo_api_endpoint, json=payload, headers=headers)
        
        if response.status_code == 200:
            logger.info("Message sent to Brevo successfully")
        else:
            logger.error(f"Failed to send message to Brevo: {response.status_code} {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send message to Brevo: {str(e)}")
    response = requests.post(brevo_api_endpoint, json=payload, headers=headers)
    
    if response.status_code == 200:
        logger.info("Message sent to Brevo successfully")
    else:
        logger.error(f"Failed to send message to Brevo: {response.status_code} {response.text}")

@app.route('/brevo-webhook', methods=['POST'])
def brevo_webhook():
    """Webhook endpoint for Brevo to send messages to this bot."""
    data = request.json
    user_message = data.get("message", "")
    
    if user_message:
        logger.info(f"Received message from Brevo: {user_message}")
       
        response, should_stop = generate_response(user_message)
        logger.info(f"{bot_name}: {response}")
        
       
        send_message_to_rocket_chat(f"User: {user_message}", room_id)
        send_message_to_rocket_chat(f"{bot_name}: {response}", room_id)
        
        
        send_message_to_brevo(response)
        
        if should_stop:
            return jsonify({"message": "Goodbye!"})
        
        return jsonify({"message": response})
    else:
        return jsonify({"error": "No message provided"}), 400

def main():
   
    threading.Thread(target=lambda: serve(app, host='0.0.0.0', port=5000), daemon=True).start()
    
    logger.info("Bot is up and running...")

if __name__ == "__main__":
    main()
