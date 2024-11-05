import time
import threading
import logging
import random
import json
import torch
import pyttsx3
import speech_recognition as sr
from flask import Flask
from waitress import serve
from rocketchat_API.rocketchat import RocketChat
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


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

def listen_for_activation():
    """Listen for the wake word."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Listening for wake word...")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            logger.info(f"Detected: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            logger.error("Service error.")
            return None
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out.")
            return None

def listen():
    """Capture audio and convert to text."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Listening...")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            logger.info(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            logger.error("Service error.")
            return None

def text_input():
    """Capture text input from the user."""
    return input("You: ")

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

def get_last_messages(room_id, count=1):
    """Get the last few messages from Rocket.Chat."""
    try:
        messages = bot.channels_history(room_id=room_id, count=count).json()
        if 'messages' in messages:
            return messages['messages']
        else:
            return []
    except Exception as e:
        logger.error(f"Failed to retrieve messages from Rocket.Chat: {str(e)}")
        return []

def monitor_rocket_chat(room_id):
    """Monitor Rocket.Chat for new messages and allow user to reply."""
    last_message_time = None

    while True:
        messages = get_last_messages(room_id, count=5)
        if messages:
            for message in reversed(messages):
                message_time = message['ts']
                if last_message_time is None or message_time > last_message_time:
                    last_message_time = message_time
                    username = message['u']['username']
                    text = message['msg']
                    if username != 'rocketchat': 
                        logger.info(f"{username}: {text}")
                        print(f"{username}: {text}")
                        
                        reply = input("You (Reply): ")
                        if reply:
                            send_message_to_rocket_chat(reply, room_id)
        time.sleep(5)

def main():
    
    threading.Thread(target=monitor_rocket_chat, args=(room_id,), daemon=True).start()
    
    while True:
        wake_word = listen_for_activation()
        if wake_word and "nova" in wake_word.lower():
            speak("How can I assist you?")
            sentence = listen()
        else:
            print("You can also type your request.")
            sentence = text_input()

        if sentence:
            
            send_message_to_rocket_chat(f"User: {sentence}", room_id)

            response, should_stop = generate_response(sentence)
            logger.info(f"{bot_name}: {response}")
            speak(response)
            send_message_to_rocket_chat(f"{bot_name}: {response}", room_id)

            if should_stop:
                speak("Goodbye!")
                break

if __name__ == "__main__":
  
    threading.Thread(target=lambda: serve(app, host='0.0.0.0', port=5000), daemon=True).start()
    
    
    main()
