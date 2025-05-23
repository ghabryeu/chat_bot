import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# modelo treinado
FILE = "data.pth"
data = torch.load(FILE)

# extrai dados do dict salvo
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define dispositivo
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# chat
bot_name = "Gabriel"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
            
    return "Desculpa, mas não entendi..."


if __name__ == "__main__":
    print("Vamos conversar! (Digite 'quit' pra sair.)")
    while True:
        sentence = input('Você: ')
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

