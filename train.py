import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] # pares de dados

# processa os padrões pra cada intent
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

# entrada e saída
X_train = []
Y_train = []

# converte cada padrão de palavras pra um vetor bag_of_words e associa ao i da tag
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) # entrada

    label = tags.index(tag)
    Y_train.append(label) # saída

# converte lista pra array numpy
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# dataset personalizada do pytorch
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# hiperparâmetros do modelo
batch_size = 8                # lote para o treinamento
hidden_size = 8               # neurônios na camada oculta
output_size = len(tags)       # tags 
input_size = len(X_train[0])  # vetor de entrada
learning_rate = 0.001         # taxa de aprendizado
num_epochs = 1000             # epochs de treinamento

# divide os dados em batches
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cria o modelo e move para o dispositivo
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# define a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# loop de treinamento
for epoch in range(num_epochs):
    for(words, labels) in train_loader:

        words = words.to(device)
        labels = labels.to(device).long()

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
        
print(f'final loss, loss={loss.item():.4f}')

# dict: salva modelo criado
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'treinamento completo. salvo em {FILE}')
