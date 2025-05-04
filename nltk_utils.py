import nltk # natural language toolkit
import numpy as np
from nltk.stem.porter import PorterStemmer

def tokenize(sentence): # dividindo a string em unidades funcionais
    return nltk.word_tokenize(sentence)

stemmer = PorterStemmer()
def stem(word): # gera a forma raiz da palavra e corta o sufixo
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words): 
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# testando

# a = "como vc tá?"
# print(a)
# a = tokenize(a)
# print(a)

# words = ["Organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)