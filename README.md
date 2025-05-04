# Experiência Conversacional com ChatBot 🧠

- Este projeto é uma aplicação web desenvolvida com Flask, que utiliza NLTK e PyTorch para treinar e executar um modelo simples de chatbot. O treinamento é feito a partir de um arquivo intents.json, que define as intenções e padrões de entrada do usuário. Com base nesse treinamento, o chatbot é capaz de identificar frases e gerar respostas apropriadas em tempo real.

### ⚙️ Principais Tecnologias
<table>
  <thead>
    <tr>
      <th>Biblioteca</th>
      <th>Função Principal</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Flask</td>
      <td>Criação da API Web</td>
      <td><a href="https://flask.palletsprojects.com/">Flask</a></td>
    </tr>
    <tr>
      <td>NLTK</td>
      <td>Processamento de linguagem natural</td>
      <td><a href="https://www.nltk.org/">NLTK</a></td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td>Treinamento do modelo de IA</td>
      <td><a href="https://pytorch.org/">PyTorch</a></td>
    </tr>
  </tbody>
</table>

## 📦 Requisitos para rodar o projeto

### Setup do ambiente:

```
git clone https://github.com/ghabryeu/chat_bot.git
```
```
python -m venv env
source env/Scripts/activate  # no Windows
```

### Instalar as dependências e [PyTorch](https://pytorch.org/)
```
pip install -r requirements.txt
```
Se você receber um erro durante a primeira execução, tente instalar nltk.tokenize.punkt . Rode no seu terminal:
```
$ python
>>> import nltk
>>> nltk.download('punkt')
```

### Com tudo instalado, rode:

```
python train.py
```

### Uma vez que train.py executar com êxito, rode chat.py para acessar o chat pelo terminal:
```
python chat.py
```

### Ou rode app.py para acessar chat pela aplicação web
```
python app.py
```

### Intents
Em intents.json, você pode modificá-lo de acordo com o contexto do chat. Você pode alterar as possíveis interações dos usuários e retornar respostas. Para usar, você executa train.py novamente sempre que o arquivo for modificado.
```
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Oi", "Olá", "E aí", "Salve", "Fala FURIA", "Tem alguém aí?", "Mano",
        "E aí, beleza?", "Fala aí, tudo certo?", "Alguém online?", "Boa tarde", "Bom dia", "Boa noite",
        "Fala, Gabriel", "Cheguei!", "Opa", "Hey", "Yo", "Tudo bem?"
      ],
      "responses": [
        "Fala, fã da FURIA! Em que posso ajudar?",
        "Oi! Tá preparado pra saber tudo sobre o time de CS?",
        "Salve! Manda a braba, o que você quer saber?"
      ]
    },
```

