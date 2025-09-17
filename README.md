# GmailSentimentAnalysis

## Description

Con questo progetto si intende sviluppare un modello di predizione del sentimento delle e-mail recapitate presso caselle di posta elettronica con dominio Gmail o domini ad esso associati. L’obiettivo principale è analizzare in maniera automatica il contenuto dei messaggi e classificarne il sentimento in base al grado di criticità che la comunicazione può rappresentare per l’ecosistema aziendale.

## Authors
- [Andrea Ruggiero](https://github.com/andrearuggiero150) - *Team Member*

## Documentation 
- [Documentazione](https://github.com/andrearuggiero150/GmailSentimentAnalysis/blob/main/documentation.pdf)
- [Demo](https://youtu.be/1rQhNQlwQ98)

## Training model guide 
Creare un ambiente virtuale
```
python -m venv venv
```
Attivare l’ambiente virtuale

```
venv\Scripts\activate
```
Installare le librerie necessarie
```
pip install -r requirements.txt
```
Avviare Jupyter Notebook
```
jupyter notebook
```
Aprire il file [regressionModel.ipynb](https://github.com/andrearuggiero150/GmailSentimentAnalysis/blob/main/regressionModel.ipynb) dal browser e lanciare le celle.

## Usage app guide 
Creare il file startup.env nella root del progetto con le seguenti variabili:
```
GOOGLE_CLIENT_ID= ---
GOOGLE_CLIENT_SECRET= ---
FLASK_SECRET_KEY= ---
FERNET_KEY= ---
```
Creare un ambiente virtuale
```
python -m venv venv
```

Attivare l’ambiente virtuale

```
venv\Scripts\activate
```


Installare le librerie necessarie
```
pip install -r requirements.txt
```

Avviare l’app Flask
```
python app.py
```

Aprire il browser all’indirizzo:
```
https://127.0.0.1:5000
```

## Contributing
Se desideri contribuire a questo progetto, segui queste linee guida:

1. Fork del repository
2. Crea un branch per le tue modifiche `git checkout -b feature/aggiungi-nuova-funzionalita`
3. Committa le tue modifiche `git commit -am 'Aggiungi nuova funzionalità'`
4. Pusha il branch `git push origin feature/aggiungi-nuova-funzionalita`
5. Apri una Pull Request

Assicurati di descrivere chiaramente le tue modifiche e di includere informazioni rilevanti o screenshot, se necessario.
## Built with
<a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,flask,gcp,html,javascript,bootstrap,gitlab" />
  </a>

`Prodotto realizzato in ambito di tirocinio curricolare esterno.`
