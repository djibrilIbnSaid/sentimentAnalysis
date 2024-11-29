# SentimentAnalysis

## Description
SentimentAnalysis est un projet en Python qui analyse les sentiments exprimés dans des textes (tweets). Il utilise des techniques avec des système multi-agents pour déterminer si un tweet est positif, négatif ou neutre.

## Installation
1. Clonez le dépôt :
    ```bash
    git clone https://github.com/djibrilIbnSaid/sentimentAnalysis.git
    ```
2. Accédez au répertoire du projet :
    ```bash
    cd SentimentAnalysis
    ```
3. Créez un environnement virtuel et activez-le :
    ```bash
    python3 -m venv env
    source env/bin/activate  
    
    # Sur Windows, utilisez 
    `env\Scripts\activate`
    ```
4. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
5. Autres dependances :
    - installer ollama: https://ollama.com
    - Pour verifier si ollama est bien installé, executer la commande suivante:
        - navigateur: http://localhost:11434
    - bash: `ollama pull llama3.2` pour telecharger le modèle llama3.2

## Configuration des Identifiants Twitter

Pour utiliser l'agent `TweetCollectorAgent`, vous devez fournir vos identifiants de connexion Twitter dans un fichier Python. Suivez les étapes ci-dessous pour configurer vos identifiants :

### Étape 1 : Créer le fichier `.env`
   ```bash
   # .env

    TWITTER_USERNAME=....          # Votre nom d'utilisateur Twitter
    TWITTER_ACCOUNT_PASSWORD=...  # Le mot de passe associé à votre compte Twitter
    TWITTER_EMAIL=...           # L'adresse email liée à votre compte Twitter
    TWITTER_EMAIL_PASSWORD=...  


## Utilisation
1. Exécutez le script d'analyse :
    ```bash
    python analyze.py
    ```
2. Les résultats seront affichés dans la console.

## Contribuer
Les contributions sont les bienvenues ! Veuillez soumettre une pull request ou ouvrir une issue pour discuter des changements que vous souhaitez apporter.

## Licence
Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Auteurs
- Abdoulaye Djibril DIALLO
- Salma KHALLAD
- Alexandre ARNAUD
- Ayoub HIDARA

## Remerciements
- Merci à toutes les bibliothèques open-source utilisées dans ce projet.
