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
- Alexendre ARNAUD
- Ayoub HIDARA

## Remerciements
- Merci à toutes les bibliothèques open-source utilisées dans ce projet.
