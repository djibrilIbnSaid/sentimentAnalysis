from langchain_core.messages import HumanMessage

class HumanAgent:
    def __init__(self):
        self.name = 'HumanAgent'
    
    def invoke(self, state):
        """Ajoute du contexte fourni par un utilisateur humain dans le workflow."""
        # Simuler une interaction où l'humain fournit le contexte
        # Dans un cadre réel, cela pourrait être une interaction utilisateur
        # ou un champ de texte dans une interface.
        text = input("Entrez le contexte: ")
        return {
            "messages": [
                HumanMessage(content=f"Context provided by human:")
            ],
            "data": "",
            "context": text
        }