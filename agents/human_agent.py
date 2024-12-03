from langchain_core.messages import HumanMessage

class HumanAgent:
    def __init__(self):
        self.name = 'HumanAgent'
    
    def invoke(self, state):
        """
        Méthode principale pour l'agent

        Args:
            state: l'état actuel de l'agent

        Returns:
            dict: l'état mis à jour de l'agent
        """
        text = input("Entrez le contexte: ")
        return {
            "messages": [
                HumanMessage(content=f"Context provided by human:")
            ],
            "data": "",
            "context": text
        }