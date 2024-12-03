from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END

class SupervisorAgent:
    def __init__(self, members):
        self.name = "SupervisorAgent"
        self.members = members
        self.options = ["FINISH"] + members
        self.system_prompt = (
            "You are a supervisor tasked with managing a workflow involving the following agents: {members}. "
            "Given the conversation below, decide which agent should act next to progress the workflow. "
            "After an agent performs its task, you will decide the next agent to proceed or to finish the workflow. "
            "Each agent will perform a specific task and respond with their results. When all tasks are completed, respond with FINISH. "
            "Please respond with a JSON object containing the field 'next', whose value is one of: {options}."
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}. "
                    "Please respond with a JSON object containing the field 'next', whose value is one of the options."
                ),
            ]
        ).partial(options=", ".join(self.options), members=", ".join(members))
    
    @staticmethod
    def agent_node(state, agent, name):
        """
        Ajoute un noeud d'agent à l'état du workflow.

        Args:
            state: l'état actuel du workflow
            agent: l'agent à invoquer
            name (str): le nom de l'agent

        Returns:
            state: l'état mis à jour du workflow
        """
        result = agent.invoke(state)
        print("--------------------")
        print(result)
        print("--------------------")
        return {
            "messages": [HumanMessage(content=result["messages"][-1].content, name=name)],
            "data": result['data'],
            "context": result.get('context', {}),
        }
    
    @staticmethod
    def supervisor_decision(x):
        """
        

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            return x["next"]
        except KeyError:
            print("Erreur: Le superviseur n'a pas fourni de décision 'next'. Terminaison du workflow.")
            return "FINISH"
    
    def supervisor_agent(self, state):
        """
        Le superviseur prend une décision sur le prochain agent à invoquer.

        Args:
            state: l'état actuel du workflow

        Returns:
            state: l'état mis à jour du workflow
        """
        
        choices = self.members[1:] + ["FINISH"]
        current_index = state.get("current_index", 0)
        decision = choices[current_index]
        next_index = (current_index + 1) % len(choices)
        print(f"Le superviseur a choisi : {decision}")
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Superviseur: Prochaine action - {decision}")],
            "next": decision,
            "current_index": next_index,
            "data": state.get("data", {}),
            "context": state.get("context", {}),
        }
    
    def conditional_map(self):
        """
        Verifie les agents disponibles pour le superviseur.

        Returns:
            List: Liste des agents disponibles pour le superviseur.
        """
        conditional_map = {k: k for k in self.members}
        conditional_map["FINISH"] = END 
        return conditional_map