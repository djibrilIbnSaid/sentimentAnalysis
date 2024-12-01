import functools

from agents.data_cleaning_agent import DataCleaningAgent
from agents.human_agent import HumanAgent
from agents.labeling_agent import LabelingAgent
from agents.tweet_collector_agent import TweetCollectorAgent
from agents.generator_tweet_agent import GeneratorTweetAgent
from agents.test_model_agent import TestModelAgent
from agents.classifier_agent import ClassifierAgent

from langgraph.graph import StateGraph, START
from agents.supervisor_agent import SupervisorAgent
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict

# Define the state structure
class AgentState(TypedDict):
    messages: list[HumanMessage]
    next: str
    current_index: int
    data: dict
    context: str

# Define members and initialize agents
members = ["Human", "TweetCollector", "DataCleaner", "Labeler", "Generator", "Classifier", "TestModel"]

tweet_collector = TweetCollectorAgent()
data_cleaner = DataCleaningAgent()
labeler = LabelingAgent()
human_agent = HumanAgent()
generator = GeneratorTweetAgent()
test_model = TestModelAgent()
classifier = ClassifierAgent()
supervisor = SupervisorAgent(members)

# Define agent nodes
human_node = functools.partial(SupervisorAgent.agent_node, agent=human_agent, name="Human")
tweet_collector_node = functools.partial(SupervisorAgent.agent_node, agent=tweet_collector, name="TweetCollector")
data_cleaner_node = functools.partial(SupervisorAgent.agent_node, agent=data_cleaner, name="DataCleaner")
labeler_node = functools.partial(SupervisorAgent.agent_node, agent=labeler, name="Labeler")
generator_node = functools.partial(SupervisorAgent.agent_node, agent=generator, name="Generator")
classifier_node = functools.partial(SupervisorAgent.agent_node, agent=classifier, name="Classifier")
test_model_node = functools.partial(SupervisorAgent.agent_node, agent=test_model, name="TestModel")

# Define the workflow
workflow = StateGraph(AgentState)
workflow.add_node("Human", human_node)
workflow.add_node("TweetCollector", tweet_collector_node)
workflow.add_node("DataCleaner", data_cleaner_node)
workflow.add_node("Labeler", labeler_node)
workflow.add_node("Generator", generator_node)
workflow.add_node("Classifier", classifier_node)
workflow.add_node("TestModel", test_model_node)
workflow.add_node("supervisor", supervisor.supervisor_agent)

# Add edges between agents and supervisor
for member in members:
    workflow.add_edge(member, "supervisor")

# Add conditional edges for decision-making by the supervisor
workflow.add_conditional_edges("supervisor", SupervisorAgent.supervisor_decision, supervisor.conditional_map())

# Define the entry point for the workflow
workflow.add_edge(START, "Human")

# Compile the workflow
graph = workflow.compile()

# Define the initial state
initial_context_message = HumanMessage(
    content="Provide context for the tweets you want to collect. "
            "For example: 'Collect tweets related to Kamala Harris.'"
)

initial_state = {
    "messages": [
        initial_context_message
    ],
    "data": {},
}

# Configure execution
execution_config = {"recursion_limit": 150}

# Execute and stream events
events = graph.stream(
    initial_state,
    execution_config,
)

# Print each event
for event in events:
    print(event)
    print("----")