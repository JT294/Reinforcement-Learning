from .agent import Agent
from .expander_agent import ExpanderAgent
from .random_agent import RandomAgent
from .MCTS_agent import MCTSAgent

class AgentFactory:
    """
    Factory class for creating agents.
    """

    def __init__(self):
        pass

    @staticmethod
    def make_agent(agent_type: str, **kwargs) -> Agent:
        """
        Creates an agent of the specified type.
        """
        if agent_type == "Random":
            return RandomAgent(**kwargs)
        elif agent_type == "Expander":
            return ExpanderAgent(**kwargs)
        elif agent_type == "MCTS":
            return MCTSAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
