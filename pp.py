# import gymnasium as gym

# from generals.agents import RandomAgent, ExpanderAgent

# # Initialize agents
# agent = RandomAgent()
# npc = ExpanderAgent()

# # Create environment
# env = gym.make("gym-generals-v0", agent=agent, npc=npc, render_mode="human")

# observation, info = env.reset()
# terminated = truncated = False
# while not (terminated or truncated):
#     action = agent.act(observation)
#     observation, reward, terminated, truncated, info = env.step(action)
#     env.render()

from generals.remote import autopilot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--user_id", type=str, default="[Bot]js6407") # Register yourself at generals.io and use this id
parser.add_argument("--lobby_id", type=str, default="3770") # The last part of the lobby url
parser.add_argument("--agent_id", type=str, default="Expander") # agent_id should be "registered" in AgentFactory

if __name__ == "__main__":
    args = parser.parse_args()
    autopilot(args.agent_id, args.user_id, args.lobby_id)