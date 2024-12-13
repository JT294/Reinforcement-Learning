import gymnasium as gym
from generals.core.grid import GridFactory
from generals.agents import ExpanderAgent  
from generals.agents.MCTS_agent import MCTSAgent
from generals.envs.gymnasium_generals import GymnasiumGenerals
agent = MCTSAgent()
npc = ExpanderAgent()
grid_factory = GridFactory(    
    grid_dims=(5, 5),  # Grid height and width
    mountain_density=0.2,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    general_positions=[(1, 2), (4, 4)],  # Positions of the generals
    seed=38,  # Seed to generate the same map every time
    )
def reward_function(observation,action,done,info):
    if done:
        if info["is_winner"]:
            return 1
        else:
            return -1
    else:
        return 0
        
def unwrap_game(env):
    while hasattr(env, "env"):  # 递归解包包装器
        env = env.env
    return getattr(env, "game", None)  # 获取底层环境的 game 属性

env = gym.make("gym-generals-v0", grid_factory=grid_factory, agent=agent, npc=npc, render_mode="human", reward_fn=reward_function)
options = {
    "replay_file": "my_replay_1",  # Save replay as my_replay.pkl
}
observation, info = env.reset(options=options)
terminated = truncated = False
prev_action = None
while not (terminated or truncated):
    game = unwrap_game(env)
    if game is None:
        raise RuntimeError("The environment does not contain a 'game' attribute.")
    action = agent.act(observation, game, terminated, prev_action) #, False)
    observation, reward, terminated, truncated, info = env.step(action)
    prev_action = action 
    env.render()
