import gymnasium as gym
from agent import Agent


env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
env = gym.wrappers.RecordVideo(
    env=env,
    video_folder='./videos',
    episode_trigger=lambda t: t % 1 == 0,
)

agent = Agent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_bound=env.action_space.high[0],
    target_entropy=-env.action_space.shape[0],
    hidden_dim=512,
    device='cuda',
)

agent.load_policy()

for ep in range(10):
    state, info = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
            
        state = next_state


env.close()

