import gymnasium as gym
from agent import Agent


env = gym.make('HalfCheetah-v4', render_mode='rgb_array')
env = gym.wrappers.RecordVideo(
    env=env,
    video_folder='./videos',
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
    ep_return = 0
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
            
        state = next_state

        ep_return += reward

    print(f'{ep=} {ep_return=}')


env.close()

