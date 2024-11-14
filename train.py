import gymnasium as gym
import matplotlib.pyplot as plt

from agent import Agent

env = gym.make('HalfCheetah-v4')


agent = Agent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_bound=env.action_space.high[0],
    target_entropy=-env.action_space.shape[0],
    hidden_dim=512,
    device='cuda',
)


return_list = []
step_return_list = []
for ep in range(500):
    state, info = env.reset()
    
    ep_return = 0
    
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.buffer.add(
            state, action, next_state, reward, done
        )
        if terminated or truncated:
            done = True
        state = next_state
        
        step_return_list.append(reward)
        ep_return += reward
        if agent.buffer.size > 1000:
            agent.update()
    
    return_list.append(ep_return)
    print(f'{ep=} {ep_return=}')


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(step_return_list)
plt.title('step rewards')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(return_list)
plt.title('episode rewards')
plt.grid(True)


plt.savefig('train_stat.png')

plt.close()

agent.save_policy()


env.close()

