import torch
import numpy as np



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # random trasitions
        ind = np.random.randint(0, self.size, size=batch_size)
        # TODO: random segments
                

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )




if __name__ == '__main__':
    buffer = ReplayBuffer(10, 3, max_size=10)
    buffer.add(
        np.random.rand(10),
        np.random.rand(3),
        np.random.rand(10),
        0.5,
        False,
    )
    
    buffer.add(
        np.random.rand(10),
        np.random.rand(3),
        np.random.rand(10),
        0.5,
        True,
    )

    states, actions, next_states, rewards, dones = buffer.sample(3)
    
    print(states.shape)
    
    print(rewards.shape)
    
