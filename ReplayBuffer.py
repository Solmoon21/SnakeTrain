import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_dims, input_dims_conv=[]):
        self.mem_size = max_size
        self.mem_cnt = 0
        self.conv_state_memory = np.zeros( (self.mem_size, *input_dims_conv), dtype=np.float32 )
        self.state_memory = np.zeros( (self.mem_size, input_dims), dtype=np.float32 )
        self.new_conv_state_memory = np.zeros( (self.mem_size, *input_dims_conv), dtype=np.float32 )
        self.new_state_memory =  np.zeros( (self.mem_size, input_dims), dtype=np.float32 )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, conv_state, state, action, reward, new_conv_state, new_state, done):
        #print(type(reward),'3')
        #print(reward)
        index = self.mem_cnt % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.conv_state_memory[index] = conv_state
        self.new_conv_state_memory[index] = new_conv_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.mem_cnt += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        conv_states = self.conv_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        new_conv_states = self.new_conv_state_memory[batch]
        terminal = self.done_memory[batch]

        return conv_states, states, actions, rewards, new_conv_states, new_states, terminal