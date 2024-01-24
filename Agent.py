from ReplayBuffer import ReplayBuffer
from DeepQNetwork import DeepQNetwork

import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.losses import MSE

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, input_dims_conv=None, eps_min=0.01, 
                 eps_dec=5e-7, replace=1000, model_version='v1'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.input_dims_conv = input_dims_conv
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.model_version = model_version

        if self.model_version == 'v1':
            #print("a v1")
            self.memory = ReplayBuffer(mem_size, input_dims)
            self.q_eval = DeepQNetwork(input_dims=input_dims, n_actions=n_actions)
            self.q_eval.compile(optimizer=Adam(learning_rate=lr))
            self.q_next = DeepQNetwork(input_dims, n_actions)
            self.q_next.compile(optimizer=Adam(learning_rate=lr))
        else:
            #print("a v2")
            self.memory = ReplayBuffer(mem_size, input_dims, input_dims_conv)
            self.q_eval = DeepQNetwork(input_dims, n_actions, input_dims_conv, isV1=False)
            self.q_eval.compile(optimizer=Adam(learning_rate=lr))
            self.q_next = DeepQNetwork(input_dims, n_actions, input_dims_conv, isV1=False)
            self.q_eval.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        self.q_eval.save_weights('model'+self.model_version+'.h5')
        print('... models saved successfully ...')

    def load_models(self):
        self.q_eval.load_weights('model'+self.model_version+'.h5')
        self.q_next.load_weights('model'+self.model_version+'.h5')
        print('... models loaded successfully ...')

    def store_transition(self, state, action, reward, state_, done, conv_state=None, conv_state_=None):
        #print(type(reward),'2')
        #print(reward)
        self.memory.store_transition(conv_state, state, action, reward, conv_state_, state_, done)

    def sample_memory(self):
        conv_state, state, action, reward, new_conv_state, new_state, done = self.memory.sample_buffer(self.batch_size)
        conv_states = tf.convert_to_tensor(conv_state)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        conv_states_ = tf.convert_to_tensor(new_conv_state)
        states_ = tf.convert_to_tensor(new_state)
        return conv_states, states, actions, rewards, conv_states_, states_, dones
    
    def choose_action(self, state, action_space):
        if np.random.random() > self.epsilon:
            if self.model_version=='v1':
              state = tf.convert_to_tensor([state])
            else:
              state = (tf.convert_to_tensor([state[0]]), tf.convert_to_tensor([state[1]])) 
            
            #print(state)
            actions = self.q_eval(state)
            sorted_actions = tf.argsort(actions, axis=1).numpy()[0]
            action = sorted_actions[-1]
            if action in action_space:
                return action
            else:
                action = sorted_actions[-2]
                return action
        else:
            action = np.random.choice(action_space)
        return action
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
        
    def learn(self):
        if self.memory.mem_cnt < self.batch_size:
            return
        
        self.replace_target_network()

        conv_states, states, actions, rewards, new_conv_states, new_states, dones = self.sample_memory()

        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        if self.model_version == 'v1':
            eval_input = states
            next_input = new_states
        else:
            eval_input = (conv_states, states)
            next_input = (new_conv_states, new_states)
        
        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.q_eval(eval_input), indices=action_indices)
            q_next = self.q_next(next_input)

            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = rewards + \
                        self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                        (1- dones.numpy())
            
            loss = MSE(q_pred, q_target)
            
        params = self.q_eval.trainable_variables 
        grads = tape.gradient(loss, params)
        self.q_eval.optimizer.apply_gradients(zip(grads, params))
        self.learn_step_counter += 1
        self.decrement_epsilon()