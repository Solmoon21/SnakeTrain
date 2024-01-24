import numpy as np
from time import sleep
from Agent import Agent
from Snake import SnakeGameAI
from consts import *



game_env = SnakeGameAI()
gamma = 0.9
epsilon = 1
lr = 0.001
n_actions = 4
mem_size = 100000
block_size = 20
input_dims = 6
input_dims_conv = (int(game_env.w / block_size), int(game_env.h / block_size), 3)
batch_size = 32

sleep(2)

#dqn_agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, n_actions=n_actions, input_dims=input_dims,
#                     mem_size=mem_size, batch_size=batch_size, eps_min=0.1, eps_dec=1e-5,
#                     replace=200, model_version='v2')

dqn_agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, n_actions=n_actions,input_dims_conv=input_dims_conv, 
                  input_dims=input_dims,mem_size=mem_size, batch_size=batch_size, eps_min=0.1, 
                  eps_dec=1e-5, replace=200, model_version='v2')

dqn_agent.q_next.build(input_shape=[(None, ) + input_dims_conv, (None, input_dims)])
dqn_agent.q_eval.build(input_shape=[(None, ) + input_dims_conv, (None, input_dims)])

loop_cnt = 0
reward_list = []
max_score = 0

#loop_cnt < 1_000_000 or max_score < 100

while True:
    loop_cnt += 1 
    #state = game_env.get_state()
    state = (game_env.get_conv_state(), game_env.get_state())
    #action = dqn_agent.choose_action(state, game_env.get_random_dir())
    action = dqn_agent.choose_action(state, game_env.get_random_dir())
    epreward, done, score = game_env.play_step(action)

    max_score = max(max_score, score)
    #new_state = game_env.get_state()
    new_state = (game_env.get_conv_state(), game_env.get_state())
    if loop_cnt % 1000 > 800:
      action = game_env.get_example_action()

    #print(type(epreward))
    #print(epreward)
    #dqn_agent.store_transition(state=state,action=action,reward=epreward,state_=new_state,done=done)
    
    dqn_agent.store_transition(conv_state=state[0], state=state[1], action=action,
                               reward=epreward, conv_state_=new_state[0], state_=new_state[1], done=done)
    dqn_agent.learn()

    reward_list.append(epreward)

    if loop_cnt % 10000 == 0:
        print('avg rewards: {}'.format(np.mean(reward_list)))
        print('food eaten: {}'.format(sum([x for x in reward_list if x == 1])))
        print('died: {}'.format(sum([x for x in reward_list if x == -1])))
        print('max score: {}'.format(max_score))
        print('current eps: {}'.format(dqn_agent.epsilon))
        reward_list = []
        dqn_agent.save_models()

    if max_score >= 100:
      print("Goal reached")
      sleep(180)