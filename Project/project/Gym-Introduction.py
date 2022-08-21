import gym
import numpy as np
from gym import wrappers ### for saving the file

env = gym.make('CartPole-v0')  ### making the environment
best_length = 0  ### average of best lengths, may be updated with running each game
episode_lengths = []  ### average of number of runs after running each game
best_weights = np.zeros(4) ### may be updated with running each game, will be used in the final game
######################
###Searching Part###
######################
for i in range(100): ### 100 different set of weights.
    weights = 2 * np.random.rand(4) - 1
    length = [] ### a list for the length of runs for each game
    # We are going to play the game 100 times for each set of weights and check the average length of trajectories for each one.
    for j in range(100):  ### 100 different run for each set of random weights
        observation = env.reset()
        done = False ### check when the game is finished
        cnt = 0 ###length of the trajectory for each the game
        while not done:
            cnt += 1
            if np.dot(weights, observation) >= 0:
                action = 1
            else:
                action = 0
            observation, reward, done, info = env.step(action)
            if done:
                break
        length.append(cnt)
    average_length = sum(length) / len(length)  ###calculate the average length
    if average_length > best_length:
        best_length = average_length
        best_weights = weights
    episode_lengths.append(average_length)
    ### print the best length every 10 games
    if i % 10 == 0:
        print('The best length is: ' + str(best_length))
##################
###final game###
##################
done = False
cnt = 0
env = wrappers.Monitor(env, 'cartpole', force=True)
observation = env.reset() ###reset the environment
### play the game with best weights
while not done:
    cnt += 1
    if np.dot(best_weights, observation) >= 0:
        action = 1
    else:
        action = 0
    observation, reward, done, info = env.step(action)
    if done:
        break
### print number of runs needed for final game
print('game lasted ' + str(cnt) + ' moves')