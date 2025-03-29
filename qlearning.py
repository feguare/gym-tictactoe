#!/usr/bin/env python
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, next_mark, tocode
from oponent import Oponent
from itertools import product
from copy import deepcopy
from random import random, choice


class QLearningAgent(object):
    def __init__(self, mark, all_actions):
        self.mark = mark
        self.all_actions = all_actions

        # initialise Q table
        all_states = product([0, 1, 2], repeat=len(all_actions))
        state_vals = {state: 0 for state in all_states}
        self.qVals = {action: deepcopy(state_vals) for action in all_actions}

    def qBellman(self, action, state, alpha, gamma, reward, next_val):
        return self.qVals[action][state] + alpha * (reward + gamma * next_val - self.qVals[action][state])

    def train(self, initial_state, num_iters, epsilon, epsilon_decay, alpha, gamma, oponent):
        start_mark = self.mark
        d_count = 0  # draw counter
        w_count = 0  # win counter
        l_count = 0  # loss counter

        for iter in range(num_iters):
            ava_actions = self.all_actions.copy()
            state = deepcopy(initial_state[0]), start_mark
            prevAction = None
            prevBoard = None
            while check_game_status(state[0]) == -1:
                board, mark = state
                
                if mark is not self.mark:
                    action = oponent.act(state, ava_actions)
                else:
                    # exploit
                    action, next_val = self.act(state, ava_actions, True)
                    # update Q table for previous move 
                    if prevAction is not None and prevBoard:  # just skip first move
                        self.qVals[prevAction][prevBoard] = self.qBellman(prevAction, prevBoard, alpha, gamma, 0, next_val)
                    # explore
                    if random() < epsilon:
                        action = choice(ava_actions)
                    prevBoard = board
                    prevAction = action

                state = after_action_state(state, action)
                ava_actions.remove(action)

            # terminal update of Q table
            result = check_game_status(state[0])
            if result == 0:
                reward = 0
                d_count += 1
            elif result == tocode(self.mark):
                reward = 1
                w_count += 1
            else:
                reward = -1
                l_count += 1

            self.qVals[prevAction][prevBoard] = self.qBellman(prevAction, prevBoard, alpha, gamma, reward, 0)

            # rotate start
            start_mark = next_mark(start_mark)
            epsilon *= 1 - epsilon_decay
            if (iter + 1) % 5000 == 0:
                print("iter: ", iter, "wins:", w_count, "draws:", d_count, "losses:", l_count)
            # print("iter: ", iter, "wins:", w_count, "draws:", d_count, "losses:", l_count) if iter%5000 == 0 else True

    def act(self, state, ava_actions, isTraining = False):
        # exploit
        maxQ = float('-inf')
        best = None
        for action in ava_actions:
            q = self.qVals[action][state[0]]
            if q > maxQ:
                maxQ = q
                best = action

        if isTraining:
            return best, maxQ 
        else:
            return best


def play(max_episode=10):
    start_mark = 'O'
    env = TicTacToeEnv()
    qAgent = QLearningAgent('O', env.available_actions())
    oponent = Oponent('X')
    # qAgent.train(env.reset(), 100000, 0.7, 0, 0.7, 0.9, oponent) # the non decaying epsilon seems to have, more wins instead of draws like the decaying
    qAgent.train(env.reset(), 100000, 1, 0.01, 0.7, 0.9, oponent)
    agents = [qAgent,
              oponent]
    

    for _ in range(max_episode):
        env.set_start_mark(start_mark)
        state = env.reset()
        while not env.done:
            _, mark = state
            env.show_turn(True, mark)
            
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
            env.render()

        env.show_result(True, mark, reward)

        # rotate start
        start_mark = next_mark(start_mark)


if __name__ == '__main__':
    play()
