#!/usr/bin/env python
from examples.base_agent import BaseAgent
from oponent import Oponent

from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark

def play(max_episode=10):
    start_mark = 'O'
    env = TicTacToeEnv()
    agents = [Oponent('O'),
              BaseAgent('X')]

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
