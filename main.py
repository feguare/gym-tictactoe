#!/usr/bin/env python
from examples.base_agent import BaseAgent
from minimax import MinimaxAgent
from oponent import Oponent

from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, next_mark

def play(max_episode=10):
    start_mark = 'O'
    env = TicTacToeEnv()
    agents = [MinimaxAgent('O'),
              Oponent('X')]

    for _ in range(max_episode):
        env.set_start_mark(start_mark)
        state = env.reset()
        while not env.done:
            _, mark = state
            env.show_turn(True, mark)
            
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, _, _ = env.step(action)
            env.render()

        env.show_result(True, mark, reward)

        # rotate start
        start_mark = next_mark(start_mark)



if __name__ == '__main__':
    play()
