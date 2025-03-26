#!/usr/bin/env python
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, next_mark


class MinimaxAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def _minimax(self, state, ava_actions, is_maximising, depth):
        # factor = 1 if is_maximising else -1  # always maximise below so this will be used to reverse the order for minimising values
        gstatus = check_game_status(state[0])
        # if its a draw
        if gstatus == 0:
            # return -5 * factor, None
            # return -5 if is_maximising esle 5, None
            return 0, None #### ***************** DISCUSS HOW CHANGING THIS AFFECTED THE FREQUENCY OF DRAWS [IF ITS -10, ITS THE SAME AS LOSING SO ITLL JUST DO WHICHEVER IT SEES FIRST INSTEAD OF FAVOURING THE DRAW, EVRYTHING ELSE SEEMS TO GIVE ABOUT 40% DRAW]
        # if the game has been won (by the previous player as they just moved)
        elif gstatus > 0:
            return -10 if is_maximising else 10, None  # who just moved?? if its not this player then reversE returns 
            # return -10 * factor, None 
        else:
            best = float('-inf') if is_maximising else float('inf')
            # best = float('-inf')
            best_action = None

            for action in ava_actions:
                nstate = after_action_state(state, action)
                nava_actions = ava_actions.copy()
                nava_actions.remove(action)
                val, _ = self._minimax(nstate, nava_actions, not is_maximising, depth+1) 
                # if val * factor > best:
                #     best = val
                #     best_action = action
                if is_maximising and val > best:
                    best = val
                    best_action = action
                elif not is_maximising and val < best:
                    best = val 
                    best_action = action
            
            return best, best_action
        
    def act(self, state, ava_actions):
        _, action = self._minimax(state, ava_actions, False, 0)
        print(action)
        return action


def play(max_episode=10):
    start_mark = 'O'
    env = TicTacToeEnv()
    agents = [MinimaxAgent('O'),
              MinimaxAgent('X')]

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
