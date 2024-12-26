from typing import List, Tuple, Any, Dict
import numpy as np
import random
import chess
from dill.pointers import children


class StateManager:
    states: set


class State:
    priors: Dict[chess.Move, float]
    children: Dict[chess.Move, Any]
    board: chess.Board
    value: float
    n_visits: int

    def __init__(self, board, value: float, priors):
        self.children = {}
        self.board = board
        self.value = value
        self.priors = priors
        self.n_visits = 0

    def return_random_child(self):
        chosen_move = random.choices(population=list(self.priors.keys()), weights=list(self.priors.values()), k=1)[0]
        return self.children[chosen_move]

    def return_ucb_child(self):
        ucb_values = [self.ucb_value(move) if move in self.children else self.priors[move] for move in self.priors.keys()]
        chosen_move = list(self.priors.keys())[np.argmax(ucb_values)]
        return self.children[chosen_move]

    def ucb_value(self, move):
        return self.children[move].value + 1.41 * self.priors[move] / (1 + self.children[move].n_visits)

    def move_leads_to_leaf(self, move):
        return move not in self.children

    def get_child(self, move):
        return self.children[move]

class Search:
    def __init__(self):
        pass

    def dummy_nn(self, state: chess.Board):
        return np.random.rand(7), np.random.rand(1)

    def puct_search(self, state: State):
        move = 'a1a2'

        curr_state = state

        mv = curr_state.return_ucb_child()

        # TODO Implement check for terminal node
        while curr_state.move_leads_to_leaf(mv):
            curr_state = curr_state.get_child(mv)
            mv = curr_state.return_ucb_child()

        # TODO Expand node


        return move, 0.5


def main():
    board = chess.Board()
    search = Search()
    search.puct_search()
    print(board)


if __name__ == '__main__':
    main()