import uuid
from typing import List, Tuple, Any, Dict, NewType
import numpy as np
import random
import chess
from tqdm import tqdm
import treelib
from pyvis.network import Network

from annotated_types.test_cases import cases
from pyarrow.types import is_boolean

from uci_to_idx import uci_to_idx

C_PUCT = 1.4


class StateManager:
    states: set


class State:
    priors: Dict[chess.Move, float]
    children: Dict[chess.Move, Any]
    parent: Any
    board: chess.Board
    value: float
    n_visits: int
    is_terminal: bool

    def __init__(self, parent: Any, board: chess.Board, priors: Dict[chess.Move, float], value: float,
                 is_terminal: bool = False):
        self.children = {}
        self.parent = parent
        self.board = board
        self.value = value
        self.priors = priors
        self.n_visits = 0
        self.is_terminal = is_terminal

    def return_random_child(self):
        chosen_move = random.choices(population=list(self.priors.keys()), weights=list(self.priors.values()), k=1)[0]
        return self.children[chosen_move]

    def return_ucb_move(self):
        if self.is_terminal:
            return None

        ucb_values = [(self.ucb_value(move), move) for move in self.priors.keys()]
        ucb, chosen_move = max(ucb_values, key=lambda x: x[0])
        return chosen_move

    def ucb_value(self, move):
        if move in self.children:
            return self.children[move].value + C_PUCT * self.priors[move] / (1 + self.children[move].n_visits)
        else:
            return 10000

    def move_leads_to_leaf(self, move):
        return move not in self.children

    def get_child(self, move):
        return self.children[move]


class Search:
    def __init__(self):
        pass

    def dummy_nn(self, board: chess.Board):
        policy = {move: np.random.rand() for move in board.legal_moves}
        # normalize policy
        policy_sum = sum(policy.values())
        policy = {move: prob / policy_sum for move, prob in policy.items()}

        return policy, np.random.rand()

    # def propagate_up(self, state: State):
    #     state.n_visits += 1
    #     total = 0
    #     total += state.value
    #     for child in state.children.values():
    #         total += child.value * child.n_visits
    #     # state.value = ((state.value * state.n_visits - 1) + state.value) / state.n_visits
    #     if state.parent is not None:
    #         self.propagate_up(state.parent)

    def propagate_up(self, state: State):
        """
        Increments n_visits, updates the state's value
        to be the average of its children's values (plus itself),
        then recurses upward.
        """
        if state is None:
            return

        # Count current node as visited
        state.n_visits += 1

        # If it has children, compute an average.
        if state.children:
            total_value = 0
            total_visits = 0
            for child in state.children.values():
                total_value += child.value * child.n_visits
                total_visits += child.n_visits

            # Include the current node’s own value & visit
            total_value += state.value
            total_visits += 1

            # Update current node’s value as an average
            state.value = total_value / total_visits

        # Now recurse upward
        if state.parent is not None:
            self.propagate_up(state.parent)

    def puct_search(self, state: State):
        for _ in tqdm(range(100000)):
            curr_state = state
            move = curr_state.return_ucb_move()

            while not curr_state.move_leads_to_leaf(move):
                curr_state = curr_state.get_child(move)
                if curr_state.is_terminal:
                    break
                move = curr_state.return_ucb_move()

            if curr_state.is_terminal:
                self.propagate_up(curr_state)
                continue

            board = curr_state.board.copy()
            board.push(move)

            is_terminal = board.outcome(claim_draw=True) is not None

            value = 0
            if is_terminal:
                match board.outcome(claim_draw=True).result():
                    case "1-0":
                        value = 100
                    case "0-1":
                        value = -100
                    case _:
                        value = 0
            else:
                priors, value = self.dummy_nn(board)

            new_state = State(parent=curr_state, board=board, value=value, priors=priors, is_terminal=is_terminal)
            curr_state.children[move] = new_state

            self.propagate_up(curr_state)
            curr_state = new_state

        return state.return_ucb_move(), state.value


def main():
    search = Search()
    board = chess.Board(fen='7r/r1kq1p2/1p1p3p/1Q2p1p1/2P5/6P1/2P2PP1/1R4K1 w - - 0 30')
    # print(board)
    p, v = search.dummy_nn(board)
    initial_state = State(None, board, p, v, False)
    move, value = search.puct_search(initial_state)
    # print([(move, child.value) for move, child in initial_state.children.items()])
    print(move, value)

    # print tree
    tree = treelib.Tree()
    startnode = tree.create_node(tag=str(initial_state.board), identifier=str(initial_state.board))

    def print_tree(state: State, root_node_id=None):
        for move, child in state.children.items():
            node = tree.create_node(tag=str(move), identifier=str(uuid.uuid1()), parent=root_node_id)
            print_tree(child, node.identifier)

    def visualize_tree_pyvis(tree: treelib.Tree):
        # Create a PyVis network object
        net = Network(directed=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

        # Add nodes and edges from the tree
        for node in tree.all_nodes():
            net.add_node(n_id=node.identifier, label=node.tag)
        for node in tree.all_nodes():
            if node.predecessor(tree.identifier):  # If the node has a parent, add an edge
                net.add_edge(node.predecessor(tree.identifier), node.identifier)

        # Configure physics for better layout
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])


        # Save and show the visualization
        net.write_html("tree_visualization.html")
        # net.show("tree_visualization.html")
    # Visualize the tree
    print_tree(initial_state, startnode.identifier)
    tree.to_graphviz("tree.dot")
    visualize_tree_pyvis(tree)


if __name__ == '__main__':
    main()
