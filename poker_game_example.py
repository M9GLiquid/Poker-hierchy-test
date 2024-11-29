import poker_environment as pe_
from poker_environment import AGENT_ACTIONS, BETTING_ACTIONS
import copy

class GameNode:
    def __init__(self, state, parent=None, action=None, g_cost=0, h_cost=0):
        self.state = state            # Reference to the GameState object
        self.parent = parent          # Reference to the parent node
        self.children = []            # List of child nodes
        self.action = action          # Action leading to this state
        self.g_cost = g_cost          # Cost to reach this node (used in A*)
        self.h_cost = h_cost          # Heuristic estimate to the goal (used in A*)
        self.f_cost = g_cost + h_cost # Total cost (used for A*)
        self.depth = parent.depth + 1 if parent else 0  # Depth in the treeclear

    def add_child(self, child):
        self.children.append(child)

    def reconstruct_path(self):
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return path[::-1]  # Reverse to get root-to-leaf order

"""
Player class
"""
class PokerPlayer(object):
    def __init__(self, current_hand_=None, stack_=300, action_=None, action_value_=None):
        self.current_hand = current_hand_
        self.current_hand_type = []
        self.current_hand_strength = []
        self.stack = stack_
        self.action = action_
        self.action_value = action_value_

    """
    identify agent hand and evaluate it's strength
    """
    def evaluate_hand(self):
        self.current_hand_type = pe_.identify_hand(self.current_hand)
        self.current_hand_strength = pe_.Types[self.current_hand_type[0]]*len(pe_.Ranks) + pe_.Ranks[self.current_hand_type[1]]

    """
    return possible actions, fold if there is not enough money...
    """
    def get_actions(self):
        actions_ = []
        for _action_ in AGENT_ACTIONS:
            if _action_[:3] == 'BET' and int(_action_[3:])>=(self.stack):
                actions_.append('FOLD')
            else:
                actions_.append(_action_)
        return set(actions_)

"""
Game State class
"""
class GameState(object):
    def __init__(self,
                 nn_current_hand_=None,
                 nn_current_bidding_=None,
                 phase_ = None,
                 pot_=None,
                 acting_agent_=None,
                 parent_state_=None,
                 children_state_=None,
                 agent_=None,
                 opponent_=None):

        self.nn_current_hand = nn_current_hand_
        self.nn_current_bidding = nn_current_bidding_
        self.phase = phase_
        self.pot = pot_
        self.acting_agent = acting_agent_
        self.parent_state = parent_state_
        self.children = children_state_
        self.agent = agent_
        self.opponent = opponent_
        self.showdown_info = None

    """
    draw 10 cards randomly from a deck
    """
    def dealing_cards(self):

        if self.nn_current_hand >= 20:
            print("random hand  ", self.nn_current_hand)
            # randomly generated hands
            agent_hand, opponent_hand = pe_.generate_2hands()
        else:
            # fixed_hands or use the function below
            print("fixed hand ", self.nn_current_hand)
            agent_hand, opponent_hand = pe_.fixed_hands[self.nn_current_hand]
        self.agent.current_hand = agent_hand
        self.agent.evaluate_hand()
        self.opponent.current_hand = opponent_hand
        self.opponent.evaluate_hand()

    """
    draw 10 cards from a fixed sequence of hands
    """
    def dealing_cards_fixed(self):
        self.agent.current_hand = pe_.fixed_hands[self.nn_current_hand][0]
        self.agent.evaluate_hand()
        self.opponent.current_hand = pe_.fixed_hands[self.nn_current_hand][1]
        self.opponent.evaluate_hand()

    """
    SHOWDOWN phase, assign pot to players
    """
    def showdown(self):

        if self.agent.current_hand_strength == self.opponent.current_hand_strength:
            self.showdown_info = 'draw'
            if self.acting_agent == 'agent':
                self.agent.stack += (self.pot - 5) / 2. + 5
                self.opponent.stack += (self.pot - 5) / 2.
            else:
                self.agent.stack += (self.pot - 5) / 2.
                self.opponent.stack += (self.pot - 5) / 2. + 5
        elif self.agent.current_hand_strength > self.opponent.current_hand_strength:
            self.showdown_info = 'agent win'
            self.agent.stack += self.pot
        else:
            self.showdown_info = 'opponent win'
            self.opponent.stack += self.pot

    # print out necessary information of this game state
    def print_state_info(self):

        print('************* state info **************')
        print('nn_current_hand', self.nn_current_hand)
        print('nn_current_bidding', self.nn_current_bidding)
        print('phase', self.phase)
        print('pot', self.pot)
        print('acting_agent', self.acting_agent)
        print('parent_state', self.parent_state)
        print('children', self.children)
        print('agent', self.agent)
        print('opponent', self.opponent)

        if self.phase == 'SHOWDOWN':
            print('---------- showdown ----------')
            print('agent.current_hand', self.agent.current_hand)
            print(self.agent.current_hand_type, self.agent.current_hand_strength)
            print('opponent.current_hand', self.opponent.current_hand)
            print(self.opponent.current_hand_type, self.opponent.current_hand_strength)
            print('showdown_info', self.showdown_info)

        print('----- agent -----')
        print('agent.current_hand',self.agent.current_hand)
        print('agent.current_hand_type',self.agent.current_hand_type)
        print('agent.current_hand_strength',self.agent.current_hand_strength)
        print('agent.stack',self.agent.stack)
        print('agent.action',self.agent.action)
        print('agent.action_value',self.agent.action_value)

        print('----- opponent -----')
        print('opponent.current_hand', self.opponent.current_hand)
        print('opponent.current_hand_type',self.opponent.current_hand_type)
        print('opponent.current_hand_strength',self.opponent.current_hand_strength)
        print('opponent.stack',self.opponent.stack)
        print('opponent.action',self.opponent.action)
        print('opponent.action_value',self.opponent.action_value)
        print('**************** end ******************')

# copy given state in the argument
def copy_state(game_state):
    _state = copy.copy(game_state)
    _state.agent = copy.copy(game_state.agent)
    _state.opponent = copy.copy(game_state.opponent)
    return _state

"""
successor function for generating next state(s)
"""
def get_next_states(last_state):

    if last_state.phase == 'SHOWDOWN' or last_state.acting_agent == 'opponent' or last_state.phase == 'INIT_DEALING':

        # NEW BETTING ROUND, AGENT ACT FIRST

        states = []

        for _action_ in last_state.agent.get_actions():

            _state_ = copy_state(last_state)
            _state_.acting_agent = 'agent'

            if last_state.phase == 'SHOWDOWN' or last_state.phase == 'INIT_DEALING':
                #_state_.dealing_cards()
                _state_.dealing_cards_fixed()

            if _action_ == 'CALL':

                _state_.phase = 'SHOWDOWN'
                _state_.agent.action = _action_
                _state_.agent.action_value = 5
                _state_.agent.stack -= 5
                _state_.pot += 5

                _state_.showdown()

                _state_.nn_current_hand += 1
                _state_.nn_current_bidding = 0
                _state_.pot = 0
                _state_.parent_state = last_state
                states.append(_state_)

            elif _action_ == 'FOLD':

                _state_.phase = 'SHOWDOWN'
                _state_.agent.action = _action_
                _state_.opponent.stack += _state_.pot

                _state_.nn_current_hand += 1
                _state_.nn_current_bidding = 0
                _state_.pot = 0
                _state_.parent_state = last_state
                states.append(_state_)


            elif _action_ in BETTING_ACTIONS:

                _state_.phase = 'BIDDING'
                _state_.agent.action = _action_
                _state_.agent.action_value = int(_action_[3:])
                _state_.agent.stack -= int(_action_[3:])
                _state_.pot += int(_action_[3:])

                _state_.nn_current_bidding += 1
                _state_.parent_state = last_state
                states.append(_state_)

            else:

                print('...unknown action...')
                exit()

        return states

    elif last_state.phase == 'BIDDING' and last_state.acting_agent == 'agent':

        states = []
        _state_ = copy_state(last_state)
        _state_.acting_agent = 'opponent'

        opponent_action, opponent_action_value = pe_.poker_strategy_example(last_state.opponent.current_hand_type[0],
                                                                            last_state.opponent.current_hand_type[1],
                                                                            last_state.opponent.stack,
                                                                            last_state.agent.action,
                                                                            last_state.agent.action_value,
                                                                            last_state.agent.stack,
                                                                            last_state.pot,
                                                                            last_state.nn_current_bidding)

        if opponent_action =='CALL':

            _state_.phase = 'SHOWDOWN'
            _state_.opponent.action = opponent_action
            _state_.opponent.action_value = 5
            _state_.opponent.stack -= 5
            _state_.pot += 5

            _state_.showdown()

            _state_.nn_current_hand += 1
            _state_.nn_current_bidding = 0
            _state_.pot = 0
            _state_.parent_state = last_state
            states.append(_state_)

        elif opponent_action == 'FOLD':

            _state_.phase = 'SHOWDOWN'

            _state_.opponent.action = opponent_action
            _state_.agent.stack += _state_.pot

            _state_.nn_current_hand += 1
            _state_.nn_current_bidding = 0
            _state_.pot = 0
            _state_.parent_state = last_state
            states.append(_state_)

        elif opponent_action + str(opponent_action_value) in BETTING_ACTIONS:

            _state_.phase = 'BIDDING'

            _state_.opponent.action = opponent_action
            _state_.opponent.action_value = opponent_action_value
            _state_.opponent.stack -= opponent_action_value
            _state_.pot += opponent_action_value

            _state_.nn_current_bidding += 1
            _state_.parent_state = last_state
            states.append(_state_)

        else:
            print('unknown_action')
            exit()
        return states

"""
    Builds a game tree starting from the initial state up to max_depth.

    Args:
        initial_state (GameState): The root game state.
        max_depth (int): The maximum depth of the tree.

    Returns:
        GameNode: The root of the constructed tree.
"""
def create_tree(initial_state):
    # Create the root node
    root = GameNode(state=initial_state)
    node_count = 1

    # Use a stack for tree construction (LIFO for DFS-style expansion)
    stack = [root]  # Start with the root node

    while stack:
        current_node = stack.pop()  # Process the top node from the stack

        # Stop expanding if max depth is reached
        if is_game_over(current_node.state):
            continue

        # Generate child states
        child_states = get_next_states(current_node.state)

        for child_state in child_states:
            # Create a child node
            child_node = GameNode(
                state=child_state,
                parent=current_node,
                action=child_state.agent.action
            )

            # Add the child node to the current node's children
            current_node.add_child(child_node)
            node_count += 1

            # Add the child node to the stack for further processing
            stack.append(child_node)

    return root, node_count

"""
    Recursively prints the tree structure in a simplified format.

    Args:
        node (GameNode): The node to print.
        indent (str): Indentation for the current node.
        is_last (bool): Whether this node is the last child of its parent.
"""
def print_tree(node, indent="", is_last=True):
    connector = "└──" if is_last else "├──"
    acting_agent = node.state.acting_agent or "None"

    print(f"{indent}{connector} Action={node.action or 'None'}"
          f" Acting={acting_agent}, Hand={node.state.nn_current_hand}")

    # Update indentation for the children
    indent += "    " if is_last else "│   "

    # Print children recursively
    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        print_tree(child, indent, is_last_child)

"""
    Checks if the game is over based on the current state.
"""
def is_game_over(state):
    agent_won = (state.agent.stack - INIT_AGENT_STACK) > 100
    opponent_out_of_money = state.opponent.stack <= 0
    agent_out_of_money = state.agent.stack <= 0
    max_hands_reached = state.nn_current_hand >= MAX_HANDS
    return agent_won or opponent_out_of_money or agent_out_of_money or max_hands_reached

"""
Game flow:
Two agents will keep playing until one of them lose 100 coins or more.
"""

MAX_HANDS = 4
INIT_AGENT_STACK = 400

# initialize 2 agents and a game_state
for i in range(1, 5):
    MAX_HANDS = i
    agent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)
    opponent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)


    init_state = GameState(nn_current_hand_=0,
                        nn_current_bidding_=0,
                        phase_ = 'INIT_DEALING',
                        pot_=0,
                        acting_agent_=None,
                        agent_=agent,
                        opponent_=opponent,
                        )

    # Build the tree up to depth 3
    tree_root, node_count = create_tree(init_state)

    # Print the tree
    #print_tree(tree_root)
    print(f"Total nodes: {node_count}, Max Hands: {MAX_HANDS}")

quit()
game_state_queue = []
game_on = True
round_init = True

while game_on:

    if round_init:
        round_init = False
        states_ = get_next_states(init_state)
        game_state_queue.extend(states_[:])
    else:

        # just an example: only expanding the last return node
        states_ = get_next_states(states_[-1])
        game_state_queue.extend(states_[:])

        for _state_ in states_:
            if is_game_over(_state_):
                end_state_ = _state_
                game_on = False

"""
Printing game flow & info
"""
state__ = end_state_
nn_level = 0

print('------------ print game info ---------------')
print('nn_states_total', len(game_state_queue))

while state__.parent_state != None:
    nn_level += 1
    print("Level:", nn_level)
    state__.print_state_info()
    state__ = state__.parent_state

print(nn_level)


"""
Perform searches
"""