import poker_environment as pe_
from poker_environment import AGENT_ACTIONS, BETTING_ACTIONS
from collections import deque
import heapq
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

    # Add a child node to the current node
    def add_child(self, child):
        self.children.append(child)

    # Reconstruct the path from the root to the current node
    def reconstruct_path(self):
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return path[::-1]  # Reverse to get root-to-leaf order

    # Compare nodes by their f_cost
    def __lt__(self, other):
        """Compare nodes by their f_cost."""
        return self.f_cost < other.f_cost

# Poker Player class
class PokerPlayer(object):
    def __init__(self, current_hand_=None, stack_=300, action_=None, action_value_=None):
        self.current_hand = current_hand_
        self.current_hand_type = []
        self.current_hand_strength = []
        self.stack = stack_
        self.action = action_
        self.action_value = action_value_

    # Evaluate the current hand
    def evaluate_hand(self):
        self.current_hand_type = pe_.identify_hand(self.current_hand)
        self.current_hand_strength = pe_.Types[self.current_hand_type[0]]*len(pe_.Ranks) + pe_.Ranks[self.current_hand_type[1]]

    # Get the available actions for the player
    def get_actions(self):
        actions_ = []
        for _action_ in AGENT_ACTIONS:
            if _action_[:3] == 'BET' and int(_action_[3:])>=(self.stack):
                actions_.append('FOLD')
            else:
                actions_.append(_action_)
        return set(actions_)

# Game State class
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

    # Deal cards to the players
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

    # Deal fixed cards to the players
    def dealing_cards_fixed(self):
        self.agent.current_hand = pe_.fixed_hands[self.nn_current_hand][0]
        self.agent.evaluate_hand()
        self.opponent.current_hand = pe_.fixed_hands[self.nn_current_hand][1]
        self.opponent.evaluate_hand()

    # Determine the winner of the showdown
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

# Get the next possible states from the current state
def get_next_states(last_state):

    # agent acts first if it is a new betting round
    if last_state.phase == 'SHOWDOWN' or \
            last_state.acting_agent == 'opponent' or \
            last_state.phase == 'INIT_DEALING':

        # NEW BETTING ROUND, AGENT ACT FIRST
        states = []

        # agent acts
        for _action_ in last_state.agent.get_actions():

            _state_ = copy_state(last_state)
            _state_.acting_agent = 'agent'

            # Deal cards
            if last_state.phase == 'SHOWDOWN' or last_state.phase == 'INIT_DEALING':
                #_state_.dealing_cards()
                _state_.dealing_cards_fixed()

            # agent calls
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
            # agent folds
            elif _action_ == 'FOLD':
                _state_.phase = 'SHOWDOWN'
                _state_.agent.action = _action_
                _state_.opponent.stack += _state_.pot

                _state_.nn_current_hand += 1
                _state_.nn_current_bidding = 0
                _state_.pot = 0
                _state_.parent_state = last_state
                states.append(_state_)

            # agent bets
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
    # opponent acts
    elif last_state.phase == 'BIDDING' and last_state.acting_agent == 'agent':
        states = []
        _state_ = copy_state(last_state)
        _state_.acting_agent = 'opponent'

        # Get the opponent's action based on the current state and strategy
        opponent_action, opponent_action_value = pe_.poker_strategy_example(last_state.opponent.current_hand_type[0],
                                                                            last_state.opponent.current_hand_type[1],
                                                                            last_state.opponent.stack,
                                                                            last_state.agent.action,
                                                                            last_state.agent.action_value,
                                                                            last_state.agent.stack,
                                                                            last_state.pot,
                                                                            last_state.nn_current_bidding)

        # opponent calls
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
        # opponent folds
        elif opponent_action == 'FOLD':
            _state_.phase = 'SHOWDOWN'

            _state_.opponent.action = opponent_action
            _state_.agent.stack += _state_.pot

            _state_.nn_current_hand += 1
            _state_.nn_current_bidding = 0
            _state_.pot = 0
            _state_.parent_state = last_state
            states.append(_state_)
        # opponent bets
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

# Create the game tree starting from the initial state
def create_tree(initial_state):
    # Create the root node
    root = GameNode(state=initial_state)
    node_count = 1

    # Use a stack for tree construction (LIFO for DFS-style expansion)
    stack = deque([root])

    while stack:
        # Pop the last node added to the stack
        current_node = stack.pop()

        # Don't expand if it's a terminal state
        if is_terminal_state(current_node.state):
            continue

        # Generate child states and sort them by action to maintain consistency
        child_states = sorted(
            get_next_states(current_node.state),
            key=lambda state: (
                state.agent.action if state.acting_agent == 'agent' else state.opponent.action
            )
        )

        for child_state in child_states:
            # Determine the action that led to this state
            action = (child_state.agent.action if child_state.acting_agent == 'agent'
                      else child_state.opponent.action)
            # Create a child node
            child_node = GameNode(
                state=child_state,
                parent=current_node,
                action=action
            )

            # Add the child node to the current node's children
            current_node.add_child(child_node)
            node_count += 1

            # Add the child node to the stack for further processing
            stack.append(child_node)

    return root, node_count

# A* search algorithm
def a_star(root):
    priority_queue = []
    heapq.heappush(priority_queue, (root.f_cost, root))
    expanded_nodes = 0

    while priority_queue:
        # Pop the node with the smallest f_cost
        _, current_node = heapq.heappop(priority_queue)
        expanded_nodes += 1

        # Check if the current state satisfies the goal condition
        if (current_node.state.agent.stack - INIT_AGENT_STACK) >= 100:
            return current_node.reconstruct_path(), expanded_nodes

        # Generate children and calculate their costs
        for child in current_node.children:
            child.g_cost = current_node.g_cost + 1
            child.h_cost = heuristic(child)
            child.f_cost = child.g_cost + child.h_cost
            heapq.heappush(priority_queue, (child.f_cost, child))

    # If no solution is found
    return None, expanded_nodes

# Heuristic function for A* search
def heuristic(node):
    state = node.state
    agent_stack = state.agent.stack
    opponent_stack = state.opponent.stack
    hands_remaining = MAX_HANDS - state.nn_current_hand

    # Heuristic based on the difference in stack sizes and hands remaining to play
    return (INIT_AGENT_STACK - agent_stack) - (INIT_AGENT_STACK - opponent_stack) + hands_remaining * 0.1

# Depth-first search algorithm
def dfs(root):
    stack = deque([root])  # Initialize the DFS stack with the root node
    expanded_nodes = 0  # Counter for expanded nodes

    while stack:
        current_node = stack.pop()  # Pop the last node added to the stack
        expanded_nodes += 1

        # Check if the current state satisfies the goal condition
        if (current_node.state.agent.stack - INIT_AGENT_STACK) >= 100:
            return current_node.reconstruct_path(), expanded_nodes

        # Enqueue child nodes for further exploration in reverse order (LIFO)
        stack.extend(current_node.children)

    # If no solution is found
    return None, expanded_nodes


# Breadth-first search algorithm
# Returns the path to the goal node and the number of expanded nodes
def bfs(root):
    queue = deque([root])  # BFS queue initialized with the root node
    expanded_nodes = 0  # Counter for expanded nodes

    while queue:
        current_node = queue.popleft()  # Dequeue the first node
        expanded_nodes += 1

        # Check if the agent has won more than 100
        if (current_node.state.agent.stack - INIT_AGENT_STACK) >= 100:
            return current_node.reconstruct_path(), expanded_nodes

        # Enqueue child nodes for further exploration
        for child in current_node.children:
            queue.append(child)

    # If no solution is found
    return None, expanded_nodes

def print_tree(node, indent="", is_last=True):
    connector = "└──" if is_last else "├──"
    acting_agent = node.state.acting_agent or "None"

    print(f"{indent}{connector} \
          Action={node.action or 'None'}"
          f" Acting={acting_agent}, \
            Hand={node.state.nn_current_hand}, \
                Phase={node.state.phase.lower()}, ")

    # Update indentation for the children
    indent += "  " if is_last else "│ "

    # Print children recursively
    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        print_tree(child, indent, is_last_child)

# Print the path from the root to the goal node
# The path is a list of nodes from the root to the goal node
def print_path(path):
    indent = ""  # Initialize the base indentation
    nodes = 0
    for i, node in enumerate(path):
        nodes += 1
        parent_action = "NULL" if node.parent is None else node.parent.action or "None"
        acting_agent = node.state.acting_agent or "None"
        if acting_agent == 'agent':
            stack = node.state.agent.stack
        elif acting_agent == 'opponent':
            stack = node.state.opponent.stack
        else:
            stack = 0

        action = node.action or "None"
        hand = node.state.nn_current_hand
        phase = node.state.phase


        connector = "└──" if i == len(path) - 1 else "└──"
        print(f"{indent}{connector}"
              f"[{phase[0]}][H:{hand}] - {acting_agent}[{action}], ${stack}, Pot={node.state.pot}")

        # Update indentation for the next level
        indent += "  " if i < len(path) - 1 else ""
    print(f'Agent Stack: {path[-1].state.agent.stack}')
    print(f'Opponent Stack: {path[-1].state.opponent.stack}')
    print(f'Nodes traversed: {nodes}')

# Check if the state is terminal (game over)
# The game is over if the agent has won more tha 100,
# the opponent or agent is out of money,
# or the maximum number of hands has been reached
def is_terminal_state(state):
    agent_won = (state.agent.stack - INIT_AGENT_STACK) >= 100
    opponent_out_of_money = state.opponent.stack <= 0
    agent_out_of_money = state.agent.stack <= 0
    max_hands_reached = state.nn_current_hand >= MAX_HANDS
    return agent_won or opponent_out_of_money or agent_out_of_money or max_hands_reached

################
#  Game Flow   #
################
MAX_HANDS = 0
INIT_AGENT_STACK = 100

# Play MAX_HANDS hands of poker with 2 agents
for i in range(1, 5):
    MAX_HANDS = i
    agent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)
    opponent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)

    init_state = GameState(
        nn_current_hand_ = 0,
        nn_current_bidding_ = 0,
        phase_ = 'INIT_DEALING',
        pot_ = 0,
        acting_agent_ = 'Start',
        agent_ = agent,
        opponent_ = opponent,
    )

    # Create the game tree starting from the initial state
    tree_root, node_count = create_tree(init_state)

    # Debug the tree
    #if MAX_HANDS < 2:
    #    print_tree(tree_root)

    # Perform BFS on the tree
    path1, expanded_nodes1 = bfs(tree_root)
    path2, expanded_nodes2 = dfs(tree_root)
    path3, expanded_nodes3 = a_star(tree_root)

    # Display results
    print('------------ print game info ---------------')
    if path1:
        print('[BFS path]')
        print_path(path1)
    if path2:
        print('[DFS path]')
        print_path(path2)
    if path3:
        print('[A* path]')
        print_path(path3)

    print(f"Total nodes: {node_count}, Max Hands: {MAX_HANDS},")
    print(f'BFS nodes: {expanded_nodes1}, DFS nodes: {expanded_nodes2}, A* nodes: {expanded_nodes3}')
