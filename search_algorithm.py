from collections import deque
import numpy as np
import math
import heapq

from poker_game_example import get_next_states
from poker_environment import MAX_HANDS, INIT_AGENT_STACK


# Priority Queue based on heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []
    def isEmpty(self):
        return len(self.elements) == 0
    def add(self, item, priority):
        heapq.heappush(self.elements,(priority,item))
    def remove(self):
        return heapq.heappop(self.elements)[1]

# Node for tree
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

    # Create the game tree starting from the initial state
def create_tree(initial_state, hand):
    # Create the root node
    root = GameNode(state=initial_state)
    node_count = 1

    # Use a stack for tree construction (LIFO for DFS-style expansion)
    stack = deque([root])

    while stack:
        # Pop the last node added to the stack
        current_node = stack.pop()

        # Don't expand if it's a terminal state
        if is_terminal_state(current_node.state, hand):
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
    # Initialize the priority queue with the root node
    priority_queue = PriorityQueue()
    priority_queue.add(root, root.f_cost)
    expanded_nodes = 0

    while not priority_queue.isEmpty():
        # Pop the node with the smallest f_cost
        current_node = priority_queue.remove()
        #_, current_node = heapq.heappop(priority_queue)
        expanded_nodes += 1

        # Check if the current state satisfies the goal condition
        if (current_node.state.agent.stack - INIT_AGENT_STACK) >= 100:
            return current_node.reconstruct_path(), expanded_nodes

        # Generate children and calculate their costs
        for child in current_node.children:
            child.g_cost = current_node.g_cost + 1
            child.h_cost = heuristic(child)
            child.f_cost = child.g_cost + child.h_cost
            priority_queue.add(child, child.f_cost)
            #heapq.heappush(priority_queue, (child.f_cost, child))

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
def is_terminal_state(state, hand):
    agent_won = (state.agent.stack - INIT_AGENT_STACK) >= 100
    opponent_out_of_money = state.opponent.stack <= 0
    agent_out_of_money = state.agent.stack <= 0
    max_hands_reached = state.nn_current_hand >= hand
    return agent_won or opponent_out_of_money or agent_out_of_money or max_hands_reached
