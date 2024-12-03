################
#  Game Flow   #
################
from poker_environment import INIT_AGENT_STACK, MAX_HANDS
from poker_game_example import GameState, PokerPlayer
from search_algorithm import a_star, bfs, create_tree, dfs, print_path, print_tree

DEBUG = False

# Play MAX_HANDS hands of poker with 2 agents
for hand in range(1, MAX_HANDS + 1):
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
    tree_root, node_count = create_tree(init_state, hand=hand)

    # Debug the tree
    if DEBUG and MAX_HANDS < 2:
        print_tree(tree_root)

    # Perform BFS on the tree
    path1, expanded_nodes1 = bfs(tree_root)
    path2, expanded_nodes2 = dfs(tree_root)
    path3, expanded_nodes3 = a_star(tree_root)

    # Display results
    print('------------ print game info ---------------')
    if DEBUG:
        if path1:
            print('[BFS path]')
            print_path(path1)
        if path2:
            print('[DFS path]')
            print_path(path2)
        if path3:
            print('[A* path]')
            print_path(path3)

    print(f"Total States: {node_count}, Max Hands: {hand}")
    print(f'BFS nodes: {expanded_nodes1}, DFS nodes: {expanded_nodes2}, A* nodes: {expanded_nodes3}')
