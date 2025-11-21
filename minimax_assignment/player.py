#!/usr/bin/env python3
import random
import time
import math

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node : Node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        max_depth = 1
        # Transposition table: key -> (stored_depth, value)
        tt = {} # {(key): (depth, value)}
        
        def make_key(node: Node, is_max_turn: bool):
            
            s = node.state
            p0, p1 = s.get_player_scores()
            hooks = s.get_hook_positions()
            my_hook = hooks[0]
            opp_hook = hooks[1]
            caught = s.get_caught()
            fish_positions = s.get_fish_positions()
            fish_pos_tuple = tuple(sorted(fish_positions.items()))
            return (is_max_turn, p0, p1, my_hook, opp_hook, caught, fish_pos_tuple)
        
        TIME_LIMIT = 0.04 # 20 milliseconds
        start_time = time.time()

        def heuristic(node: Node):
            if node.depth == len(node.observations):
                p0, p1 = node.state.get_player_scores()
                return p0 - p1
            
            # 1) Score difference (our score - opponent's score)
            p0, p1 = node.state.get_player_scores()
            value = p0 - p1

            # Get position data
            hooks = node.state.get_hook_positions()
            my_hook = hooks[0]
            opp_hook = hooks[1]

            fish_positions = node.state.get_fish_positions()
            fish_scores = node.state.get_fish_scores()

            # 2) If anyone currently has a fish on the hook
            my_caught, opp_caught = node.state.get_caught()

            # Bonus if we have a good fish, penalty if it's a bad one
            if my_caught is not None:
                fs = fish_scores[my_caught]
                # strongly reward if we have already caught something (80%)
                value += fs * 0.8  

            # Penalize if the opponent has something on their hook (their catch is bad for us)
            if opp_caught is not None:
                fs = fish_scores[opp_caught]
                value -= fs * 0.8

            # 3) Proximity to fish (only those that are not yet caught)
            # The closer to a good fish -> the better
            # The closer to a bad fish -> the worse

            for fid, (fx, fy) in fish_positions.items():
                fs = fish_scores[fid]

                # Distance from our hook to the fish
                dx_my = my_hook[0] - fx
                dy_my = my_hook[1] - fy
                dist_my = abs(dx_my) + abs(dy_my)

                # Distance from opponent's hook to the fish
                dx_opp = opp_hook[0] - fx
                dy_opp = opp_hook[1] - fy
                dist_opp = abs(dx_opp) + abs(dy_opp)

                # Close = 1.0, Far = approaches 0.0
                closeness_my = 1.0 / (dist_my + 1.0)
                closeness_opp = 1.0 / (dist_opp + 1.0)

                value += fs * (closeness_my - closeness_opp)

            return value

        def alphabeta(state: Node, depth: int, alpha: float, beta: float, is_max_turn: bool):
                
            key = make_key(state, is_max_turn)
            
            #print(depth)

            if(time.time() - start_time) >= TIME_LIMIT:
                time_up = True
            else:
                time_up = False

            entry = tt.get(key)
            print(entry)
            if entry is not None:
                stored_depth, stored_val = entry
                if stored_depth >= depth:
                    return stored_val
            
            children = state.compute_and_get_children()

            # If the node has no children (empty list) or we have looked at specific depth
            if not children or depth == 0:
                val = heuristic(state)
                tt[key] = (depth, val)
                return val
            
            if is_max_turn:
                children.sort(key=heuristic, reverse=True)
            else:
                children.sort(key=heuristic, reverse=False)

            if is_max_turn:
                # our turn (MAX)
                v = float('-inf')
                for child in children:
                    v = max(v, alphabeta(child, depth - 1, alpha, beta, False))
                    alpha = max(alpha, v)
                    if beta <= alpha or time_up:
                        break # pruning
            
            else:
                # opponent's turn (MIN) 
                v = float('inf')
                for child in children:
                    v = min(v, alphabeta(child, depth - 1, alpha, beta, True))
                    beta = min(beta, v)
                    if beta <= alpha or time_up:
                        break # pruning
            
            tt[key] = (depth, v)
            return v
        
        # Start with the root node
        root_children = initial_tree_node.compute_and_get_children()
        # If there are no children
        if not root_children:
            return ACTION_TO_STR[0] # "stay"
        
        root_children.sort(key=heuristic, reverse=True)
        
        # We want to maximize
        best_value = float('-inf')
        best_moves = []
        alpha = float('-inf')
        beta = float('inf')

        while(max_depth <= 1):
            # At root it's our turn (MAX).
            for child in root_children:
                # Opponents turn, returns 5 values
                val = alphabeta(child, max_depth, alpha, beta, False) # Worse value on Kattis if >4 why??

                # Check the 5 value
                if val > best_value:
                    best_value = val
                    best_moves = [child.move]
                    # If some values are the same, add both to the list
                elif val == best_value:
                    best_moves.append(child.move)
                    
                alpha = max(alpha, best_value)

                max_depth += 1

        choosen_move = random.choice(best_moves) # This does so that we sometimes get different results on Kattis with the same depth
        return ACTION_TO_STR[choosen_move]

                
        
        
        

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        #random_move = random.randrange(5)
        #return ACTION_TO_STR[random_move]