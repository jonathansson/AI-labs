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
    
        #Timer for IDS
        TIME_LIMIT = 0.07 # 70 milliseconds
        start_time = time.time()
        time_up = False

        def heuristic(node: Node):
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
                dist_my = math.hypot(dx_my, dy_my)  # sqrt(dx^2 + dy^2)

                # Distance from opponent's hook to the fish
                dx_opp = opp_hook[0] - fx
                dy_opp = opp_hook[1] - fy
                dist_opp = math.hypot(dx_opp, dy_opp)

                # Close = 1.0, Far = approaches 0.0
                closeness_my = 1.0 / (dist_my + 1.0)
                closeness_opp = 1.0 / (dist_opp + 1.0)

                value += fs * (closeness_my - closeness_opp)

            return value

        def alphabeta(state: Node, depth: int, alpha: float, beta: float, is_max_turn: bool):
            nonlocal time_up

            if time.time() - start_time >= TIME_LIMIT:
                time_up = True
                return heuristic(state)
            else:
                time_up = False
            
            children = state.compute_and_get_children()

            # If the node has no children (empty list) or we have looked at specific depth
            if not children or depth == 0:
                return heuristic(state)

            if is_max_turn:
                # our turn (MAX)
                v = float('-inf')
                for child in children:
                    v = max(v, alphabeta(child, depth - 1, alpha, beta, False))
                    if time_up:
                        return v
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break # pruning
                return v
            else:
                # opponent's turn (MIN) 
                v = float('inf')
                for child in children:
                    v = min(v, alphabeta(child, depth - 1, alpha, beta, True))
                    if time_up:
                        return v
                    beta = min(beta, v)
                    if beta <= alpha:
                        break # pruning
                return v
        
        # Start with the root node
        root_children = initial_tree_node.compute_and_get_children()
        # If there are no children
        if not root_children:
            return ACTION_TO_STR[0] # "stay"
        
        # IDS
        best_move_IDS = None
        max_depth = 200

        for current_depth in range(1, max_depth + 1):
            # Check if time left
            if time.time() - start_time >= TIME_LIMIT:
                break

            time_up = False
            # We want to maximize
            best_value = float('-inf')
            best_moves = []
            alpha = float('-inf')
            beta = float('inf')

            # Search current depth
            for child in root_children:
                if time_up:
                    break
                
                # Opponents turn
                val = alphabeta(child, current_depth, alpha, beta, False)

                if val > best_value:
                    best_value = val
                    best_moves = [child.move]
                
                # If some values are the same, add both to the list
                elif val == best_value:
                    best_moves.append(child.move)

                alpha = max(alpha, best_value)

            # If completed on time, update best move
            if not time_up and best_moves:
                best_move_IDS = random.choice(best_moves)

            if time_up:
                break
        
        # Return best move found from deepest completed search
        if best_move_IDS is not None:
            return ACTION_TO_STR[best_move_IDS]
        else:
            return ACTION_TO_STR[0]

                
        
        
        

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        #random_move = random.randrange(5)
        #return ACTION_TO_STR[random_move]
