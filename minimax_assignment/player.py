#!/usr/bin/env python3
import random
import time
#import sys  #for debug prints

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

    def search_best_next_move(self, initial_tree_node: Node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # Transposition table
        tt = {}

        # Time limit per move
        TIME_LIMIT = 0.060
        start_time = time.time()
        time_up = False

        nodes_visited = 0  # used for debug

        def make_key(node: Node, is_max_turn: bool):
            s = node.state
            p0, p1 = s.get_player_scores()
            hooks = s.get_hook_positions()
            caught = s.get_caught()
            fish_positions = s.get_fish_positions()

            fish_pos_tuple = tuple(fish_positions.items())

            score_diff = p0 - p1

            return (is_max_turn, score_diff, hooks[0], hooks[1], caught, fish_pos_tuple)

        def heuristic(node: Node):
            s = node.state

            # If we reached the end of observations, just return score diff
            if node.depth == len(node.observations):
                p0, p1 = s.get_player_scores()
                return p0 - p1

            # 1) Score difference (our score - opponent's score)
            p0, p1 = s.get_player_scores()
            value = p0 - p1

            # Get position data
            hooks = s.get_hook_positions()
            my_hook = hooks[0]
            opp_hook = hooks[1]

            fish_positions = s.get_fish_positions()
            fish_scores = s.get_fish_scores()

            # 2) If anyone currently has a fish on the hook
            my_caught, opp_caught = s.get_caught()

            # Bonus if we have a good fish, penalty if it's a bad one
            if my_caught is not None:
                fs = fish_scores[my_caught]
                value += fs * 0.95

            # Penalize if the opponent has something on their hook
            if opp_caught is not None:
                fs = fish_scores[opp_caught]
                value -= fs * 0.95

            # 3) Proximity to fish (only those that are not yet caught)
            for fid, (fx, fy) in fish_positions.items():
                fs = fish_scores[fid]

                # Distance from our hook to the fish (Manhattan)
                dx_my = my_hook[0] - fx
                dy_my = my_hook[1] - fy
                dist_my = abs(dx_my) + abs(dy_my)

                # Distance from opponent's hook to the fish
                dx_opp = opp_hook[0] - fx
                dy_opp = opp_hook[1] - fy
                dist_opp = abs(dx_opp) + abs(dy_opp)

                closeness_my = 1.0 / (dist_my + 1.0)
                closeness_opp = 1.0 / (dist_opp + 1.0)

                value += fs * (closeness_my - closeness_opp)

            return value

        def alphabeta(state: Node, depth: int, alpha: float, beta: float, is_max_turn: bool):
            nonlocal time_up, nodes_visited

            # debug-print for each node
            nodes_visited += 1 # used for debug
            # print(
            #     f"[AB] node={nodes_visited}, tree_depth={state.depth}, remaining_depth={depth}, "
            #     f"{'MAX' if is_max_turn else 'MIN'}",
            #     file=sys.stderr,
            # )

            # Time cutoff
            if time_up or (time.time() - start_time) >= TIME_LIMIT:
                time_up = True
                return heuristic(state)

            key = make_key(state, is_max_turn)
            entry = tt.get(key)
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

            # Move ordering by heuristic only
            if is_max_turn:
                children.sort(key=heuristic, reverse=True)
            else:
                children.sort(key=heuristic, reverse=False)

            if is_max_turn:
                # our turn (MAX)
                v = float('-inf')
                for child in children:
                    if time_up or (time.time() - start_time) >= TIME_LIMIT:
                        time_up = True
                        break

                    val = alphabeta(child, depth - 1, alpha, beta, False)
                    if val > v:
                        v = val
                    if v > alpha:
                        alpha = v
                    if beta <= alpha:
                        break

            else:
                # opponent's turn (MIN)
                v = float('inf')
                for child in children:
                    if time_up or (time.time() - start_time) >= TIME_LIMIT:
                        time_up = True
                        break

                    val = alphabeta(child, depth - 1, alpha, beta, True)
                    if val < v:
                        v = val
                    if v < beta:
                        beta = v
                    if beta <= alpha:
                        break

            tt[key] = (depth, v)
            return v

        # Start with the root node
        root_children = initial_tree_node.compute_and_get_children()
        # If there are no children
        if not root_children:
            return ACTION_TO_STR[0]  # "stay"

        # Move ordering at root
        root_children.sort(key=heuristic, reverse=True)

        # Iterative deepening
        best_move_overall = root_children[0].move
        best_value_overall = float('-inf')
        best_heur_overall = heuristic(root_children[0])

        # max search depth
        max_depth = 8

        for current_depth in range(1, max_depth + 1):
            if (time.time() - start_time) >= TIME_LIMIT:
                break

            # print(f"\n=== IDS depth {current_depth} ===", file=sys.stderr)

            time_up = False
            best_value = float('-inf')
            best_heur = float('-inf')
            best_moves = []
            alpha = float('-inf')
            beta = float('inf')

            for child in root_children:
                if time_up or (time.time() - start_time) >= TIME_LIMIT:
                    time_up = True
                    break

                val = alphabeta(child, current_depth, alpha, beta, False)
                if time_up:
                    break

                h = heuristic(child)

                # First by minimax value, then by heuristic
                if val > best_value or (val == best_value and h > best_heur):
                    best_value = val
                    best_heur = h
                    best_moves = [child.move]
                elif val == best_value and h == best_heur:
                    best_moves.append(child.move)

                alpha = max(alpha, best_value)

            # Only accept this depth if we finished it in time
            if not time_up and best_moves:
                best_value_overall = best_value  # For debug
                best_heur_overall = best_heur # For debug
                # Allow random among truly equal moves
                best_move_overall = random.choice(best_moves)
                # Reorder root children so best move is first next iteration
                root_children.sort(key=lambda c: c.move != best_move_overall)

                # print(
                #     f"Depth {current_depth} complete. Best value={best_value_overall}, "
                #     f"best_heur={best_heur_overall}, best_move={best_move_overall}",
                #     file=sys.stderr,
                # )
            # else:
            #     print(f"Stopped early at depth {current_depth} (time_up={time_up})", file=sys.stderr)

            if time_up:
                break

        # print(
        #     f"Chosen move={best_move_overall}, total_nodes_visited={nodes_visited}",
        #     file=sys.stderr,
        # )

        choosen_move = best_move_overall
        return ACTION_TO_STR[choosen_move]


        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        #random_move = random.randrange(5)
        #return ACTION_TO_STR[random_move]