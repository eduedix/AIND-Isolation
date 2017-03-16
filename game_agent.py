"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    own_moves_length = len(own_moves)
    opponent_moves_length = len(opponent_moves)

    def eval_1():
        return float((own_moves_length - opponent_moves_length) * (own_moves_length + opponent_moves_length))

    def eval_2():
        return float(own_moves_length - 2 * opponent_moves_length)

    def eval_3():
        game_size = game.width * game.height
        return float(own_moves_length - opponent_moves_length - game.move_count)

    def eval_4():
        game_size = game.width * game.height

        if game.move_count < 0.1 * game_size:
            return float(own_moves_length - opponent_moves_length)
        elif game.move_count < 0.8 * game_size:
            return float(own_moves_length - opponent_moves_length - game.move_count)
        else:
            return float(own_moves_length - opponent_moves_length)

    def eval_5():
        import random
        possibility = random.random()

        if possibility < 0.4:
            return own_moves_length - opponent_moves_length
        else:
            return own_moves_length - opponent_moves_length - game.move_count

    def eval_6():
        game_size = game.width * game.height
        if game.move_count < 0.9 * game_size:
            return float(own_moves_length - opponent_moves_length)
        else:
            return float(own_moves_length - opponent_moves_length - game.move_count)

    def eval_7():
        game_size = game.width * game.height
        if game.move_count < 0.1 * game_size:
            return float(own_moves_length)
        elif game.move_count < 0.9 * game_size:
            return float(own_moves_length - opponent_moves_length + game.move_count)
        else:
            return float(own_moves_length - opponent_moves_length - game.move_count)

    def eval_8():
        game_size = game.width * game.height
        if game.move_count < 0.1 * game_size:
            return float(own_moves_length)
        else:
            return float(own_moves_length - opponent_moves_length + game.move_count)

    def eval_9():
        return float(own_moves_length - opponent_moves_length + game.move_count)

    def eval_10():
        game_size = game.width * game.height
        occupancy = game.move_count / game_size
        return occupancy * (own_moves_length - opponent_moves_length) + game.move_count

    def eval_11():
        game_size = game.width * game.height
        occupancy = game.move_count / game_size
        return occupancy * (own_moves_length - (2 * occupancy) * opponent_moves_length) + game.move_count


    if game.is_loser(player):
        return float('-inf')

    if game.is_winner(player):
        return float('inf')

    return eval_9()


class CustomPlayer(object):
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)

        score, move = float('-inf'), legal_moves[0]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == 'minimax':
                method = self.minimax
            elif self.method == 'alphabeta':
                method = self.alphabeta

            if self.iterative:
                depth = 1
                while score is not float('inf'):
                    score, move = max(
                        method(game, depth), (score, move))
                    depth += 1
            else:
                score, move = max(
                    method(game, self.search_depth), (score, move))

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return move

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        move = (-1, -1)

        if depth == 0:
            return self.score(game, self), move

        if maximizing_player:
            eval_func = max
            score = float('-inf')
        else:
            eval_func = min
            score = float('inf')

        for legal_move in game.get_legal_moves():
            current_score, _ = self.minimax(
                game.forecast_move(legal_move), depth - 1, not maximizing_player)
            score, move = eval_func((score, move), (current_score, legal_move))

        return score, move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        move = (-1, -1)

        if depth == 0:
            return self.score(game, self), move

        if maximizing_player:
            score = float('-inf')
            legal_moves = game.get_legal_moves()
        else:
            score = float('inf')
            legal_moves = game.get_legal_moves(game.get_opponent(self))

        for legal_move in legal_moves:
            current_score, _ = self.alphabeta(
                game.forecast_move(legal_move), depth - 1, alpha, beta, not maximizing_player)
            if maximizing_player:
                if current_score > score:
                    score, move = current_score, legal_move
                    if score > alpha:
                        alpha = score
                    if score >= beta:
                        return score, move
            else:
                if current_score < score:
                    score, move = current_score, legal_move
                    if score < beta:
                        beta = score
                score, move = min((score, move), (current_score, legal_move))
                if score <= alpha:
                    return score, move

        return score, move
