"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import operator


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def _avg(l):
    """Return average of the list

    Parameters
    ----------
    l : list
        list of numerical values

    Returns
    -------
    float
        Average of the list
    """
    return sum(l)/float(len(l))


def _evaluate_weighted_moves(game, weighted_moves, player_location):
    """
    Calculate score by aggregating weights given for open moves

     Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    weigted_moves : list
        List containing tuples with list of possible moves and weight for the moves

    player_location : tuple
        Location of the player being evaluated

    Returns
    -------
    float
       weighted score
    """

    r, c = player_location
    score = 0
    for w, moves in weighted_moves:
        score += sum([w for dr, dc in moves if game.move_is_legal((r - dr, c - dc))])

    return score


def _heuristic_inv_distance(game, player):
    """
    This heuristic measures the measures the distance to each free cell and
    use the inverse (calculated by subtracting player position from board size) as the score.
    Heuristic uses the sum of distance to neighbors as the final value.

    Note: Distance between two cells is maximum of row displacement or column displacement.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_move = game.get_player_location(player)
    opponent_move = game.get_player_location(game.get_opponent(player))
    if own_move == game.NOT_MOVED:
        return game.get_blank_spaces()

    def average_distance(_move):
        return _avg(
            [game.width - max(abs(_move[0] - r), abs(_move[1] - c))
                for r in range(game.height)
                for c in range(game.width)
                    if game.move_is_legal((r, c))])

    player_score = average_distance(own_move)
    opponent_score = average_distance(opponent_move)

    return float(player_score - opponent_score)


def _heuristic_free_neighbors(game, player):
    """
    This heuristic measures occupancy of the neighboring cells.
    For each free neighbor (within [-3, +3] range) score of 1 is added;
    except for the hard_to_reach cells.
    Score of -1 is given because these cells take 4 moves to reach from the current position.
    Difference between two players are used as heuristic value.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    move = game.get_player_location(player)
    opponent_move = game.get_player_location(game.get_opponent(player))
    if move == game.NOT_MOVED:
        return game.get_blank_spaces()

    hard_to_reach = [(2, 2), (-2, -2), (2, -2), (-2, 2)]
    neighbours = [(dr, dc) for dr in range(-3, +4) for dc in range(-3, +4)]

    weighted_moves = [
        (1, list(set(neighbours) - set(hard_to_reach) - {(0, 0)})),
        (-1, hard_to_reach)
    ]

    player_score = _evaluate_weighted_moves(game, weighted_moves, move)
    opponent_score = _evaluate_weighted_moves(game, weighted_moves, opponent_move)

    return float(player_score - opponent_score)


def _heuristic_improved_open_moves(game, player):
    """
    This improves open moves heuristic by giving a -0.5 weight for the hard to reach cells hard_to_reach.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_move = game.get_player_location(player)
    opponent_move = game.get_player_location(game.get_opponent(player))
    if own_move == game.NOT_MOVED:
        return game.get_blank_spaces()

    open_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    hard_to_reach = [(2, 2), (-2, -2), (2, -2), (-2, 2)]

    weighted_moves = [
        (1, open_moves),
        (-0.5, hard_to_reach)
    ]

    player_score = _evaluate_weighted_moves(game, weighted_moves, own_move)
    opponent_score = _evaluate_weighted_moves(game, weighted_moves, opponent_move)

    return float(player_score - opponent_score)


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

    return _heuristic_improved_open_moves(game, player)


class CustomPlayer:
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

    ILLEGAL_MOVE = -1, -1

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.SEARCH_FUNCTIONS = {
            'minimax': self.minimax,
            'alphabeta': self.alphabeta
        }

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
        next_move = self.ILLEGAL_MOVE

        if len(legal_moves) != 0:
            # get search function
            search_fn = self.SEARCH_FUNCTIONS[self.method]

            try:
                # The search method call (alpha beta or minimax) should happen in
                # here in order to avoid timeout. The try/except block will
                # automatically catch the exception raised by the search method
                # when the timer gets close to expiring
                if self.iterative:
                    depth = 1
                    while True:
                        _, next_move = search_fn(game, depth)
                        depth += 1
                else:
                    _, next_move = search_fn(game, self.search_depth)

            except Timeout:
                # Handle any actions required at timeout, if necessary
                pass

        # Return the best move from the last completed search iteration
        return next_move

    def utility_score(self, game, maximizing_player):
        """Find utility score for the game state

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current game state
        """
        if maximizing_player:
            player = game.active_player
        else:
            player = game.inactive_player

        return self.score(game, player)


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

        if depth == 0:
            best_move = self.utility_score(game, maximizing_player), self.ILLEGAL_MOVE
        else:
            comparator_fn, starting_value = self._generate_minmax_fn(maximizing_player)
            best_move = (starting_value, self.ILLEGAL_MOVE)

            for move in game.get_legal_moves(game.active_player):
                temp_board = game.forecast_move(move)

                _score, _ = self.minimax(temp_board, depth-1, not maximizing_player)
                if comparator_fn(_score, best_move[0]):
                    best_move = _score, move

        return best_move


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

        if depth == 0:
            best_move = self.utility_score(game, maximizing_player), self.ILLEGAL_MOVE
        else:
            comparator_fn, starting_value = self._generate_minmax_fn(maximizing_player)
            should_prune_fn, update_alphabeta_fn = self._generate_alphabeta_fn(maximizing_player)

            best_move = (starting_value, self.ILLEGAL_MOVE)

            for move in game.get_legal_moves(game.active_player):
                temp_board = game.forecast_move(move)

                _score, _ = self.alphabeta(temp_board, depth - 1, alpha, beta, not maximizing_player)
                if comparator_fn(_score, best_move[0]):
                    best_move = _score, move

                if should_prune_fn(_score, alpha, beta):
                    return best_move
                alpha, beta = update_alphabeta_fn(_score, alpha, beta)

        return best_move

    @staticmethod
    def _generate_minmax_fn(maximizing_player):
        """Generate comparator function and starting value depending on player type

        Parameters
        ----------
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        comparator
            comparator depending on min or max strategy
        starting_value
            starting value depending on min or max strategy (-inf for min, +inf for max)
        """
        if maximizing_player:
            return operator.gt, float("-inf")
        else:
            return operator.lt, float("inf")

    @staticmethod
    def _generate_alphabeta_fn(maximizing_player):
        """Return prune check function and alpha-beta update function depending on the player type

        Parameters
        ----------
        maximizing_player : bool
           Flag indicating whether the current search depth corresponds to a
           maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        should_prune_fn
           branch prune check function
        update_alphabeta_fn
           function that updates alpha beta values
        """
        if maximizing_player:
            def should_prune_fn(value, _, beta):
                return value >= beta

            def update_alphabeta_fn(value, alpha, beta):
                alpha = value if value > alpha else alpha
                return alpha, beta
        else:
            def should_prune_fn(value, alpha, _):
                return value <= alpha

            def update_alphabeta_fn(value, alpha, beta):
                beta = value if value < beta else beta
                return alpha, beta

        return should_prune_fn, update_alphabeta_fn
