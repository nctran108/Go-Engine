from go.goboard import Move, GameState, Point

def is_ladder_capture(game_state, candidate, recursion_depth=50):
    return is_ladder(True, game_state, candidate, None, recursion_depth)

def is_ladder_escape(game_state, candidate, recursion_depth=50):
    return is_ladder(False, game_state, candidate, None, recursion_depth)

def is_ladder(try_capture, game_state: GameState, candidate: Point, ladder_stones=None, recursion_depth=50):
    """Ladders are played out in reversed roles, one player tries to capture,
    the other to escape. We determine the ladder status by recursively calling
    is_ladder in opposite roles, providing suitable capture or escape candidates.

    Arguments:
    try_capture: boolean flag to indicate if you want to capture or escape the ladder
    game_state: current game state, instance of GameState
    candidate: a move that potentially leads to escaping the ladder or capturing it, instance of Move
    ladder_stones: the stones to escape or capture, list of Point. Will be inferred if not provided.
    recursion_depth: when to stop recursively calling this function, integer valued.

    Returns True if game state is a ladder and try_capture is true (the ladder captures)
    or if game state is not a ladder and try_capture is false (you can successfully escape)
    and False otherwise.
    """

    move = Move(candidate)
    if not game_state.is_valid_move(move) or not recursion_depth:
        return False
    
    next_player = game_state.next_player
    capture_player = next_player if try_capture else next_player.other
    escape_player = capture_player.other

    if ladder_stones is None:
        ladder_stones = guess_ladder_stones(game_state, candidate, escape_player)
        
    for ladder_stone in ladder_stones:
        current_state = game_state.apply_move(move)

        if try_capture:
            candidates = detemine_escape_candidates(game_state, ladder_stone, capture_player)
            # try to escape
            attempted_escapes = [is_ladder(False, current_state, escape_candidate, ladder_stones, recursion_depth - 1) for escape_candidate in candidates]
            if not any(attempted_escapes):
                return True # if at least one escape fails, we capture
        else:
            if count_liberties(current_state, ladder_stone) >= 3:
                return True # successful escape
            if count_liberties(current_state, ladder_stone) == 1:
                continue # failed escape, other might still do
            candidates = liberties(current_state, ladder_stone)
            # try to capture
            attempted_captures = [is_ladder(True, current_state, capture_candidate, ladder_stones, recursion_depth -1) for capture_candidate in candidates]

            if any(attempted_captures):
                continue # failed escape, try others
            return True # candidate can't be caught in a ladder, escape
    return False # no captures / no escapes

def is_candidate(game_state: GameState, point: Point, player):
    # if the stone has same color as current player and has 2 liberties then posible labder candidate
    return game_state.next_player == player and count_liberties(game_state, point) == 2

def guess_ladder_stones(game_state: GameState, point: Point, escape_player):
    adjacent_strings = [game_state.board.get_go_string(nb) for nb in point.neighbors() if game_state.board.get_go_string(nb)]
    if adjacent_strings:
        neighbors = []
        for string in adjacent_strings:
            stones = string.stones
            for stone in stones:
                neighbors.append(stone)
        return [nb for nb in neighbors if is_candidate(game_state, nb, escape_player)]
    return []

def detemine_escape_candidates(game_state: GameState, point: Point, capture_player):
    escape_candidates = [nb for nb in point.neighbors() if game_state.board.is_on_grid(nb)]
    if game_state.board.get_go_string(point):
        for other_ladder_stone in game_state.board.get_go_string(point).stones:
            for neighbor in other_ladder_stone.neighbors():
                right_color = game_state.board.get(neighbor) == capture_player
                one_liberty = count_liberties(game_state, neighbor) == 1
                if right_color and one_liberty:
                    escape_candidates += liberties(game_state, neighbor)
    return escape_candidates

def count_liberties(game_state: GameState, point: Point):
    if game_state.board.get_go_string(point):
        return game_state.board.get_go_string(point).num_liberties
    return 0

def liberties(game_state: GameState, point: Point) -> list[Point]:
    if game_state.board.get_go_string(point):
        return list(game_state.board.get_go_string(point).liberties)
    return []
