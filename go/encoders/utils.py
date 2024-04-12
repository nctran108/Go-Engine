from go.goboard import Move, GameState, Point

def is_ladder_capture(game_state, candidate, recursion_depth=30):
    return is_ladder(True, game_state, candidate, None, recursion_depth)

def is_ladder_escape(game_state, candidate, recursion_depth=30):
    return is_ladder(False, game_state, candidate, None, recursion_depth)

def is_ladder(try_capture, game_state: GameState, candidate: Point, ladder_stones=None, recursion_depth=30):
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
            candidates = detemine_escape_candidates(current_state, ladder_stone, capture_player)
            # try to escape
            attempted_escapes = [is_ladder(False, current_state, escape_candidate, [ladder_stone], recursion_depth - 1) for escape_candidate in candidates]
            if not any(attempted_escapes):
                return True # if at least one escape fails, we capture
        else:
            if current_state.board.get_go_string(ladder_stone).num_liberties >= 3:
                return True # successful escape
            if current_state.board.get_go_string(ladder_stone).num_liberties == 1:
                continue # failed escape, other might still do
            candidates = liberties(current_state, ladder_stone)
            # try to capture
            attempted_captures = [is_ladder(True, current_state, capture_candidate, [ladder_stone], recursion_depth -1) for capture_candidate in candidates]

            if any(attempted_captures):
                continue # failed escape, try others
            return True # candidate can't be caught in a ladder, escape
    return False # no captures / no escapes

def is_candidate(game_state: GameState, point: Point, escape_player):
    current_player = game_state.board.get(point)
    liberties = count_liberties(game_state, point)
    # if the stone has same color as escape player and has 2 liberties then posible labder candidate
    return game_state.board.get(point) == escape_player and count_liberties(game_state, point) == 2

def guess_ladder_stones(game_state: GameState, point: Point, escape_player) -> list[Point]:
    adjacent_strings = [game_state.board.get_go_string(nb) for nb in point.neighbors() if game_state.board.get_go_string(nb)]
    if adjacent_strings:
        neighbors = []
        for string in adjacent_strings:
            if string is None:
                continue
            stones = string.stones
            for stone in stones:
                if is_candidate(game_state, stone, escape_player):
                    neighbors.append(stone)
        return neighbors
    return []

def detemine_escape_candidates(game_state: GameState, point: Point, capture_player):
    escape_candidates = list(game_state.board.get_go_string(point).liberties)
    if game_state.board.get_go_string(point):
        for other_ladder_stone in game_state.board.get_go_string(point).stones: # get all stones in the ladder
            candidate = []
            for neighbor in other_ladder_stone.neighbors(): # get the capture stones
                candidate = get_escape_candidates(game_state, neighbor, capture_player) # add all liberties
            escape_candidates += candidate
    return escape_candidates

def get_escape_candidates(game_state: GameState, point: Point, capture_player) -> list[Point]:
    if game_state.board.is_on_grid(point) and game_state.board.get(point) == capture_player and game_state.board.get_go_string(point).num_liberties == 1:
        return liberties(game_state, point) # expand the list with all liberties candidate that help to escape
    return []


def count_liberties(game_state: GameState, point: Point):
    # By doing this rather than get string liberties to reduce the stone with 0 liberty in same string
    # just get the stone that actual has liberties
    # hence, increase the speed of recursive
    liberties = 0
    for neighbor in point.neighbors():
        if not game_state.board.is_on_grid(neighbor):
            continue
        neighbor_color = game_state.board.get(neighbor)
        if neighbor_color is None:
            liberties += 1
    return liberties

def liberties(game_state: GameState, point: Point) -> list[Point]:
    if game_state.board.get_go_string(point):
        return list(game_state.board.get_go_string(point).liberties)
    return []
