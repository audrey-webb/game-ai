
#hexes that make up the board configuration
_HEX_GRID = {(q,r) for q in range(-3, +3+1) for r in range(-3, +3+1) if -q-r in range(-3, +3+1)}

#values for corner, edge, and middle positions as we highly value safety / defensive players
CORNER_DEFENSE = 1.5
EDGE_DEFENSE = 1
MIDDLE_DEFENSE = 0.5


something= {(-2, 0), (-1, 0), (3, 0), (2, 1),
(0, -2), (-3, 3), (3, -3), (0, 3),
(1, -2), (1, -1), (1, 2), (-2, 1),
(-1, 1), (2, -3), (-3, 2), (-1, -2),
(-1, -1), (-2, -1), (1, -3), (1, 1),
(0, 0), (-2, 2), (-1, 2), (2, -2),
(2, -1), (-3, 1), (1, 0), (0, -3),
(0, 1), (-2, 3), (-1, 3), (-3, 0),
(3, -2), (3, -1), (2, 0), (0, -1), (0, 2)}

#defensive value for each hex on board (37 total hexes)
_DEFENSIVE_VALUES = {(-2, 0): MIDDLE_DEFENSE , (-1, 0): MIDDLE_DEFENSE, (3, 0): 3, (2, 1): EDGE_DEFENSE,
(0, -2): MIDDLE_DEFENSE, (-3, 3): CORNER_DEFENSE, (3, -3) : CORNER_DEFENSE, (0, 3) : CORNER_DEFENSE,
(1, -2) : MIDDLE_DEFENSE, (1, -1): MIDDLE_DEFENSE, (1, 2): EDGE_DEFENSE, (-2, 1): MIDDLE_DEFENSE,
(-1, 1): MIDDLE_DEFENSE, (2, -3): EDGE_DEFENSE, (-3, 2): EDGE_DEFENSE, (-1, -2): EDGE_DEFENSE,
(-1, -1): MIDDLE_DEFENSE , (-2, -1): EDGE_DEFENSE, (1, -3): EDGE_DEFENSE, (1, 1):MIDDLE_DEFENSE ,
(0, 0): MIDDLE_DEFENSE, (-2, 2): MIDDLE_DEFENSE, (-1, 2): MIDDLE_DEFENSE, (2, -2): MIDDLE_DEFENSE,
(2, -1): MIDDLE_DEFENSE, (-3, 1): EDGE_DEFENSE , (1, 0): MIDDLE_DEFENSE, (0, -3): CORNER_DEFENSE,
(0, 1): MIDDLE_DEFENSE, (-2, 3):EDGE_DEFENSE , (-1, 3): EDGE_DEFENSE, (-3, 0): CORNER_DEFENSE,
(3, -2):EDGE_DEFENSE , (3, -1): EDGE_DEFENSE, (2, 0): MIDDLE_DEFENSE, (0, -1): MIDDLE_DEFENSE, (0, 2) :MIDDLE_DEFENSE}



###HELPER FUNCTIONS:
def give_defense(hex):
    return _DEFENSIVE_VALUES[hex]

def make_distance_board(exits):
    '''#distance calculation from exit hex for each player'''
    board = dict()
    for hex in _HEX_GRID:
        board[hex] = compute_man_dis(exits, hex) + 1
    return board


def axial_to_cube(hex):
    '''
    Arguments:
    'hex' a hex tuple

    coverts cubic hexagonal coordinates to axial coordinates'''
    cube = [hex[0], hex[1], (-hex[0]-hex[1])]
    return cube

def compute_man_dis(goal_hexs, hex):
    '''
    Arguments:

    'goal_hexs': a list of hexs

    'hex': a hex tuple

    calculates the hexagonal manhattan distance from each hex to the
    nearest goal hex'''
    c_hex = axial_to_cube(hex)
    fastest = 100000
    for goal in goal_hexs:
        c_goal = axial_to_cube(goal)
        dis = max(abs(c_hex[0]-c_goal[0]),abs(c_hex[1]-c_goal[1]),abs(c_hex[2]-c_goal[2]))
        if dis < fastest:
            fastest = dis
    return fastest
