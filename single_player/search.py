"""
COMP30024 Artificial Intelligence, Semester 1 2019
Solution to Project Part A: Searching

Authors: Luke Di Giuseppe, Audrey Webb
"""

from collections import defaultdict, deque
from utils import (
    is_in, argmin, argmax, argmax_random_tie, probability, weighted_sampler,
    memoize, print_table, open_data, PriorityQueue, name,
    distance, vector_add
)
import math
import sys
import json
import copy
import time

BOARD_SIZE = 3

#Exit hexs for each colour
RED_EXITS = [(3,-3),(3,-2),(3,-1), (3, 0)]
BLUE_EXITS = [(0,-3), (-1,-2), (-2,-1),(-3,0)]
GREEN_EXITS = [(-3,3),(-2,3),(-1,3),(0,3)]
MOVE = 'MOVE'
EXIT = 'EXIT'
JUMP = 'JUMP'
TIMEOUT = 25


### -------------------------------------------------------------------------
#CLASSES

#####FROM AIMA TEXTBOOK:-------------------------------------------------
class Problem(object):

    """ The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return 1

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


## MODIFIED SUBCLASSES:--------------------------------------------------

class Single_chexers_problem(Problem):
    '''Single player chexers game with multiple pieces
    '''

    def __init__(self, initial, goal=None):
        Problem.__init__(self, initial, goal)

    def goal_test(self, state):

        #if all the pieces have exited, we have reached our goal
        if len(state.pieces) == 0:
            return True
        return False


    def actions(self, state):
        action_list = []

        #look at the possible moves for each piece
        for i in range(len(state.pieces)):
            q,r = state.pieces[i][0], state.pieces[i][1]

            #if the piece is on an exit square, we can exit
            for hex in self.goal:
                if (q,r) == hex:
                    action_list.append([EXIT, (q,r)])
            #otherwise, check adjacent hexs for possible moves
            for x in range(-1,2):
                for y in range(-1,2):
                    if x != y:
                        if valid_move((q,r),(q + x,r + y), state):
                            action_list.append([MOVE,[(q,r),(q + x, r + y)]])

                        ##if the piece can't move, can it jump?
                        elif valid_jump((q,r),(q + x,r + y),(q + 2*x,r + 2*y), state):
                            action_list.append([JUMP,[(q,r),(q + 2*x, r + 2*y)]])

        return action_list

    def result(self, state, action):
        new_pieces = copy.deepcopy(state.pieces)

        #remove the piece from the board if exiting
        if action[0] is EXIT:
            new_pieces.remove(action[1])
        #change the piece's location if moving
        else:
            new_pieces.remove(action[1][0])
            new_pieces.append(action[1][1])

        return Board_state(new_pieces, state.blocks)

    def h(self, node):
        '''our simple heuristic function, calculating the minimum manhattan
        distance of our pieces to the exit, then optimistically assuming the piece
        can jump all the way to the end, and is therefore admissable
        '''
        goals = self.goal
        pieces = node.state.pieces
        est_distance = 0
        for hex in pieces:
            #add 1 move for the exit step
            est_distance += (compute_man_dis(goals, hex)/2 + 1)

        return est_distance

class Board_state(object):
    '''board state for single player, multiple piece board ,
    takes into account piece and block locations
    '''
    def __init__(self,pieces,blocks):
        self.pieces = pieces
        self.blocks = blocks

    def __eq__(self, other):
        ''' states are equal if all pices are in the same location
        '''
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            if len(self.pieces) != len(other.pieces):
                return False
            for hex in self.pieces:
                if hex not in other.pieces:
                    return False
            return True

    def __str__(self):
        return ('Piece location: ' + str(self.pieces) + ' Blocks: ' + str(self.blocks))


def main():

    '''Main functions
    Works in two steps:
    Attempts to find the optimal solution to the problem in under 25 seconds
    using an astar search accross the whole board.

    If unsuccessful, the program uses a much faster(and suboptimal) method of
    computing the best path for each individual piece, then letting them exit
    in the correct order as to not impede each other's path
    '''
    #collect data
    with open(sys.argv[1]) as file:
        data = json.load(file)

    #make and print the board
    board_dict = make_board_dict(data)
    print_board(board_dict, message="", debug=True)

    #eliminate blocked exit hexs from our goal
    goal_hexs = set_goal_hexs(data)

    #record the initial state and set the problem
    initial_state = (Board_state(convert_to_tuples(data['pieces']),
                        convert_to_tuples(data['blocks'])))
    problem = Single_chexers_problem(initial_state, goal_hexs)

    # run the astar search
    result = astar_search(problem, problem.h)

    if result:
        #if the astar executed in the alloted time, print the result
        print_moves(result.solution())
    else:
        #the search was too slow, so set up the simpler search

        #arrays of each of the pieces movements and their starting hex
        all_pieces = []
        all_starts = []

        #run the pathfinding search for each individual piece
        for piece in data['pieces']:
            #add the starting hex
            all_starts.append(tuple(piece))

            #set up the initial state and the problems
            initial_state2 = Board_state([tuple(piece)], convert_to_tuples(data['blocks']))
            problem2 = Single_chexers_problem(initial_state2, goal_hexs)

            #add the path to the list
            result2 = astar_search(problem2)
            all_pieces.append(result2.solution())


        #print the movement paths in a valid order
        while(len(all_pieces) > 1):

            #choose a peice to exit and print its moves
            next = choose_next(all_pieces,all_starts)
            print_moves(all_pieces[next])

            #remove it from the list
            all_pieces.pop(next)
            all_starts.pop(next)

        #print the final piece
        print_moves(all_pieces[0])




###HELPER FUNCTIONS:

def print_moves(result):
    '''
    Arguments:
    'result': a list of actions

    Formats the action list for PRINTING'''
    for action in result:
        if action[0] == EXIT:
            print('EXIT from ' + str(action[1]) + '.')
        else:
            print(str(action[0]) + ' from ' + str(action[1][0]) + ' to ' + str(action[1][1]) + '.')

def choose_next(all_pieces,all_starts):
    '''
    Arguments:
    'all_pieces' - a list of lists of actions

    'all_starts' - a list of hexs


    Takes an array of movelists and starting locations, then selects
    a movelist that does not move through any other pieces starting locations
    and returns its index
    '''
    #iterate over all movelists and startlists
    num_left = len(all_pieces)
    for i in range(num_left):
        #tracks if a clash has been found
        found = False
        for y in range(num_left):
            #ignore the movelist's own startpoint
            if i != y:

                #EXIT moves have less arguments, so select the correct index
                for move in all_pieces[i]:
                    if move[0] is EXIT:
                        submove = move[1]
                    else:
                        submove = move[1][1]

                    #if we have a clash, record it
                    if submove == all_starts[y]:
                        found = True
        #if the piece is valid, return it's index
        if not found:
            return i

def convert_to_tuples(list_list):
    '''
    Arguments:

    'list_lists'

    converts a list of lists to a list of tuples
    '''

    new_list = []
    for list in list_list:
        new_list.append(tuple(list))
    return new_list

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

def valid_hex(hex, state):
    '''
    Arguments:

    'hex': a hex tuple

    'state': a board state

    returns if the given hex would be a valid place to
    put a piece, ie on the board and also not blocked by another piece
    '''
    range_start = -BOARD_SIZE
    range_fin = BOARD_SIZE
    if hex[0] < 1:
        range_start = range_start - hex[0]
    else:
        range_fin = range_fin - hex[0]

    return ((-BOARD_SIZE <= hex[0] <= BOARD_SIZE)
            and (range_start <= hex[1] <= range_fin)
            and (hex not in state.blocks)
            and (hex not in state.pieces))

def valid_move(current_hex, next_hex, state):
    '''
    returns if the given piece move is non-arbitrary
    '''
    return (valid_hex(next_hex, state) and (current_hex != next_hex))

def valid_jump(current_hex, jump_hex, land_hex, state):
    ''' returns if the given piece jump is valid
    '''
    return (valid_hex(land_hex, state) and (not valid_hex(jump_hex, state)))

def make_board_dict(data):
    '''make the board data readable by Matt's printing function'''
    dict = {}
    for block in data['blocks']:
        dict[tuple(block)] = '@@@@@'
    for piece in data ['pieces']:
        dict[tuple(piece)] = 'PIECE'
    return dict

def set_goal_hexs(data):
    '''removes blocked hexs from the list of exit hexs'''
    if data['colour'] == 'red':
        exits = RED_EXITS
    elif data['colour'] == 'blue':
        exits = BLUE_EXITS
    else:
        exits = GREEN_EXITS

    free_exits = []
    for hex in exits:
        if hex not in convert_to_tuples(data['blocks']):
            free_exits.append(hex)
    return free_exits


### SEARCH FUNCTIONS---------------------------------------------------------
'''All search functions are taken from the AIMA textbook,
our modifications are few, but have been noted
'''
def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))

def best_first_graph_search(problem, f):
    '''MODIFICATION: a timeout check has been added'''


    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    start = time.time()
    while frontier and (time.time() - start < TIMEOUT):
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(tuple(sorted(node.state.pieces)))
        for child in node.expand(problem):
            if tuple(sorted(child.state.pieces)) not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def print_board(board_dict, message="", debug=False):
    """
    Helper function to print a drawing of a hexagonal board's contents.

    Arguments:

    * `board_dict` -- dictionary with tuples for keys and anything printable
    for values. The tuple keys are interpreted as hexagonal coordinates (using
    the axial coordinate system outlined in the project specification) and the
    values are formatted as strings and placed in the drawing at the corres-
    ponding location (only the first 5 characters of each string are used, to
    keep the drawings small). Coordinates with missing values are left blank.

    Keyword arguments:

    * `message` -- an optional message to include on the first line of the
    drawing (above the board) -- default `""` (resulting in a blank message).
    * `debug` -- for a larger board drawing that includes the coordinates
    inside each hex, set this to `True` -- default `False`.
    * Or, any other keyword arguments! They will be forwarded to `print()`.
    """

    # Set up the board template:
    if not debug:
        # Use the normal board template (smaller, not showing coordinates)
        template = """# {0}
#           .-'-._.-'-._.-'-._.-'-.
#          |{16:}|{23:}|{29:}|{34:}|
#        .-'-._.-'-._.-'-._.-'-._.-'-.
#       |{10:}|{17:}|{24:}|{30:}|{35:}|
#     .-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
#    |{05:}|{11:}|{18:}|{25:}|{31:}|{36:}|
#  .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
# |{01:}|{06:}|{12:}|{19:}|{26:}|{32:}|{37:}|
# '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
#    |{02:}|{07:}|{13:}|{20:}|{27:}|{33:}|
#    '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
#       |{03:}|{08:}|{14:}|{21:}|{28:}|
#       '-._.-'-._.-'-._.-'-._.-'-._.-'
#          |{04:}|{09:}|{15:}|{22:}|
#          '-._.-'-._.-'-._.-'-._.-'"""
    else:
        # Use the debug board template (larger, showing coordinates)
        template = """# {0}
#              ,-' `-._,-' `-._,-' `-._,-' `-.
#             | {16:} | {23:} | {29:} | {34:} |
#             |  0,-3 |  1,-3 |  2,-3 |  3,-3 |
#          ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
#         | {10:} | {17:} | {24:} | {30:} | {35:} |
#         | -1,-2 |  0,-2 |  1,-2 |  2,-2 |  3,-2 |
#      ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
#     | {05:} | {11:} | {18:} | {25:} | {31:} | {36:} |
#     | -2,-1 | -1,-1 |  0,-1 |  1,-1 |  2,-1 |  3,-1 |
#  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
# | {01:} | {06:} | {12:} | {19:} | {26:} | {32:} | {37:} |
# | -3, 0 | -2, 0 | -1, 0 |  0, 0 |  1, 0 |  2, 0 |  3, 0 |
#  `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
#     | {02:} | {07:} | {13:} | {20:} | {27:} | {33:} |
#     | -3, 1 | -2, 1 | -1, 1 |  0, 1 |  1, 1 |  2, 1 |
#      `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
#         | {03:} | {08:} | {14:} | {21:} | {28:} |
#         | -3, 2 | -2, 2 | -1, 2 |  0, 2 |  1, 2 | key:
#          `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' ,-' `-.
#             | {04:} | {09:} | {15:} | {22:} |   | input |
#             | -3, 3 | -2, 3 | -1, 3 |  0, 3 |   |  q, r |
#              `-._,-' `-._,-' `-._,-' `-._,-'     `-._,-'"""

    # prepare the provided board contents as strings, formatted to size.
    ran = range(-3, +3+1)
    cells = []
    for qr in [(q,r) for q in ran for r in ran if -q-r in ran]:
        if qr in board_dict:
            cell = str(board_dict[qr]).center(5)
        else:
            cell = "     " # 5 spaces will fill a cell
        cells.append(cell)

    # fill in the template to create the board drawing, then print!
    board = template.format(message, *cells)
    print(board)

# when this module is executed, run the `main` function:
if __name__ == '__main__':
    main()
