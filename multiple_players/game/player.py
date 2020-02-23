"""
COMP30024 Artificial Intelligence, Semester 1 2019
Solution to Project Part B: Playing the Game

Authors: Luke Di Giuseppe, Audrey Webb
"""


"""Provides some utilities widely used by other modules"""
import random
import math as m
import numpy as np
from married_at_first_percept.evals import make_distance_board, give_defense


#actions
MOVE = 'MOVE'
EXIT = 'EXIT'
JUMP = 'JUMP'

#Starting hexes for each colour
_STARTING_HEXES = {
    'red': {(-3,3), (-3,2), (-3,1), (-3,0)},
    'green': {(0,-3), (1,-3), (2,-3), (3,-3)},
    'blue': {(3, 0), (2, 1), (1, 2), (0, 3)},
}
#Exit hexes for each colour
_FINISHING_HEXES = {
    'red': {(3,-3), (3,-2), (3,-1), (3,0)},
    'green': {(-3,3), (-2,3), (-1,3), (0,3)},
    'blue': {(-3,0),(-2,-1),(-1,-2),(0,-3)},
}
_ADJACENT_STEPS = [(-1,+0),(+0,-1),(+1,-1),(+1,+0),(+0,+1),(-1,+1)]

#hexes that make up the board configuration
_HEXES = {(q,r) for q in range(-3, +3+1) for r in range(-3, +3+1) if -q-r in range(-3, +3+1)}

#players with corresponding values
_PLAYERS = {'red' : 0, 'green' : 1, 'blue': 2}

_PNUM = 3

_MAX_DEPTH = 3

#dictionary of distance from exit hex for each player
_DISTANCE_HEXES = {
    'red': make_distance_board(_FINISHING_HEXES['red']),
    'green': make_distance_board(_FINISHING_HEXES['green']),
    'blue': make_distance_board(_FINISHING_HEXES['blue']),
}

### -------------------------------------------------------------------------
#CLASS AND SEARCH FUNCTION

#####FROM PROVIDED SKELETON CODE:-------------------------------------------------
'''All search functions are taken from the AIMA textbook or lecture slides,
our modifications are few, but have been noted
'''
class Player:

    def __init__(self, colour):
        '''Called once at the beginning of the game to initialise a player.
        Sets up internal representation of the game state, and other information about the
    game state we would like to maintain for the duration of the game. The parameter
    colour will be a string representing the player the program will play as
    (Red, Green or Blue). The value will be one of the strings "red", "green",
    or "blue" correspondingly.'''
        self.state = self.initialise_board()
        self.colour = colour
        self.number = _PLAYERS[colour]
        self.red_distances = make_distance_board(_FINISHING_HEXES['red'])
        self.green_distances = make_distance_board(_FINISHING_HEXES['green'])
        self.blue_distances = make_distance_board(_FINISHING_HEXES['blue'])


    def action_rand(self):
        '''Called to request a choice of random actions from the max^N
        program. Simply  failsafe in case Max-N fails for some reason, the
        program will still give a valid move '''

    #available actions for player to choose in current state
        action_list = available_actions(self.state)


        i = random.randint(0,len(action_list) -1)
        if action_list[0][0] == EXIT:
            return action_list[0]
        return action_list[i]

    def action(self):
        '''Called at the beginning of each turn to request a choice of action
        from the max^N program. Based on the current state of the game,the player should
        select and return an allowed action to play on their turn. If there are no allowed
        actions, the player will return a pass instead. The action (or pass) is represented
        as tuples of two components - a string representing the action type, and a value
        representing the action argument.'''

        utility, action = self.max_n(self.state, self.colour)
        if action:
            return action
        else:
            return self.action_rand()

    def update(self, colour, action):
        '''Called at the end of every turn (including current player’s turns) to inform
        the player about the most recent action. Maintains the internal representation of
        the game state and any other information about the game we are storing.'''

        self.state.update(colour,action)


    def initialise_board(self):
        '''Creates the starting board setup.'''

        board = {qr: ' ' for qr in _HEXES}
        for colour in ("red", "green","blue"):
            for qr in _STARTING_HEXES[colour]:
                board[qr] = colour

        score = {'red': 0, 'green': 0, 'blue': 0}
        turn = 'red'

        return Board_state(board, score, turn, 0)

    def max_n(self, state, player):
        """max^N is an N-player adversarial search algorithm (in our case N = 3).
        Similar to minimax algorithm, but layers of the tree alternate between all n players; for each
        cut-off state compute utility values (or evaluation values) for all players and store
        in an n-dimensional utility vector; and backup values to non-cut-off states by choosing
        the vector with the highest value in the next player’s dimension. """
        if state.cutoff_test():
            end = self.utility(state)
            ##print(end)
            return (end, None)

        v_max = np.full(3, float(-10000))
        best_a = None

        for action in available_actions(state):
            (v, irrelevant) = self.max_n(state.result(player, action), next_turn(player))
            if v[_PLAYERS[player]] > v_max[_PLAYERS[player]]:
                v_max = v
                best_a = action

        ##if best_a is not None and best_a[0] == EXIT:
            ##print(v_max)
            #print(best_a)
        return (v_max, best_a)

    def utility(self, state):
        '''Gives numeric outcome for the game. Also, describes the cutoff-test and
        evaluation function, which cuts the search off at some point and applies a
        heuristic evaluation function that estimates the utility of a state.'''

    #cut-off test
        if state.cutoff_test():
            utility_vector = np.full(3, float(-10000))
            for player,score in state.score.items():
                if score > 3:
                    utility_vector[_PLAYERS[player]] = 10000
                    return utility_vector

        num_pieces = np.full(3, float(0))
        total_score = np.full(3, float(0))
        total_defense = np.full(3, float(0))
        total_distance = np.full(3, float(0))

        for qr, piece in state.board.items():
            if piece != ' ':
                index = _PLAYERS[piece]
                num_pieces[index] += 1
                total_defense[index] += give_defense(qr)
                total_distance[index] -= m.pow(_DISTANCE_HEXES[piece][qr],2)


        for player,score in state.score.items():
            total_score[_PLAYERS[player]] = score


        average_value = (total_defense + total_distance)/num_pieces

    #evaluation function
        utility_vector = average_value + total_score*1000 + num_pieces*100
        return utility_vector


## MODIFIED SUBCLASS:--------------------------------------------------
class Board_state(object):
    '''board state for three player, multiple piece board ,
    takes into account all piece locations.'''
    def __init__(self, board, score, turn, depth):
        self.board = board.copy()
        self.score= score.copy()
        self.turn = turn
        self.depth = depth

    def update(self, colour, action):
        type = action[0]
        coordinates = action[1]
        if type == MOVE:
            qr_a, qr_b = coordinates
            self.board[qr_a] = ' '
            self.board[qr_b] = colour
        elif type == JUMP:
            qr_a, qr_b = (q_a, r_a), (q_b, r_b) = coordinates
            qr_c = (q_a+q_b)//2, (r_a+r_b)//2
            self.board[qr_a] = ' '
            self.board[qr_b] = colour
            self.board[qr_c] = colour
        elif type == EXIT:
            qr = coordinates
            self.board[qr] = ' '
            self.score[colour] += 1
        else: # atype == "PASS":
            pass
        self.turn = next_turn(colour)

    def result(self, colour, action):
        new_state = Board_state(self.board,self.score,self.turn, self.depth + 1)
        new_state.update(colour, action)
        return new_state



    #cut-off test function
    def cutoff_test(self):
        for score in self.score.values():
            if self.depth > _MAX_DEPTH:
                return True
            if score > 3:
                return True
        return False



###HELPER FUNCTIONS:
def blank_utility():
    utility_vector = []
    for i in range(0,_PNUM):
        utility_vector.append(-m.inf)
    return utility_vector

def next_turn(player):
    if player == 'red':
        return 'green'
    elif player == 'green':
        return 'blue'
    return 'red'



#available actions
def available_actions(state):
    ''' limits actions:
    1. if exit is an available move at state, then exit
    2. if there are no possible actions then you can move to hex behind current hex,
    otherwise keep moving forwards. '''
    backward_actions = []
    available_actions = []
    for qr in _HEXES:
        if state.board[qr] == state.turn:

            #just exit!! (simple, fairly effective)
            if qr in _FINISHING_HEXES[state.turn]:
                available_actions.append((EXIT, qr))
                return [(EXIT, qr)]
            q, r = qr
            for dq, dr in _ADJACENT_STEPS:
                for i, atype in [(1, MOVE), (2, JUMP)]:
                    tqr = q+dq*i, r+dr*i
                    if tqr in _HEXES:
                        if state.board[tqr] == ' ':
                            #movement is forward, or captures a piece
                            if (_DISTANCE_HEXES[state.turn][qr] > _DISTANCE_HEXES[state.turn][tqr]
                            or (atype == JUMP and not state.board[tqr] == state.turn)):
                                available_actions.append((atype, (qr, tqr)))
                                break

                            #action is backward, and doesnt capture
                            else:
                                backward_actions.append((atype, (qr, tqr)))
                                break

    #if there are no available forward/captures, go backward
    if not available_actions:
        available_actions = backward_actions
        if not available_actions:
            available_actions.append(("PASS", None))

    #shuffle prevents strange patterns of movement/loops when two states
    #evaluate evenly
    random.shuffle(available_actions)
    return available_actions
