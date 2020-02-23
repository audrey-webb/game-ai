# README

### Objective 
Play the game of Chexers, a three-player game (Red, Green, Blue) with four pieces per player. The goal for each player is to exit all of its pieces from the board using legal actions when it is its turn. Legal actions include adjacent moves, passing, jumping over its own pieces or over another player's piece (if piece is directly adjacent to it). If your player jumps over another player's piece, it essentially 'captures' that other piece and you can replace it with a piece from your player.  

#### Single Player 
Use the A* Search algorithm to play a single-player variant of the game, Chexers, in various situations involving either one player or multiple players of the same color. 

#### Multiple Players 
Use the maxN game tree search algorithm (similarily N-Person Minimax Algorithm) to play the original Chexers game. Also advanced the algorithm's evaluation function by considering distance from the goal, capturing another player, tracking piece location on the board, and compiling a "book" of opening and end games. Finally, implemented performance boosting. This involved performing maxN on the best possible moves which is defined through limiting the types of actions being taken. 


We give license for our code to be shared.
