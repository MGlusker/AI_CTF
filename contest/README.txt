CTF README
This project was built by Oliver Baldwin Edwards and Martin Glusker.
This is the final project for the Artificial Intelligence class (COSC 241) at Amherst College.

We use particle filtering to estimate the position of enemies, observing the enemies based on the noisy sensor reading, and then updating the particles by sampling from an even distribution of the possible moves an enemy can take from that position. We maintain an independent set of particles for each enemy. We have implemented a feature called "JailTimer" that finds the path from the jail position to the end of the jail Path. When we eat an opponent we assume that the enemy will get out of jail as quickly as possible, and we update the current position based the next position in the recorded jail path.

We select our move based on a minimax algorithm, implemented with alpha beta pruning, that calls an evaluation function. We maintain two sets of features: one for our defensive agent, and one for our offensive agent. 

Our evaluation function contains the following offensive features:

	-Food Score
	This incentivizes getting close to and eating capsules, and returning food to our side. The number of food eaten before returning to our side depends on whether or not an opponent has just been eaten. 
	-Enemy Closeness
	This incentivizes approaching enemies when we should (on our side or when the opponent is scared) and running away when we shouldn't (when we are on the opponent's side).
	-Capsule Score, 
	This incentivizes getting close to and eating capsules. 
	-Overall Score
	Just a measure of the overall score, and increasing our score is good. 

Our defensive features include:

	-Food Score
	This incentivizes preventing opponents from eating food. 
	-Enemy Closeness
	This incentivizes approaching enemies and eating opponents when they're on our side.
	-Capsule Score, 
	This incentivizes preventing opponents from eating capsules. 
	-Num Invaders Score
	This incentivizes preventing enemies from entering our side. 
	-Overall Score
	Just a measure of the overall score, and preventing our score from decreasing is good.
