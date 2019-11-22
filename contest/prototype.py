from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveBaseAgent', second = 'DefensiveBaseAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class BaseCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start = gameState.getAgentPosition(self.index)
    self.ourTeamAgents = self.getTeam(gameState)
    self.opponentAgents = self.getOpponents(gameState)

    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    expectimax action
    """
    action = self.expectimaxGetAction(gameState)
    return action

  ###################
  # Expectimax Code #
  ###################

  def expectimaxGetAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing -uniformly at random from their
      legal moves.
    """

    #Always our turn at beginning our expectimax

    # list to keep track of agents
    self.expectimaxAgents = []
    # storing our team as indices 0 and 1
    self.expectimaxAgents[0:2] = self.ourTeamAgents
    # storing opponents as indices 2 and 3
    self.expectimaxAgents[2:4] = self.opponentAgents

    ourSuccessors = [] #list of our successor GameStates
    ourSuccessorsEvalScores = [] #list of our GameStates' returned scores


    ourLegalActions = gameState.getLegalActions(self.expectimaxAgents[0])

    for action in ourLegalActions:
      ourSuccessors.append(gameState.generateSuccessor(self.expectimaxAgents[0], action))
    
    for child in ourSuccessors:
      ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, 1))

    return pacmanLegalActions[ourSuccessorsEvalScores.index(max(pacmanSuccessorsEvalScores))]
   

  def getActionRecursiveHelper(self, gameState, depthCounter):
    
    ## to do
    # particle filtering so we can get legal actions
    # make it so expectimax doesn't simply return average of all possible actions
    # need a better evaluation function that actually works 
    # alpha / beta?? 

    NUM_AGENTS = 4

    #In terms of moves, not plies
    DEPTH = 20 
    
    agentIndex = depthCounter%NUM_AGENTS

    ############################################################
    # NEED TO CHANGE depth variable if we want variable depth  #
    ############################################################


    #When we hit our depth limit AKA cutoff test
    if(DEPTH == depthCounter):
      return self.evaluationFunction(gameState)

    #implement a terminal test
    if(gameState.isOver()):
      return self.evaluationFunction(gameState)

    # When it's our turn
    # This may need to be changed based on whose turn it
    if(agentIndex == 0 or agentIndex == 1): 

      ourSuccessors = [] #list of GameStates
      ourSuccessorsEvalScores = [] #list of GameStates' returned scores

      ourLegalActions = gameState.getLegalActions(self.expectimaxAgents[agentIndex])

      for action in ourLegalActions:
        ourSuccessors.append(gameState.generateSuccessor(self.expectimaxAgents[agentIndex], action))

      currentEvalScores = []
      for child in ourSuccessors:
        currentEvalScores.append(self.evaluationFunction(child))

      # only add best 3 states to fully evaluate
      sorted(currentEvalScores, reverse = True)
      topThree = currentEvalScores[0:3]

      # only fully explores top 3 (out of 5) moves
      for i in range(3):
        child = ourSuccessors[ourSuccessors.index(topThree[i])]
        ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, depthCounter+1))

      return max(ourSuccessorsEvalScores)


    #When it's the other team's turn 
    else: 

      opponentSuccessors = [] #list of GameStates
      opponentSuccessorsEvalScores = [] #list of GameStates returned scores

      opponentLegalActions = gameState.getLegalActions(self.expectimaxAgents[agentIndex])

      for action in opponentLegalActions:
        opponentSuccessors.append(gameState.generateSuccessor(self.expectimaxAgents[agentIndex], action))

      currentEvalScores = []
      for child in opponentSuccessors:
        currentEvalScores.append(self.evaluationFunction(child))

      #only adds top 3 to be explored
      sorted(currentEvalScores, reverse = True)
      topThree = currentEvalScores[0:3]

      for i in range(3):
        child = opponentSuccessors[opponentSuccessors.index(topThree[i])]
        opponentSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, depthCounter+1))
    
      averages = []
      total = sum(opponentSuccessorsEvalScores)
      
      for i in range(len(opponentSuccessorsEvalScores)): 
        averages[i] = (opponentSuccessorsEvalScores[i]/total)*opponentSuccessorsEvalScores[i]
      
      return sum(averages)
  
 


  ##################
  # HELPER METHODS #
  ##################


  def getSuccessor(self, gameState):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluationFunction(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    return features * weights
  
  

class OffensiveBaseAgent(BaseCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState):
    features = util.Counter()
    successor = gameState
    foodList = self.getFood(gameState).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)


    features["foodScore"] = self.getFoodScore()
    features["capsuleScore"] = self.getCapsuleScore()
    
    #foodToEat = self.getFood(gameState)   
    #foodList = foodToEat.asList()
    #features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    """
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    """
    return features

  def getWeights(self, gameState):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveBaseAgent(BaseCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState):
    features = util.Counter()
    successor = self.getSuccessor(gameState)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
