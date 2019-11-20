# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class BaseCaptureAgent(CaptureAgent):

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
    CaptureAgent.registerInitialState(self, gameState)

    self.ourTeamAgents = gameState.getTeam()
    self.opponentAgents = gameState.getOpponents()


    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState)
  
    action = expectimaxGetAction(self, gameState)

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
    self.expectimaxAgents[0:1] = gameState.getTeam()
    # storing opponents as indices 2 and 3
    self.expectimaxAgents[2:3] = gameState.getOpponents()

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


      for action in pacmanLegalActions:
        ourSuccessors.append(gameState.generateSuccessor(self.expectimaxAgents[agentIndex], action))


      currentEvalScores = []
      for child in pacmanSuccessors:
        currentEvalScores.append(self.evaluationFunction(child))

      #only adds top 3 to be explored 
      sorted(currentEvalScores, reverse = True)
      top3 = currentEvalScores[1:3]

      #only recurses for the top 3 of 5 moves 
      for i in range(3):
        child = pacmanSuccessors[pacmanSuccessors.index(top3[i])]
        ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, depthCounter+1))
        

      return max(ourSuccessorsEvalScores)


    #When it's the other team's turn 
    else: 

      opponenttSuccessors = [] #list of GameStates
      opponentSuccessorsEvalScores = [] #list of GameStates returned scores

      opponentLegalActions = gameState.getLegalActions(self.expectimaxAgents[agentIndex])

      for action in opponentLegalActions:
        opponentSuccessors.append(gameState.generateSuccessor(self.expectimaxAgents[agentIndex], action))

      currentEvalScores = []
      for child in opponentSuccessors:
        currentEvalScores.append(self.evaluationFunction(child))

      #only adds top 3 to be explored 
      sorted(currentEvalScores, reverse = True)
      top3 = currentEvalScores[1:3]

      #only recurses for the top 3 of 5 moves 
      for i in range(3):
        child = opponentSuccessors[opponentSuccessors.index(top3[i])]
        opponentSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, depthCounter+1))

      averages = []
      for i in range(len(opponentSuccessorsEvalScores)):
        total = sum(opponentSuccessorsEvalScores)
        averages[i] = opponentSuccessorsEvalScores[i]/total 

      return sum(averages)
  
      

  def evaluationFunction(currentGameState): 

    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()

    #This doesn't make sense
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsules = currentGameState.getCapsules()


    foodList = currentFood.asList()
    numFood = len(foodList)
    foodDistances = []
    finalScore = 0.0
    foodScore = 0.0
    originalFoodScore = 0.0
    closestFoodScore = 0.0


    ## CREATE A FOOD SCORE
    for food in foodList:    
            foodDistances.append(manhattanDistance(currentPos, food))
   
    # if there's no food left in the game
    if(numFood == 0): 
      # then this state is really good
      return 1000000
     
    else:
      # reward states with less food
      originalFoodScore = 2.5 * (1.0/numFood)
      
      # and reward states that have food that are close by
      closestFoodDistance = min(foodDistances)
      
      # if there's food right next to pacman this is a good state
      if closestFoodDistance == 0:
        closestFoodScore = 200.0

      # otherwise make it so closer food gets a higher score
      else:  
        closestFoodScore = 2.80 * (1.0/closestFoodDistance) 

      # create a final food score
      foodScore = closestFoodScore + originalFoodScore


    ## CREATE A CAPSULE SCORE
    capsuleScore = 0.0
    distanceToCapsules = []
    minCapsuleDistance = None 

    for capsule in currentCapsules:
      distanceToCapsules.append(manhattanDistance(currentPos, capsule))

    if not len(distanceToCapsules) == 0:
      minCapsuleDistance = min(distanceToCapsules)
      # reward being close to ghosts and capsules
      if minCapsuleDistance == 0:
        capsuleScore = 500.0
      else:
        capsuleScore = 2.80 * (1.0/(minCapsuleDistance))#+closestGhostDistance))
    else:
      capsuleScore = 20.0 #20.0
    

    ## FIND DISTANCES TO GHOSTS
    #creates a list of distances to ghosts
    distanceToGhosts = []
    for ghost in currentGhostStates:
      distanceToGhosts.append(manhattanDistance(currentPos, ghost.getPosition()))

    # manhattan distance to the closest ghost
    closestGhostDistance = min(distanceToGhosts)

    
    ## CREATE A FINALSCORE TO RETURN
    # if the ghost is right next to pacman thats bad
    if closestGhostDistance == 0:
      finalScore -= 100.0

    # otherwise scale the distance to the ghost and add in the foodscore and capsulescore
    else:
      finalScore -= 2.0 * (1.0/closestGhostDistance)

      finalScore += foodScore + capsuleScore
    

    return finalScore + scoreEvaluationFunction(currentGameState) #+ numFood



  ####################
  #  Helper Methods  #
  ####################






