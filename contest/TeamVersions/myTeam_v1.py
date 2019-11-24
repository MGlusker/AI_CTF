# myTeam_v1.py
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
               first = 'BaseCaptureAgent', second = 'BaseCaptureAgent'):
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

    self.ourTeamAgents = self.getTeam(gameState)
    self.opponentAgents = self.getOpponents(gameState)


    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
  
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
        currentEvalScores.append(self.evaluationFunction(gameState))

      # only add best 3 states to fully evaluate
      sorted(currentEvalScores, reverse = True)
      topThree = currentEvalScores[0:3]

      # only fully explores top 3 (out of 5) moves
      for i in range(3):
        child = pacmanSuccessors[pacmanSuccessors.index(topThree[i])]
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
  
      

  def evaluationFunction(currentGameState): 

    

    # our agents are at self.expectimaxAgents[0,1] opponents at [2,3] 
    # i dont think this works 
    allAgentStates = [currentGameState.getAgentState(i) for i in range(4)]

    # a list to hold our current states
    ourCurrentStates = []
    # a list to hold the enemies' current states
    enemyCurrentStates = []
    # want to have the correct states assigned to the correct teams
    for i in range(4):
      if i in self.ourTeamAgents:
        ourCurrentStates.append(allAgentStates[i])
      else:
        enemyCurrentStates.append(allAgentStates[i])
    
    currentEnemyScaredTimes = [enemyState.scaredTimer for enemyState in enemyCurrentStates]

    
    finalScore = 0.0
    foodScore = self.getFoodScore(currentGameState)
    capsuleScore = self.getCapsuleScore(currentGameState)
    

    ## FIND DISTANCES TO ENEMIES
    #creates a list of distances to enemies
    # this is where we want to implement our noisy reading ************

    distanceToEnemies = []
    for enemy in enemyCurrentStates:
      # 0 or 1 is us 2 or 3 is enemy
      distanceToEnemies.append(self.getExpectedDistance(expectimaxAgents[0], expectimaxAgents[2]))
      #distanceToEnemies.append(manhattanDistance(currentPos, enemy.getPosition()))

    # manhattan distance to the closest enemy
    closestEnemyDistance = min(distanceToEnemies)

    
    ## CREATE A FINALSCORE TO RETURN
    # if the ghost is right next to pacman thats bad
    if closestEnemyDistance == 0:
      finalScore -= 100.0

    # otherwise scale the distance to the ghost and add in the foodscore and capsulescore
    else:
      finalScore -= 2.0 * (1.0/closestEnemyDistance)

      finalScore += foodScore + capsuleScore
    

    return finalScore + self.getScore(currentGameState)
