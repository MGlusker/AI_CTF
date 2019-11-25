# team_particleFiltering_112319_v0.py
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



class JointParticleFilter:
  """
  JointParticleFilter tracks a joint distribution over tuples of all ghost
  positions.
  """

  def __init__(self, numParticles=100):
      self.setNumParticles(numParticles)

  def setNumParticles(self, numParticles):
      self.numParticles = numParticles

  def initialize(self, ourTeamAgents, opponentAgents, gameState, legalPositions):
      "Stores information about the game, then initiaself.numGhosts = gameState.getNumAgents() - 1"

      self.ourTeamAgents = ourTeamAgents
      self.opponentAgents = opponentAgents
      self.legalPositions = legalPositions
      self.initializeParticles(gameState)



  def initializeParticles(self, gameState):
      #This should only be called at the very beginning of the game because thats the only time
     
      #Initialize new array of particles
      self.particles = []
    
      #Iterate through the agents and find their starting postion
      startPos = []
      for opponentAgentIndex in self.opponentAgents:
        startPos.append(gameState.getInitialAgentPosition(opponentAgentIndex))

      #this turns the list of startPos into a tuple
      startPos = tuple(startPos)

      #Each particle is of the format: [enemy agent 0 location, enemy agent 1 location]
      #Note which agents these are depend will vary basedd on which team we are
      for i in range(self.numParticles):
          self.particles.append(startPos)

  def initializeParticlesUniformly(self, gameState):
    #reset particles to uniform
    import itertools

    self.particles = []

    # get all possible permutations of legal ghost positions
    product = itertools.product(self.legalPositions, repeat=2)

    possiblePositions = list(product)

    # randomize permutations
    random.shuffle(possiblePositions)

    
    parts = self.numParticles
    
    length = len(possiblePositions)

    for i in range(parts):
        self.particles.append(possiblePositions[i%length])

  def observeState(self, gameState, currentAgentIndex):
      """
      Reweights and then resamples the set of particles using the likelihood of the noisy
      observations.
      """
      import functools
      pos = []
      unknownParticleIndices = []

      #If we know where opponents are then all particles should agree with that evidence
      for index, opponentAgentIndex in enumerate(self.opponentAgents):

        #If we can see the position then we know then it returns the correct position
        #Else, it returns None
        pos.append(gameState.getAgentPosition(opponentAgentIndex))

        #If the pos is None, add to array to signal that we don't know where this agent is 
        if pos[index] == None:
          unknownParticleIndices.append(opponentAgentIndex)


      #This returns an array of noisy Distances from our current agent
      noisyDistances = gameState.getAgentDistances()      
      particleWeights = []

      #current agent's position
      #this is what we will be calculating the true distance based on
      myPos = gameState.getAgentPosition(currentAgentIndex) 

      #weighting the particles
      for index, p in enumerate(self.particles):

        listOfLocationWeights = []

        # loop through all agents that we can't see
        # unknownParticleIndices refers to the indices in the particle itself and not the agents more broadly
        # therefore the options the list will be [0], [1], or [0,1], see the code in the for loop above to remedy confusion
        for i, opponentAgentIndex in enumerate(unknownParticleIndices):

          # find the true distance from pacman to the current ghost that we're iterating through
          trueDistance = util.manhattanDistance(p[i], myPos)

          # weight each particle by the probability of getting to that position (use emission model)
          # account for our current belief distribution (evidence) at this point in time
          listOfLocationWeights.append(gameState.getDistanceProb(trueDistance, noisyDistances[opponentAgentIndex]))

        if len(listOfLocationWeights) != 0:
            particleWeights.append(functools.reduce(lambda x,y: x*y, listOfLocationWeights))
        else:

            #If there are no unknown agents, that means we exactly where both agents are
            #therefore particle weight is 1 if the particle agrees with both positions
            #or 0 if either of positions are wrong
            if p == pos:
              particleWeights.append(1)
            else:
              particleWeights.append(0)


      # now create a counter and count up the particle weights observed
      particleDictionary = util.Counter()
 
      for index, p in enumerate(self.particles):
          #particleDictionary[self.particles[i]] += particleWeights[i]
          particleDictionary[p] += particleWeights[index]


      particleDictionary.normalize() 
      

      if particleDictionary.totalCount() == 0:
        
        self.initializeParticlesUniformly(gameState)
        #I'm not sure it makes sense to reinitialize the particles 
        #It's necessary however otherwise the program crashes
        #Does it make sense for our distribution to go to zero somtimes?
        
      # otherwise, go ahead and resample based on our new beliefs 
      else:
          
        keys = []
        values = []

        # find each key, value pair in our counter
        keys, values = zip(*particleDictionary.items())

        self.particles = util.nSample(values, keys, self.numParticles)


  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the
    gameState.
    """
    newParticles = []

    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of ghost positions
      # now loop through and update each entry in newParticle...
      
      temp = []
      for opponentIndex in self.opponentAgents:

        ourPostDist = self.getOpponentDist(gameState, newParticle, opponentIndex)

        temp.append(util.sample(ourPostDist))
          
      newParticle = temp
      newParticles.append(tuple(newParticle))
    
    self.particles = newParticles


  def getBeliefDistribution(self):
      # convert list of particles into a counter
      beliefs = util.Counter() 

      # count each unique position 
      for p in self.particles:
          beliefs[p] += 1
      
      # normalize the count above
      beliefs.normalize()
      return beliefs

  #This methods isonly slightly different from the above get belief distribution method
  #Returns a slightly different formatted distribution for use in displaying the distribution on the map
  def modGetBeliefDistribution(self):
 
    # convert list of particles into a counter
    beliefsOpponentOne = util.Counter() 
    beliefsOpponentTwo = util.Counter() 

    # count each unique position 
    for p in self.particles:
        beliefsOpponentOne[p[0]] += 1
        beliefsOpponentTwo[p[1]] += 1
    
    # normalize the count above
    beliefsOpponentOne.normalize()
    beliefsOpponentTwo.normalize()

    return [beliefsOpponentOne, beliefsOpponentTwo]

  ##################################
  # HELPER METHODS FOR ELAPSE TIME #
  ##################################

  def getOpponentDist(self, gameState, particle, opponentIndex):

    #create a gamestate that corresponds to the particle 
    opponentPositionGameState = self.setOpponentPositions(gameState, particle)

    dist = util.Counter()

    opponentLegalActions = opponentPositionGameState.getLegalActions(opponentIndex)
    prob = float(1)/float(len(opponentLegalActions))
    

    for action in opponentLegalActions:
      successor = opponentPositionGameState.generateSuccessor(opponentIndex, action)
      pos = successor.getAgentState(opponentIndex).getPosition()
      dist[pos] = prob

    return dist

  def setOpponentPositions(self, gameState, opponentPositions):
    "Sets the position of all opponent to the values in the particle"

    for index, pos in enumerate(opponentPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[self.opponentAgents[index]] = game.AgentState(conf, False)
    return gameState

  ##########################
  # END PARTICLE FILTERING #
  ##########################


  #BEGIN REFLEX CODE

class BaseCaptureAgent(CaptureAgent):
  """
  A base class for our agents that chooses score-maximizing actions
  """

  #Particle Filtering Stuff
  #This is the global instance of our particle filtering code that lets us share particles
  #between our two agents
  jointInference = JointParticleFilter()
  
  
  #ef initialize(self, ourTeamAgents, opponentAgents, gameState, legalPositions):
  def registerInitialState(self, gameState):

    legalPositions = self.findLegalPositions(gameState)
    self.ourTeamAgents = self.getTeam(gameState)
    self.opponentAgents = self.getOpponents(gameState)

    BaseCaptureAgent.jointInference.initialize(self.ourTeamAgents, self.opponentAgents, gameState, legalPositions)

    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    ######################
    # PARTICLE FILTERING #
    ######################
    
    
    start = time.time()
  
    self.jointInference.observeState(gameState, self.index)

    displayDist = self.jointInference.modGetBeliefDistribution()
    dist = self.jointInference.getBeliefDistribution()
    self.displayDistributionsOverPositions(displayDist)

    self.jointInference.elapseTime(gameState)
    print "Particle Filtering time:", time.time() - start

    ##########################
    # END PARTICLE FILTERING #
    ##########################

  
    
    action = self.expectimaxGetAction(gameState, dist)
    print "Total time:", time.time() - start
    return action

  def expectimaxGetAction(self, gameState, dist):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
    """

    #Always our turn at beginning our expectimax
    mostLikelyState = dist.argMax()
    probableGameState = self.setOpponentPositions(gameState, mostLikelyState)

    # list to keep track of agents
    self.expectimaxAgents = []
    # storing our team as indices 0 and 1
    self.expectimaxAgents[0:2] = self.ourTeamAgents
    # storing opponents as indices 2 and 3
    self.expectimaxAgents[2:4] = self.opponentAgents

    ourSuccessors = [] #list of our successor GameStates
    ourSuccessorsEvalScores = [] #list of our GameStates' returned scores


    ourLegalActions = probableGameState .getLegalActions(self.expectimaxAgents[0])

    for action in ourLegalActions:
      ourSuccessors.append(probableGameState.generateSuccessor(self.expectimaxAgents[0], action))
    
    for child in ourSuccessors:
      ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, 1))

    print ourSuccessorsEvalScores
    return ourLegalActions[ourSuccessorsEvalScores.index(max(ourSuccessorsEvalScores))]
   


  def getActionRecursiveHelper(self, gameState, depthCounter):

    
    ## to do
    # particle filtering so we can get legal actions
    # make it so expectimax doesn't simply return average of all possible actions
    # need a better evaluation function that actually works 
    # alpha / beta?? 



    NUM_AGENTS = 4

    #In terms of moves, not plies
    DEPTH = 8
    
    agentIndex = depthCounter%NUM_AGENTS

    ############################################################
    # NEED TO CHANGE depth variable if we want variable depth  #
    ############################################################

    #print depthCounter
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

      currentEvalScores = util.Counter()
      for child in ourSuccessors:
        currentEvalScores[child] = (self.evaluationFunction(gameState))

      # only add best 3 states to fully evaluate
      topThree = currentEvalScores.sortedKeys()[0:2]

      # only fully explores top 3 (out of 5) moves
      for successor in topThree:
          ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, depthCounter+1))

      return max(ourSuccessorsEvalScores)


    #When it's the other team's turn 
    else: 

      opponentSuccessors = [] #list of GameStates
      opponentSuccessorsEvalScores = [] #list of GameStates returned scores

      opponentLegalActions = gameState.getLegalActions(self.expectimaxAgents[agentIndex])

      for action in opponentLegalActions:
        opponentSuccessors.append(gameState.generateSuccessor(self.expectimaxAgents[agentIndex], action))

      currentEvalScores = util.Counter()
      for child in opponentSuccessors:
        currentEvalScores[child] = (self.evaluationFunction(gameState))

      # only add best 3 states to fully evaluate
      topThree = currentEvalScores.sortedKeys()[0:2]

      # only fully explores top 3 (out of 5) moves
      for successor in topThree:
          opponentSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, depthCounter+1))
    
      # averages = []
      # total = sum(opponentSuccessorsEvalScores)
      
      # for i in range(len(opponentSuccessorsEvalScores)): 
      #   averages[i] = (opponentSuccessorsEvalScores[i]/total)*opponentSuccessorsEvalScores[i]
      
      return sum(opponentSuccessorsEvalScores)/len(opponentSuccessorsEvalScores)
  
      

  def evaluationFunction(self, currentGameState): 

    

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

    # distanceToEnemies = []
    # for enemy in enemyCurrentStates:
    #   # 0 or 1 is us 2 or 3 is enemy
      
    #   distanceToEnemies.append(self.getMazeDistance(currentGameState.getAgentPosition(self.index), enemy.getPosition()))

    # # manhattan distance to the closest enemy
    # closestEnemyDistance = min(distanceToEnemies)

    
    # ## CREATE A FINALSCORE TO RETURN
    # # if the ghost is right next to pacman thats bad
    # if closestEnemyDistance == 0:
    #   finalScore -= 100.0

    # # otherwise scale the distance to the ghost and add in the foodscore and capsulescore
    # else:
    #   finalScore -= 2.0 * (1.0/closestEnemyDistance)

    finalScore += foodScore + capsuleScore
    

    return finalScore

  def getFoodScore(self, gameState):
    #this is meant as an offensive food score

    #This is for food we are trying to eat
    if gameState.isOnRedTeam(self.index):
      foodList = gameState.getBlueFood().asList()
    else:
      foodList = gameState.getRedFood().asList()


    foodDistances = []
    for food in foodList:    
      foodDistances.append(self.getMazeDistance(gameState.getAgentPosition(self.index), food))

    #foodDistances = sorted(foodDistances)
    #print foodDistances
    # get the closest food and scale it up to make it more desireable
    closestFoodDistances = min(foodDistances)
    #print closestFoodDistance
    closestFoodScore = closestFoodDistances * -500.0

    return closestFoodScore


  def getCapsuleScore(self, gameState):
    #this is meant as an offensive capsule score

    #This is for capsules we are trying to eat
    if gameState.isOnRedTeam(self.index):
      capsuleList = gameState.getBlueCapsules()
    else:
      capsuleList = gameState.getRedCapsules()
    
    distanceToCapsules = []
    capsuleScore = 0.0
    minCapsuleDistance = None 
    
    
    for capsule in capsuleList:
      distanceToCapsules.append(self.getMazeDistance(gameState.getAgentPosition(self.index), capsule))

    if not len(distanceToCapsules) == 0:
      minCapsuleDistance = min(distanceToCapsules)
      # reward being close to ghosts and capsules
      if minCapsuleDistance == 0:
        capsuleScore = 500.0
      else:
        capsuleScore = 2.80 * (1.0/(minCapsuleDistance))#+closestGhostDistance))
    else:
      capsuleScore = 20.0 #20.0

    return capsuleScore
    

  #################
  # HELPER METHOD #
  #################
  #this is necessary for initializing particles uniformly
  def findLegalPositions(self, gameState):
    mapHeight = gameState.getWalls().height
    mapWidth = gameState.getWalls().width

    legalPositions = []
    for x in range(mapWidth):
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          legalPositions.append((x,y))

    return legalPositions

  def setOpponentPositions(self, gameState, opponentPositions):
    "Sets the position of all opponent to the values in the particle and return a gameState item"

    for index, pos in enumerate(opponentPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[self.opponentAgents[index]] = game.AgentState(conf, False)
    return gameState


  
          


