# team_evalFunc_112619_v1.py
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
               first = 'OffensiveCaptureAgent', second = 'DefensiveCaptureAgent'):
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



#############################
#  PARTICLE FILTERING CLASS #
#############################

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
    
    if opponentIndex in self.opponentAgents:
      for action in opponentLegalActions:
        successor = opponentPositionGameState.generateSuccessor(opponentIndex, action)
        pos = successor.getAgentState(opponentIndex).getPosition()
        dist[pos] = prob
    else:
      print "This was the opponentIndex:", opponentIndex
      print "I think something went wrong"

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





################################
# Baseline Capture Agent Class #
################################
  
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

    CaptureAgent.registerInitialState(self, gameState)
    # if we're on blue team this is always [1, 3]
    # if we're on red team this is always [0, 2]
    self.ourTeamAgents = self.getTeam(gameState)

    # this is opposite of above
    self.opponentAgents = self.getOpponents(gameState)
    self.mySideList = self.getMySide(gameState)


    BaseCaptureAgent.jointInference.initialize(self.ourTeamAgents, self.opponentAgents, gameState, legalPositions)

    self.start = gameState.getAgentPosition(self.index)
   


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
    #print "Particle Filtering time:", time.time() - start

    ##########################
    # END PARTICLE FILTERING #
    ##########################

  
    print "It's agent #", self.index, "'s turn"
    action = self.expectimaxGetAction(gameState, dist, self.index)
    
    print "Total time:", time.time() - start
    return action

  def expectimaxGetAction(self, gameState, dist, currentAgentIndex):
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


    ourLegalActions = probableGameState.getLegalActions(currentAgentIndex)

    for action in ourLegalActions:
      ourSuccessors.append(probableGameState.generateSuccessor(currentAgentIndex, action))
    
    print "currentAgentIndex", currentAgentIndex
    for child in ourSuccessors:
      ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(child, 1, currentAgentIndex+1))

    print ourSuccessorsEvalScores
    print "action chosen: ", ourLegalActions[ourSuccessorsEvalScores.index(max(ourSuccessorsEvalScores))]
    return ourLegalActions[ourSuccessorsEvalScores.index(max(ourSuccessorsEvalScores))]
   


  def getActionRecursiveHelper(self, gameState, depthCounter, currentAgentIndex):


    NUM_AGENTS = 4

    #In terms of moves, not plies
    DEPTH = 4
    
    if currentAgentIndex>=4:
      #print "index reset"
      currentAgentIndex = 0

    #print currentAgentIndex
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
    if(currentAgentIndex == self.ourTeamAgents[0] or currentAgentIndex == self.ourTeamAgents[1]): 

      ourSuccessors = [] #list of GameStates
      ourSuccessorsEvalScores = [] #list of GameStates' returned scores

      ourLegalActions = gameState.getLegalActions(currentAgentIndex)

      for action in ourLegalActions:
        ourSuccessors.append(gameState.generateSuccessor(currentAgentIndex, action))

      currentEvalScores = util.Counter()
      for child in ourSuccessors:
        currentEvalScores[child] = (self.evaluationFunction(gameState))

      # only add best 3 states to fully evaluate
      #topThree = currentEvalScores.sortedKeys()[0:3]

      # only fully explores top 3 (out of 5) moves
      # for successor in topThree:
      #     ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(successor, depthCounter+1, currentAgentIndex +1))

      for successor in ourSuccessors:
          ourSuccessorsEvalScores.append(self.getActionRecursiveHelper(successor, depthCounter+1, currentAgentIndex +1))

      return max(ourSuccessorsEvalScores)


    #When it's the other team's turn 
    else: 

      opponentSuccessors = [] #list of GameStates
      opponentSuccessorsEvalScores = [] #list of GameStates returned scores

      opponentLegalActions = gameState.getLegalActions(currentAgentIndex)

      for action in opponentLegalActions:
        opponentSuccessors.append(gameState.generateSuccessor(currentAgentIndex, action))

      currentEvalScores = util.Counter()
      for child in opponentSuccessors:
        currentEvalScores[child] = (self.evaluationFunction(gameState))

      # only add best 3 states to fully evaluate
      #topThree = currentEvalScores.sortedKeys()[1:5]

      # only fully explores top 3 (out of 5) moves
      for successor in opponentSuccessors:
          opponentSuccessorsEvalScores.append(self.getActionRecursiveHelper(successor, depthCounter+1, currentAgentIndex +1))
    
      # averages = []
      # total = sum(opponentSuccessorsEvalScores)
      
      # for i in range(len(opponentSuccessorsEvalScores)): 
      #   averages[i] = (opponentSuccessorsEvalScores[i]/total)*opponentSuccessorsEvalScores[i]
      
      return sum(opponentSuccessorsEvalScores)/len(opponentSuccessorsEvalScores)
  
      

  def evaluationFunction(self, currentGameState): 
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(currentGameState)
    weights = self.getWeights(currentGameState)
    return features * weights
 


  ##################
  # HELPER METHODS #
  ##################

  def findLegalPositions(self, gameState):
    #this is necessary for initializing particles uniformly
    mapHeight = gameState.getWalls().height
    mapWidth = gameState.getWalls().width

    legalPositions = []
    for x in range(mapWidth):
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          legalPositions.append((x,y))

    return legalPositions

  def setOpponentPositions(self, gameState, opponentPositions):
    """
    Sets the position of all opponent to the values in the particle 
    and returns a gameState item
    """

    for index, pos in enumerate(opponentPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[self.opponentAgents[index]] = game.AgentState(conf, False)
    return gameState

  def getMySide(self, gameState):
    """
    this returns a list with all of the accesible poisitions 
    on our side of the map (checks to see if we're red or blue team)
    """

    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height
    mySideList = []

    #red is always on the left; blue always on right 
    # if we're on the RED team 
    #print gameState.isOnRedTeam(self.index)
    if gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)-1
      mySideX = x
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    # if we're on the BLUE team
    #print not gameState.isOnRedTeam(self.index)
    if not gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)
      mySideX = x
      #print "BLUE"
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    return mySideList



#######################
##  Offensive Agent  ##
#######################

class OffensiveCaptureAgent(BaseCaptureAgent):


  def getFeatures(self, currentGameState):

    features = util.Counter()

    #ourPositions = self.getPositions(currentGameState, True)
    #enemyPositions = self.getPositions(currentGameState, False)
    
    
    
    #currentEnemyScaredTimes = [enemyState.scaredTimer for enemyState in enemyCurrentStates]

    #print currentEnemyScaredTimes

    foodScore = self.getFoodScore(currentGameState)
    capsuleScore = self.getCapsuleScore(currentGameState)
    enemyClosenessScore = self.getEnemyClosenessScore(currentGameState)


    features["foodScore"] = foodScore
    features["capsuleScore"] = capsuleScore
    features["enemyClosenessScore"] = enemyClosenessScore
    features["scoreOfGame"] = self.getScore(currentGameState)

    """
    print "FOOD: ", 100*foodScore
    print "CAPSULE: ", 10*capsuleScore
    print "ENEMY: ", 1*enemyClosenessScore
    print "ACTUAL SCORE: ", 1*self.getScore(currentGameState)
    """

    return features

  
  def getWeights(self, gameState):
    # foodScore is positive
    # capsuleScore is positive
    # enemyClosenessScore is negative
    # socreOfGame is negative if losing, positive if winning
    Weights = {"foodScore": 10, "capsuleScore": 10, "enemyClosenessScore": 10, "scoreOfGame": 1000}

      
    return Weights

  ###################################
  ## methods to get feature scores ##
  ###################################
  
  def getFoodScore(self, gameState):
    """
    Returns a score based on how much food is on the board 
    and based on how close we are to food
    (less food is better and being closer to food is better)
    """

    foodScore = 0.0

    # list of food we're trying to eat
    foodList = self.getFood(gameState).asList()


    # a list of all accesible poisitions on our side of the map
    mySideList = self.getMySide(gameState)

    foodDistances = []

    # add the distance from every food pellet to our agent
    for food in foodList:        
      foodDistances.append(self.getMazeDistance(gameState.getAgentPosition(self.index), food))

    
    
    # if no food left in the game this is good 
    if len(foodList) == 0:
      foodScore = 1000000

    # otherwise there's still food left
    else:
      
      # find the closest distance to a pellet of food
      closestFoodDistance = min(foodDistances)


      # first check to see if our agent is carrying 3 food (or more) 
      # and there's no other food close by, then incentivize going home (to our side)
      if gameState.getAgentState(self.index).numCarrying > 3 and closestFoodDistance > 3:
        # find the shortest distance back to our side
        minDistanceHome = min([self.getMazeDistance(gameState.getAgentPosition(self.index), position) for position in mySideList])

        # make it better to be closer to home
        foodScore = 100.0 * (1.0/minDistanceHome)
        
     
      # otherwise, we want to eat more food so reward states that are close to food
      else:  

        # reward states with less food 
        foodLeftScore = 100.0 * (1.0/len(foodList))

        # reward states that have food that is close by:
        # if food is right next to us this is really good
        if closestFoodDistance == 0:
          closestFoodScore = 400.0

        # otherwise make it so the closer the food, the higher the score
        else: 
          closestFoodScore = 100.0 * (1.0/closestFoodDistance)

        # create a final food score
        foodScore = closestFoodScore + foodLeftScore 

    return foodScore


  def getCapsuleScore(self, gameState):
    #this is meant as an offensive capsule score

    #This is for capsules we are trying to eat

    capsuleScore = 0.0

    capsuleList = self.getCapsules(gameState)

    distanceToCapsules = []
   
    #minCapsuleDistance = None 
    
    
    for capsule in capsuleList:
      distanceToCapsules.append(self.getMazeDistance(gameState.getAgentPosition(self.index), capsule))

    # if no capsules left in game this is good
    if len(distanceToCapsules) == 0:
      capsuleScore = 50.0

    # otherwise reward states with fewer capsules 
    else: 
      minCapsuleDistance = min(distanceToCapsules)
      
      # reward being close to capsules
      if minCapsuleDistance == 0:
        capsuleScore = 500.0
      
      else:
        capsuleScore = 100.0 * (1.0/(minCapsuleDistance)) #+closestGhostDistance))
    

    return capsuleScore
    
  def getEnemyClosenessScore(self, gameState): 
    """
    punish our agent being close to enemies 
    (unless we're on our own side)
    """

    # a boolean telling us if we're on our own side or not
    onMySide = self.areWeOnOurSide(gameState)

    # a list of the enemy positions (as determined by particleFiltering)
    enemyPositions = self.getPositions(gameState, False)
    
    distanceToEnemies = []

    enemyClosenessScore = 0.0

    # find distance to each enemy
    for enemy in enemyPositions:
      distanceToEnemies.append(self.getMazeDistance(gameState.getAgentPosition(self.index), enemy))

    closestEnemyDistance = min(distanceToEnemies)

    
    # if we're on our side it's good to be close to enemies
    if onMySide:
      if closestEnemyDistance == 0.0:
        enemyClosenessScore = 1000.0

      else:
        enemyClosenessScore = 100.0 * (1.0/closestEnemyDistance)

    # otherwise it's not good to be close to enemies
    else:
      if closestEnemyDistance == 0.0:
        enemyClosenessScore = -1000.0

      else:
        enemyClosenessScore = -100.0 * (1.0/closestEnemyDistance)


    return enemyClosenessScore


  #################################
  ## helper methods for features ##
  #################################


  def areWeOnOurSide(self, gameState):
    """
    this returns true if our agent is on our side
    and false if our agent is on the enemy's side
    """ 
    myPos = gameState.getAgentPosition(self.index)
    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height

    # if we're on the red team
    if gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)-1
      mySideX = x
   
    # if we're on the blue team 
    if not gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)
      mySideX = x

    onMySide = True 
    if myPos[0] >= mySideX and gameState.isOnRedTeam(self.index):
      onMySide = False
    if myPos[0] <= mySideX and not gameState.isOnRedTeam(self.index):
      onMySide = False

    return onMySide

  def getPositions(self, currentGameState, findOurs):
    """
    This takes a gameState where we know everyones position
    and returns either enemy or our positions in a list

    If findOurs is true then this returns our positions,
    if false then returns enemy positions
    """
    allPositions = [currentGameState.getAgentPosition(i) for i in range(4)]
    

    ourPositions = []
    enemyPositions = []
    
    # want to have the correct positions assigned to the correct teams
    for i in range(4):
      if i in self.ourTeamAgents:
        ourPositions.append(allPositions[i])
      else:
        enemyPositions.append(allPositions[i])

    if findOurs:
      return ourPositions
    else:
      return enemyPositions


#######################
##  Defensive Agent  ##
#######################

class DefensiveCaptureAgent(BaseCaptureAgent):

  def getFeatures(self, currentGameState):

    features = util.Counter()

    #ourPositions = self.getPositions(currentGameState, True)
    #enemyPositions = self.getPositions(currentGameState, False)
    
    
    
    #currentEnemyScaredTimes = [enemyState.scaredTimer for enemyState in enemyCurrentStates]

    #print currentEnemyScaredTimes

    foodScore = self.getFoodScore(currentGameState)
    capsuleScore = self.getCapsuleScore(currentGameState)
    numInvadersScore = self.getNumInvadersScore(currentGameState)
    enemyClosenessScore = self.getEnemyClosenessScore(currentGameState)


    features["foodScore"] = foodScore
    features["capsuleScore"] = capsuleScore
    features["numInvadersScore"] = numInvadersScore
    features["enemyClosenessScore"] = enemyClosenessScore
    features["scoreOfGame"] = self.getScore(currentGameState)

    """
    print "FOOD: ", foodScore
    print "CAPSULE: ", capsuleScore
    print "NUM INVADERS: ", numInvadersScore
    print "ENEMY: ", enemyClosenessScore
    print "ACTUAL SCORE: ", self.getScore(currentGameState)
    """
    return features

  
  def getWeights(self, gameState):
    # foodScore is positive
    # capsuleScore is positive
    # numInvadersScore is negative if we have invaders, positive if we don't
    # enemyClosenessScore is negative
    # socreOfGame is negative if losing, positive if winning
    Weights = {"foodScore": 10000, "capsuleScore": 10, "numInvadersScore": 100, "enemyClosenessScore": 100000, "scoreOfGame": 1}

      
    return Weights

  #######################################
  ## helper methods for feature scores ##
  #######################################
   
  def getFoodScore(self, gameState):
    """
    Returns a score based on how much food is on the board 
    and based on how close enemies are to food
    (more food is better and being closer to food is worse)
    """

    foodScore = 0.0

    # list of food we're defending
    foodList = self.getFoodYouAreDefending(gameState).asList()

    enemyPositions = self.getPositions(gameState, False)

    closestEnemyDistance = min(enemyPositions)

    # a list of all accesible poisitions on our side of the map
    mySideList = self.getMySide(gameState)

    foodDistances = []

    # it's bad if enemies are close to our food 

    # add the distance from every food pellet to each enemy
    for food in foodList:     
      for enemy in enemyPositions:   
        foodDistances.append(self.getMazeDistance(enemy, food))


    # if no food left in the game this is bad 
    if len(foodList) == 0:
      foodScore = -1000000

    # otherwise there's still food left
    else:
      
      # find the closest distance of an enemy to a pellet of food
      closestFoodDistance = min(foodDistances)
              
      # punish states with less food 
      foodLeftScore = -100.0 * (1.0/len(foodList))

      # punish states that have food close to an enemy:
      # if food is right next to enemy this is really bad
      if closestFoodDistance == 0:
        closestFoodScore = -200.0

      # otherwise make it so the closer the food, the lower the score
      else: 
        closestFoodScore = -100.0 * (1.0/closestFoodDistance)

      # create a final food score
      foodScore = closestFoodScore + foodLeftScore 

    return foodScore


  def getCapsuleScore(self, gameState):
    #this is meant as an defensive capsule score

    #This is for capsules we are trying to eat

    capsuleScore = 0.0

    capsuleList = self.getCapsulesYouAreDefending(gameState)
    enemyPositions = self.getPositions(gameState, False)

    distanceToCapsules = []
   
    #minCapsuleDistance = None 
    
    # it's bad if enemies are close to our pellets 
    # add the distance from every food capsule to each enemy
    
    for capsule in capsuleList:
      for enemy in enemyPositions:
        distanceToCapsules.append(self.getMazeDistance(enemy, capsule))

    # if no capsules left in game this is bad
    if len(distanceToCapsules) == 0:
      capsuleScore = -50.0

    # otherwise reward states with more capsules 
    else: 
      minCapsuleDistance = min(distanceToCapsules)
      
      # punish enemy being very close to capsule
      if minCapsuleDistance == 0:
        capsuleScore = -500.0
      
      # punish enemies being close to capsules
      else:
        capsuleScore = -100.0 * (1.0/minCapsuleDistance) #+closestGhostDistance))
    

    return capsuleScore
    
  
  def getNumInvadersScore(self, gameState):
    """
    counts how many invaders are on our side and returns
    a lower score for more invaders
    """

    enemyPositions = self.getPositions(gameState, False)

    # count how many invaders are on our side
    numInvaders = 0

    # a list of all accesible poisitions on our side of the map
    mySideList = self.getMySide(gameState)

    for enemy in enemyPositions:
      for position in mySideList:
        if enemy == position:
          numInvaders += 1
          print numInvaders, "numInvaders"


    return numInvaders * -500 

     


  def getEnemyClosenessScore(self, gameState): 
    """
    reward our agent being close to invaders 
    (unless we're on their side)
    """

    # a boolean telling us if we're on our own side or not
    onMySide = self.areWeOnOurSide(gameState)

    # a list of the enemy positions (as determined by particleFiltering)
    enemyPositions = self.getPositions(gameState, False)
    
    distanceToEnemies = []

    enemyClosenessScore = 0.0

    # find distance to each invader
    for enemy in enemyPositions:
      distanceToEnemies.append(self.getMazeDistance(gameState.getAgentPosition(self.index), enemy))

    closestEnemyDistance = min(distanceToEnemies)

    
    # if we're on our side it's good to be close to enemies
    if onMySide:
      if closestEnemyDistance == 0.0:
        enemyClosenessScore = 200.0

      else:
        enemyClosenessScore = 100.0 * (1.0/closestEnemyDistance)

    # otherwise it's not good to be close to enemies
    else:
      if closestEnemyDistance == 0.0:
        enemyClosenessScore = -200.0

      else:
        enemyClosenessScore = -100.0 * (1.0/closestEnemyDistance)

    
    return enemyClosenessScore

  #################################
  ## helper methods for features ##
  #################################

  def getMySide(self, gameState):
    """
    this returns a list with all of the accesible poisitions 
    on our side of the map (checks to see if we're red or blue team)
    """

    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height
    mySideList = []

    #red is always on the left; blue always on right 
    # if we're on the RED team 
    #print gameState.isOnRedTeam(self.index)
    if gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)-1
      mySideX = x
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    # if we're on the BLUE team
    #print not gameState.isOnRedTeam(self.index)
    if not gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)
      mySideX = x
      #print "BLUE"
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    return mySideList

  def areWeOnOurSide(self, gameState):
    """
    this returns true if our agent is on our side
    and false if our agent is on the enemy's side
    """ 
    myPos = gameState.getAgentPosition(self.index)
    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height

    # if we're on the red team
    if gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)-1
      mySideX = x
   
    # if we're on the blue team 
    if not gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)
      mySideX = x

    onMySide = True 
    if myPos[0] >= mySideX and gameState.isOnRedTeam(self.index):
      onMySide = False
    if myPos[0] <= mySideX and not gameState.isOnRedTeam(self.index):
      onMySide = False

    return onMySide

  def getPositions(self, currentGameState, findOurs):
    """
    This takes a gameState where we know everyones position
    and returns either enemy or our positions in a list

    If findOurs is true then this returns our positions,
    if false then returns enemy positions
    """
    allPositions = [currentGameState.getAgentPosition(i) for i in range(4)]
    

    ourPositions = []
    enemyPositions = []
    
    # want to have the correct positions assigned to the correct teams
    for i in range(4):
      if i in self.ourTeamAgents:
        ourPositions.append(allPositions[i])
      else:
        enemyPositions.append(allPositions[i])

    if findOurs:
      return ourPositions
    else:
      return enemyPositions
  
          


