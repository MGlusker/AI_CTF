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
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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

  def __init__(self, CaptureAgent, numParticles=100):
      self.captureAgent = CaptureAgent
      self.setNumParticles(numParticles)

  def setNumParticles(self, numParticles):
      self.numParticles = numParticles

  def initialize(self, gameState, legalPositions):
      "Stores information about the game, then initiaself.numGhosts = gameState.getNumAgents() - 1"

      self.ourTeamAgents = self.captureAgent.getTeam(gameState)
      self.opponentAgents = self.captureAgent.getOpponents(gameState)
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
      #!!!!not sure we need this!!!!
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
      Resamples the set of particles using the likelihood of the noisy
      observations.
      To loop over the ghosts, use:
        for i in range(self.numGhosts):
          ...
      A correct implementation will handle two special cases:
        1) When a ghost is captured by Pacman, all particles should be updated
           so that the ghost appears in its prison cell, position
           self.getJailPosition(i) where `i` is the index of the ghost.
           As before, you can check if a ghost has been captured by Pacman by
           checking if it has a noisyDistance of None.
        2) When all particles receive 0 weight, they should be recreated from
           the prior distribution by calling initializeParticles. After all
           particles are generated randomly, any ghosts that are eaten (have
           noisyDistance of None) must be changed to the jail Position. This
           will involve changing each particle if a ghost has been eaten.
      self.getParticleWithGhostInJail is a helper method to edit a specific
      particle. Since we store particles as tuples, they must be converted to
      a list, edited, and then converted back to a tuple. This is a common
      operation when placing a ghost in jail.
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
        #print index, pos
        if pos[index] == None:
          unknownParticleIndices.append(opponentAgentIndex)

        #print index, pos, unknownParticleIndices

      #This returns an array of noisy Distances from our current agent
      noisyDistances = gameState.getAgentDistances()      
      particleWeights = []

      #current agent's position
      #this is what we will be calculating the true distance based on
      myPos = gameState.getAgentPosition(currentAgentIndex) 



      # print "noisyDistances:", noisyDistances
      # print "unknownParticleIndices length", len(unknownParticleIndices)
      # print "enumerate", enumerate(unknownParticleIndices)
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
          #print noisyDistances[i]
          listOfLocationWeights.append(gameState.getDistanceProb(trueDistance, noisyDistances[opponentAgentIndex]))





        #print "locationWeights Length:", len(listOfLocationWeights)
        #print listOfLocationWeights
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






      #print particleWeights
      #print "number of particles", len(self.particles)
      # now create a counter and count up the particle weights observed
      particleDictionary = util.Counter()

      #for i in range(self.numParticles): 
      for index, p in enumerate(self.particles):
          #particleDictionary[self.particles[i]] += particleWeights[i]
          particleDictionary[p] += particleWeights[index]

      # print "particleDictionary", particleDictionary
      #print "number of particles", len(self.particles)

      particleDictionary.normalize() 
      
      if particleDictionary.totalCount() == 0:
        print "initializeParticlesUniformly"
        self.initializeParticlesUniformly(gameState)
        #I'm not sure it makes sense to reinitialize the particles here
        

      # otherwise, go ahead and resample based on our new beliefs 
      else:
          
        keys = []
        values = []

        # find each key, value pair in our counter
        keys, values = zip(*particleDictionary.items())
        # print "Particle dictionary: ", particleDictionary
        # print "length keys: ", len(keys)
        # print keys
        # print "length values: ", len(values)
        # print values
        # resample self.particles
        self.particles = util.nSample(values, keys, self.numParticles)



  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the
    gameState.
    To loop over the ghosts, use:
      for i in range(self.numGhosts):
        ...
    Then, assuming that `i` refers to the index of the ghost, to obtain the
    distributions over new positions for that single ghost, given the list
    (prevGhostPositions) of previous positions of ALL of the ghosts, use
    this line of code:
      newPosDist = getPositionDistributionForGhost(
         setGhostPositions(gameState, prevGhostPositions), i, self.ghostAgents[i]
      )
    Note that you may need to replace `prevGhostPositions` with the correct
    name of the variable that you have used to refer to the list of the
    previous positions of all of the ghosts, and you may need to replace `i`
    with the variable you have used to refer to the index of the ghost for
    which you are computing the new position distribution.
    As an implementation detail (with which you need not concern yourself),
    the line of code above for obtaining newPosDist makes use of two helper
    functions defined below in this file:
      1) setGhostPositions(gameState, ghostPositions)
          This method alters the gameState by placing the ghosts in the
          supplied positions.
      2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
          This method uses the supplied ghost agent to determine what
          positions a ghost (ghostIndex) controlled by a particular agent
          (ghostAgent) will move to in the supplied gameState.  All ghosts
          must first be placed in the gameState using setGhostPositions
          above.
          The ghost agent you are meant to supply is
          self.ghostAgents[ghostIndex-1], but in this project all ghost
          agents are always the same.
    """

    newParticles = []

    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of ghost positions
      # now loop through and update each entry in newParticle...
      
      temp = []
      for opponentIndex in self.opponentAgents:

        #This line from P4
        #newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, newParticle), i, self.ghostAgents[i])



        # loop through all agents that we can't see
        # unknownAgentIndices refers to the indices of rhe agents more broadly
        # this is different from 
        ourPostDist = self.getOpponentDist(gameState, newParticle, opponentIndex)

        #print ourPostDist
        #print ourPostDist.__class__ == Counter
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

  #HELPER METHODS FOR ELAPSE TIME


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

  #From p4 idk what this does - Martin
  # One JointInference module is shared globally across instances of MarginalInference
  #jointInference = JointParticleFilter()


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

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):

    legalPositions = self.findLegalPositions(gameState)

    self.jointInference = JointParticleFilter(self)
    self.jointInference.initialize(gameState, legalPositions)

    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.ourTeamAgents = self.getTeam(gameState)
    self.opponentAgents = self.getOpponents(gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    #PARTICLE FILTERING!!

    print "chooseAction"
    self.jointInference.observeState(gameState, self.index)

    dists = self.jointInference.modGetBeliefDistribution()

    self.displayDistributionsOverPositions(dists)

    self.jointInference.elapseTime(gameState)


    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    # pos = successor.getAgentState(self.index).getPosition()
    # if pos != nearestPoint(pos):
    #   # Only half a grid position was covered
    #   return successor.generateSuccessor(self.index, action)
    # else:
    #   return successor

    return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  #HELPER METHOD
  def findLegalPositions(self, gameState):
    mapHeight = gameState.getWalls().height
    mapWidth = gameState.getWalls().width

    legalPositions = []
    for x in range(mapWidth):
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          legalPositions.append((x,y))

    return legalPositions

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    #I need to figure out where my side is 
    agentState = gameState.data.agentStates[self.index]
    myPos = successor.getAgentState(self.index).getPosition()

    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height
    mySideList = []

    #red is always on the left; blue always on right 
    #find my side 
    #print gameState.isOnRedTeam(self.index)
    if gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)-1
      mySideX = x
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    #print not gameState.isOnRedTeam(self.index)
    if not gameState.isOnRedTeam(self.index):
      x = (mapWidth/2)
      mySideX = x
      #print "BLUE"
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    #print mySideList



    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry

      #If I'm carrying more than 2 food go back home
      if agentState.numCarrying < 2:
        
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistance
      else:
        minDistance = min([self.getMazeDistance(myPos, mySide) for mySide in mySideList])  
        features['distanceToFood'] = minDistance


    #make Pacman Scared 
    knownPositions = []
    knownAgentsIndices = []



    for i in self.opponentAgents:
        pos = gameState.getAgentPosition(i) 

        if pos != None:
          #print "Ghosts within range of offensive agent"
          knownPositions.append(pos)
          knownAgentsIndices.append(i)


          #rint knownPositions


    onMySide = True 
    if myPos[0] >= mySideX and gameState.isOnRedTeam(self.index):
      onMySide = False
    if myPos[0] <= mySideX and not gameState.isOnRedTeam(self.index):
      onMySide = False

    
    if len(knownAgentsIndices) != 0: #If a opponent is clsoe
      finalScore = 0 
      closestGhostDistance = min([self.getMazeDistance(myPos, opponentLocation) for opponentLocation in knownPositions])
      
      #print "IM SCARED"
      #print closestGhostDistance
      if(closestGhostDistance == 0):
        finalScore -= float("inf")
      if(closestGhostDistance == 1):
        finalScore -= 10
      if(closestGhostDistance == 2):
        finalScore -= 5
      if(closestGhostDistance == 3):
        finalScore -= 0.5
          

    
      if onMySide == False:
        features['scaredScore'] = finalScore
      if onMySide == True:
        if gameState == successor:
          features['scaredScore'] += 10


    return features

  def getWeights(self, gameState, action):
    Weights = {'successorScore': 100, 'distanceToFood': -1, 'scaredScore': 1}

    #if features['scaredScore'] > 0:
      
    return Weights

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

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

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  
          


