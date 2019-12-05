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
               first = 'OffensiveCaptureAgent', second = 'OffensiveCaptureAgent'):
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

  def __init__(self, numParticles=400):
      #NOTE: numParticles is the number of particles per set of particles
      #total particles is 2*numParticles
      self.setNumParticles(numParticles)

  def setNumParticles(self, numParticles):
      self.numParticles = numParticles

  def initialize(self, ourTeamAgents, opponentAgents, gameState, legalPositions, mazeDistanceAgent):
      "Stores information about the game, then initiaself.numGhosts = gameState.getNumAgents() - 1"

      #these are lists of indices
      self.ourTeamAgents = ourTeamAgents
      self.opponentAgents = opponentAgents

      self.legalPositions = legalPositions 
      self.enemySideList = self.getMySide(self.opponentAgents[0],gameState)
      self.mazeDistanceAgent = mazeDistanceAgent
      self.particles = util.Counter()

      
      self.jailPathList = self.findJailPath(gameState, mazeDistanceAgent, legalPositions)

      # print "0", gameState.getInitialAgentPosition(self.opponentAgents[0])
      # print "1", gameState.getInitialAgentPosition(self.opponentAgents[1])

      self.jailTimer = util.Counter()
      self.setParticlesToStart(gameState, self.opponentAgents[0])
      self.setParticlesToStart(gameState, self.opponentAgents[1])


  def findJailPath(self, gameState, mazeDistanceAgent, legalPositions):

    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height
    startPos = gameState.getInitialAgentPosition(self.opponentAgents[1])

    enemyHalfGrid = [[mazeDistanceAgent.getMazeDistance(startPos, (x,y)) if (x,y) in legalPositions else None for x in range(1,(mapWidth/2)+1)] for y in range(1, mapHeight+1)] 

    currentPos = startPos
    currentDist = 0 
    toReturnList = []
    notDone = True

    while(notDone):
      toReturnList.append(currentPos)
      x, y = currentPos

      neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
      checkCounter = 0
      for neighbor in neighbors:
        if neighbor in legalPositions:
          checkCounter += 1
          tempDist = mazeDistanceAgent.getMazeDistance(startPos, neighbor)
          tempPos = neighbor
          
          if (currentDist + 1) == tempDist:
            currentPos = tempPos
            currentDist = tempDist
            
      if checkCounter==3:
        notDone = False

    self.jailPaths = util.Counter()
    self.jailPaths[self.opponentAgents[0]] = toReturnList[1:len(toReturnList)]
    self.jailPaths[self.opponentAgents[1]] = toReturnList
    


  def printJailMap(self, enemyHalfGrid, gameState):
    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height

    out = [[str(enemyHalfGrid[x][y])[0] for x in range(mapWidth/2)] for y in range(mapHeight)]
    out.reverse()
    print '\n'.join([''.join(x) for x in out])



  def setParticlesToJailTimer(self, gameState, opponentAgentIndex, currentAgentIndex):
    self.particles[opponentAgentIndex] = []

    whereOnJailPath = self.jailPaths[opponentAgentIndex][-self.jailTimer[opponentAgentIndex]]

    for i in range(self.numParticles):
      self.particles[opponentAgentIndex].append(whereOnJailPath)

    if (currentAgentIndex == opponentAgentIndex+1) or opponentAgentIndex == 3:
      self.jailTimer[opponentAgentIndex] -= 1

    return whereOnJailPath


  def setParticlesToStart(self, gameState, opponentAgentIndex):
      #This should only be called at the very beginning of the game because thats the only time both particles are in jail
      print "set to Start"
      self.particles[opponentAgentIndex] = []

      #Find the starting postion of this opponent  
      startPos = gameState.getInitialAgentPosition(opponentAgentIndex)

      #this turns the list of startPos into a tuple
      startPos = tuple(startPos)
      #print "startPos: ", startPos

      #Note which agents these are depend will vary basedd on which team we are
      for i in range(self.numParticles):
          self.particles[opponentAgentIndex].append(startPos)


      self.jailTimer[opponentAgentIndex] = len(self.jailPaths[opponentAgentIndex])


  def initializeParticlesUniformly(self, gameState, opponentIndex):
    #reset particles to uniform
    import itertools

    self.particles[self.opponentAgents[opponentIndex]] = []

    # get all possible permutations of legal ghost positions
    possiblePositions = self.legalPositions

    # randomize permutations
    random.shuffle(possiblePositions)
    
    length = len(possiblePositions)

    for i in range(self.numParticles):
        self.particles[self.opponentAgents[opponentIndex]].append(possiblePositions[i%length])


  def hasBeenEaten(self, gameState, opponentAgentIndex, currentAgentIndex, thisAgent):

    previousGameState = thisAgent.getPreviousObservation()

    currentEnemyPos = gameState.getAgentPosition(opponentAgentIndex)
    currentMyPos = gameState.getAgentPosition(currentAgentIndex)
    startPos = gameState.getInitialAgentPosition(currentAgentIndex)

    if previousGameState != None:
      previousEnemyPos = previousGameState.getAgentPosition(opponentAgentIndex)
      previousMyPos = previousGameState.getAgentPosition(currentAgentIndex)
    else:
      return False #previousGameState only None during first turn

    if currentEnemyPos == None:
      currentDistance = None
    else:
      currentDistance = thisAgent.getMazeDistance(currentMyPos, currentEnemyPos)

    if previousEnemyPos == None:
      previousDistance = None
    else:
      previousDistance = thisAgent.getMazeDistance(previousMyPos, previousEnemyPos)


    if startPos == currentMyPos:  #We just got eaten, we shouldn't think they got eaten instead
      return False 

    if previousDistance == 1 and not (currentDistance == 1 or currentDistance == 2 or currentDistance == 3):
      self.sendParticlesToJail(gameState, opponentAgentIndex)
      return True
    else:
      return False

    

  def sendParticlesToJail(self, gameState, opponentAgentIndex):
    self.setParticlesToStart(gameState, opponentAgentIndex)


  def observeState(self, gameState, currentAgentIndex, thisAgent):
    """
    Reweights and then resamples the set of particles using the likelihood of the noisy
    observations.
    """



    #current agent's position
    #this is what we will be calculating the true distance based on
    myPos = gameState.getAgentPosition(currentAgentIndex) 

    enemyPosList = []
    #If we know where opponents are then all particles should agree with that evidence
    for opponentAgentIndex in self.opponentAgents:
      #If we can see the position then we know then it returns the correct position
      #Else, it returns None
      enemyPosList.append(gameState.getAgentPosition(opponentAgentIndex))


    #This returns an array of noisy Distances from our current agent
    noisyDistances = gameState.getAgentDistances()      
    particleWeights = util.Counter()
    particleDictionary = util.Counter()

    for i in range(2):

      
      particleWeights[self.opponentAgents[i]] = []
      particleDictionary[self.opponentAgents[i]] = util.Counter()

      if self.jailTimer[self.opponentAgents[i]] != 0:
        whereOnJailPath = self.setParticlesToJailTimer(gameState, self.opponentAgents[i], currentAgentIndex) #returns where on jail path
        particleDictionary[self.opponentAgents[i]][whereOnJailPath] = 1
      #Has Been Eaten
      elif self.hasBeenEaten(gameState, self.opponentAgents[i], currentAgentIndex, thisAgent):
        jailPos = gameState.getInitialAgentPosition(self.opponentAgents[i])
        particleDictionary[self.opponentAgents[i]][jailPos] = 1

      #Not Eaten
      else:
        for particleIndex, particle in enumerate(self.particles[self.opponentAgents[i]]):

          if enemyPosList[i] == None:
            # find the true distance from pacman to the current ghost that we're iterating through
            trueDistance = util.manhattanDistance(particle, myPos)
            # weight each particle by the probability of getting to that position (use emission model)
            # account for our current belief distribution (evidence) at this point in time
            particleWeights[self.opponentAgents[i]].append(gameState.getDistanceProb(trueDistance, noisyDistances[self.opponentAgents[i]]))

          else:
            #set particles to reality
            self.particles[self.opponentAgents[i]][particleIndex] = enemyPosList[i]
            particleWeights[self.opponentAgents[i]].append(1)




        # now create a counter and count up the particle weights observed
        
        for index, p in enumerate(self.particles[self.opponentAgents[i]]):
          particleDictionary[self.opponentAgents[i]][p] += particleWeights[self.opponentAgents[i]][index]

        particleDictionary[self.opponentAgents[i]].normalize()


      #reinitialize if 0 for that set of particles
      if particleDictionary[self.opponentAgents[i]].totalCount() == 0:
        self.initializeParticlesUniformly(gameState, i)

      # otherwise, go ahead and resample based on our new beliefs 
      else:
          
        keys = []
        values = []

        # find each key, value pair in our counter
        keys, values = zip(*particleDictionary[self.opponentAgents[i]].items())
        self.particles[self.opponentAgents[i]] = util.nSample(values, keys, self.numParticles)
      
  

  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the
    gameState.
    """
    newParticles = util.Counter()

    for i in range(2):

      newParticles[self.opponentAgents[i]] = []

      for oldParticle in self.particles[self.opponentAgents[i]]:

        newParticle = oldParticle 
        
        ourPostDist = self.getOpponentDist(gameState, newParticle, self.opponentAgents[i])
            
        newParticle = util.sample(ourPostDist)
        newParticles[self.opponentAgents[i]].append(tuple(newParticle))
    
    self.particles = newParticles


  #This methods isonly slightly different from the above get belief distribution method
  #Returns a slightly different formatted distribution for use in displaying the distribution on the map
  def getBeliefDistribution(self):
 
    # convert list of particles into a counter
    beliefsOpponentOne = util.Counter() 
    beliefsOpponentTwo = util.Counter() 

    # count each unique position 
    for p in self.particles[self.opponentAgents[0]]:
      beliefsOpponentOne[p] += 1

    for p in self.particles[self.opponentAgents[1]]:
      beliefsOpponentTwo[p] += 1
    
    # normalize the count above
    beliefsOpponentOne.normalize()
    beliefsOpponentTwo.normalize()

    return [beliefsOpponentOne, beliefsOpponentTwo]


  ##################################
  # HELPER METHODS FOR ELAPSE TIME #
  ##################################

  def getOpponentDist(self, gameState, particle, opponentIndex):
    #create a gamestate that corresponds to the particle 
    opponentPositionGameState = self.setOpponentPosition(gameState, particle, opponentIndex)

    dist = util.Counter()

    sidePointDistances = [self.mazeDistanceAgent.getMazeDistance(particle, sidePoint) for sidePoint in self.enemySideList]
    minDistToSide = min(sidePointDistances)
    closestSidePoint = self.enemySideList[sidePointDistances.index(minDistToSide)]
    resetParticle = False

    #ourLegalActions[ourSuccessorsEvalScores.index(max(ourSuccessorsEvalScores))]

    opponentLegalActions = opponentPositionGameState.getLegalActions(opponentIndex)
    prob = float(1)/float(len(opponentLegalActions))
    
    
    for action in opponentLegalActions:
      try:
        successor = opponentPositionGameState.generateSuccessor(opponentIndex, action)
        pos = successor.getAgentState(opponentIndex).getPosition()
        dist[pos] = prob
      except:
        print "ELAPSE TIME MARTIN EXCEPTION"
        print "original particle", particle 
        print "action", action
        print "all actions", opponentLegalActions
        print "areWeOnOurSide", self.isOpponentOnTheirSide(particle, gameState)
        dist[particle] = prob
       

      # if self.isOpponentOnTheirSide(opponentIndex, opponentPositionGameState):
      #   distToSide = float(self.mazeDistanceAgent.getMazeDistance(pos, closestSidePoint))
      #   dist[pos] = self.div(1,distToSide*1000)
      # else:
    
      
    dist.normalize()
    return dist

  def div(self, x,y):
    if y == 0:
        return 0
    return x / y

  def setOpponentPosition(self, gameState, opponentPosition, opponentIndex):
    "Sets the position of all opponents to the values in the particle and then returns that gameState"

    returnGameState = gameState
    checkGameState = gameState

    conf = game.Configuration(opponentPosition, game.Directions.STOP)
    #checkGameState.data.agentStates[opponentIndex] = game.AgentState(conf, True)

    if self.isOpponentOnTheirSide(opponentPosition, gameState):
      tempIsPacman = False
    else:
      tempIsPacman = True

    returnGameState.data.agentStates[opponentIndex] = game.AgentState(conf, tempIsPacman)
    return returnGameState

  def isOpponentOnTheirSide(self, myPos, gameState):
    """
    this returns true if this agent is on their side
    and false if this agent is on their enemy's side
    """ 

    #myPos = gameState.getAgentPosition(selfIndex)
    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height
    isOnRedTeam = gameState.isOnRedTeam(self.opponentAgents[0])

    # if we're on the red team
    if isOnRedTeam:
      x = (mapWidth/2)-1
      mySideX = x
   
    # if we're on the blue team 
    if not isOnRedTeam:
      x = (mapWidth/2)
      mySideX = x

    onMySide = True 
    if myPos[0] >= mySideX and isOnRedTeam:
      onMySide = False
    if myPos[0] <= mySideX and not isOnRedTeam:
      onMySide = False

    return onMySide  

  def getMySide(self, selfIndex, gameState):
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
    if gameState.isOnRedTeam(selfIndex):
      x = (mapWidth/2)-1
      mySideX = x
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    # if we're on the BLUE team
    #print not gameState.isOnRedTeam(self.index)
    if not gameState.isOnRedTeam(selfIndex):
      x = (mapWidth/2)
      mySideX = x
      #print "BLUE"
      for y in range(mapHeight):
        if not gameState.hasWall(x,y):
          mySideList.append((x,y))

    return mySideList



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
  qValues = util.Counter()
  
 
  
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
    


    BaseCaptureAgent.jointInference.initialize(self.ourTeamAgents, self.opponentAgents, gameState, legalPositions, self)

    self.start = gameState.getAgentPosition(self.index)
   


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    ######################
    # PARTICLE FILTERING #
    ######################
    
    
    start = time.time()
  
    self.jointInference.observeState(gameState, self.index, self)

    displayDist = self.jointInference.getBeliefDistribution()
    dists = self.jointInference.getBeliefDistribution()


    self.displayDistributionsOverPositions(displayDist)

    self.jointInference.elapseTime(gameState)

    print "Particle Filtering time:", time.time() - start

    ##########################
    # END PARTICLE FILTERING #
    ##########################

    #Displays my side List
    # sideDist = self.getMySideDist(self.mySideList)
    # self.displayDistributionsOverPositions([sideDist])

    action = self.getActionAlphaBeta(gameState, dists, self.index)
    
    print "Total time:", time.time() - start
    return action




  def getActionAlphaBeta(self, gameState, dists, currentAgentIndex):
    #It's necessarily pacman's turn cause this is at the root 

    DEPTH = 4 #In terms of moves, not plies

    mostLikelyState = (dists[0].argMax(), dists[1].argMax())
    probableGameState = self.setOpponentPositions(gameState, mostLikelyState)

    ourSuccessors = [] #list of GameStates
    ourSuccessorsEvalScores = [] #list of GameStates returned scores
    alpha = -float('Inf')
    beta = float('Inf')
    v = -float('Inf')

    ourLegalActions = gameState.getLegalActions(currentAgentIndex)

    for action in ourLegalActions:
      ourSuccessors.append(gameState.generateSuccessor(currentAgentIndex, action))
    
    for child in ourSuccessors:
      v = max([v, self.minRecursiveHelper(child, 1, currentAgentIndex+1, alpha, beta, DEPTH)])
      ourSuccessorsEvalScores.append(v)

      if(v > beta):
        break

      alpha = max([alpha, v])

    print ourSuccessorsEvalScores
    return ourLegalActions[ourSuccessorsEvalScores.index(max(ourSuccessorsEvalScores))]


  def maxRecursiveHelper(self, gameState, depthCounter, currentAgentIndex, alpha, beta, DEPTH):

    NUM_AGENTS = 4   
    v = -float('Inf')

    if currentAgentIndex>=4:
      currentAgentIndex = 0
    
    #cutoff test
    if(DEPTH == depthCounter):
      return self.evaluationFunction(gameState)

    #implement a terminal test
    if(gameState.isOver()):
      return self.evaluationFunction(gameState)

    ourLegalActions = gameState.getLegalActions(currentAgentIndex)
    for action in ourLegalActions:
      try: 
        child = gameState.generateSuccessor(currentAgentIndex, action)
      except: 
        print "Max in minimax - Martin Exception"
        continue
      v = max([v, self.minRecursiveHelper(child, depthCounter+1, currentAgentIndex+1, alpha, beta, DEPTH)])

      if(v > beta):
        return v

      alpha = max([alpha, v])

    return v


  def minRecursiveHelper(self, gameState, depthCounter, currentAgentIndex, alpha, beta, DEPTH):

    NUM_AGENTS = 4
    v = float('Inf')

    if currentAgentIndex>=4:
      currentAgentIndex = 0
    
    #cutoff test
    if(DEPTH == depthCounter):
      return self.evaluationFunction(gameState)

    #implement a terminal test
    if(gameState.isOver()):
      return self.evaluationFunction(gameState)


    opponentLegalActions = gameState.getLegalActions(currentAgentIndex)

    for action in opponentLegalActions:
      try:
       child = gameState.generateSuccessor(currentAgentIndex, action)
      except: 
        print "Min in minimax - Martin Exception"
        continue
      v = min([v, self.maxRecursiveHelper(child, depthCounter+1, currentAgentIndex+1, alpha, beta, DEPTH)])
      
      if(v < alpha):
        return v
    
      beta = min([beta, v])
    
    return v
     

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
    returnGameState = gameState

    for index, pos in enumerate(opponentPositions):
        conf = game.Configuration(pos, game.Directions.STOP)

        if self.areWeOnOurSide(gameState, self.opponentAgents[index]):
          tempIsPacman = False
        else:
          tempIsPacman = True

        returnGameState.data.agentStates[self.opponentAgents[index]] = game.AgentState(conf, tempIsPacman)
    return returnGameState

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


    dist = util.Counter

    return mySideList

  def getMySideDist(self, mySideList):
    dist = util.Counter()

    for position in mySideList:
      dist[position] = 1

    dist.normalize()
    return dist


  def areWeOnOurSide(self, gameState, selfIndex = None):
    """
    this returns true if our agent is on our side
    and false if our agent is on the enemy's side
    """ 
    if selfIndex == None:
      selfIndex = self.index

    myPos = gameState.getAgentPosition(selfIndex)
    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height

    # if we're on the red team
    if gameState.isOnRedTeam(selfIndex):
      x = (mapWidth/2)-1
      mySideX = x
   
    # if we're on the blue team 
    if not gameState.isOnRedTeam(selfIndex):
      x = (mapWidth/2)
      mySideX = x

    onMySide = True 
    if myPos[0] >= mySideX and gameState.isOnRedTeam(selfIndex):
      onMySide = False
    if myPos[0] <= mySideX and not gameState.isOnRedTeam(selfIndex):
      onMySide = False

    return onMySide

  def getPositions(self, currentGameState, findOurs):
    """
    This takes a gameState where we know everyones position
    and returns either enemy or our positions in a list

    If findOurs is true then this returns our positions,
    if false then returns enemy positions
    """
    allPositions = [currentGameState.getAgentPosition(i) for i in xrange(4)]
    

    ourPositions = []
    enemyPositions = []
    
    # want to have the correct positions assigned to the correct teams
    for i in xrange(4):
      if i in self.ourTeamAgents:
        ourPositions.append(allPositions[i])
      else:
        enemyPositions.append(allPositions[i])

    if findOurs:
      return ourPositions
    else:
      return enemyPositions


  ######################
  ##    Q LEARNING    ##
  ######################


  def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """

        # just return the qvalue from the counter 
        return self.qValues[(state, action)]


  def computeValueFromQValues(self, state):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      legalActions = self.getLegalActions(state)
    
      # if we're at the terminal state return none
      if len(legalActions) == 0:
        return 0.0

      maxQValue = -float("Inf")
      
      # otherwise take the action that corresponds with the highest q value
      for action in legalActions:
        qValue = self.getQValue(state, action)
        
        if qValue > maxQValue:
          maxQValue = qValue


      return maxQValue


  def computeActionFromQValues(self, state):
      """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
      """
         
      legalActions = self.getLegalActions(state)
      
      # if we're at the terminal state return none
      if len(legalActions) == 0:
        return None

      maxQValue = -float("Inf")
      bestAction = None
      
      # otherwise take the action that corresponds with the highest q value
      for action in legalActions:
        qValue = self.getQValue(state, action)
        
        if qValue > maxQValue:
          maxQValue = qValue
          bestAction = action

        # break tiebreakers randomly using random.choice()
        elif qValue == maxQValue:
          temp = [bestAction, action]
          
          # randomly keep the previous action or use the new one
          bestAction = random.choice(temp)


      return bestAction

 
  def getAction(self, state):
      """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
      """
      # Pick Action
      legalActions = self.getLegalActions(state)
      action = None
      
      # choose the best action at a probability of 1 - self.epsilon 
      # choose a random action at a probabilty of self.epsilon

      # if findRandom action is true this means we find random action (with probability epsilon)
      # if false it means we return the best action
      findRandom = util.flipCoin(self.epsilon)

      if findRandom:
        bestAction = random.choice(legalActions)

      else:
        bestAction = self.computeActionFromQValues(state)
         
      return bestAction

  def update(self, state, action, nextState, reward):
      """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
      """
      
      # old qvalue + alpha(immediate reward + discount*expected future reward - old q value)
      newValue = self.qValues[(state, action)] + (self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.qValues[(state, action)] ) )

      self.qValues[(state, action)] = newValue



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
    #capsuleScore = self.getCapsuleScore(currentGameState)
    #enemyClosenessScore = self.getEnemyClosenessScore(currentGameState)


    features["foodScore"] = foodScore
    #print "food score: ", foodScore
    #features["capsuleScore"] = capsuleScore
    #features["enemyClosenessScore"] = enemyClosenessScore
    #features["scoreOfGame"] = self.getScore(currentGameState)

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
    Weights = {"foodScore": 1000}#, "capsuleScore": 10, "enemyClosenessScore": 10, "scoreOfGame": 1000}

      
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
    
    # a list of all accesible positions on our side of the map
    mySideList = self.getMySide(gameState)
    #[length of food, minFoodDistance]
    foodStats = self.getFoodStats(gameState, gameState.getAgentPosition(self.index))
    numFood = foodStats[0]
    closestFoodDistance = foodStats[1]

    myPos = gameState.getAgentPosition(self.index)
    minDistanceHome = min([self.getMazeDistance(myPos, position) for position in mySideList])

    realGameState = self.getCurrentObservation()
    thisMinimaxState = gameState


    thisMinimaxStateScore = self.getScore(thisMinimaxState)
    realScore = self.getScore(realGameState)

    


    # if we're home than this is really good 
    if myPos in mySideList and thisMinimaxStateScore > realScore:
      print "We're home!"
      foodScore = 20000000.0

    # first check to see if our agent is carrying 3 food (or more) 
    # and there's no other food close by, then incentivize going home (to our side)
    elif gameState.getAgentState(self.index).numCarrying > 2 and closestFoodDistance > 2:
      foodScore = 1000000.0 * (1.0/minDistanceHome)
      
   
    # otherwise, we want to eat more food so reward states that are close to food
    else:  

      # reward states with less food 
      foodLeftScore = 1000000.0 * (1.0/numFood)

      # reward states that have food that is close by:
      # if food is right next to us this is really good
      if closestFoodDistance == 1:
        closestFoodScore = 400.0

      # otherwise make it so the closer the food, the higher the score
      else: 
        closestFoodScore = 100.0 * (1.0/closestFoodDistance)

      # create a final food score
      foodScore = closestFoodScore + foodLeftScore 

    return foodScore

  def getFoodStats(self, gameState, myPos):
    '''
    returns a list of [length of food, minFoodDistance]
    '''
    foodHalfGrid = self.getFood(gameState)
    numFood = 0
    minimumDistance = float('inf')

    for x in xrange(foodHalfGrid.width):
      for y in xrange(foodHalfGrid.height):
        if foodHalfGrid[x][y] == True:
          numFood += 1
          dist = self.getMazeDistance((x,y), myPos)
          if  dist < minimumDistance:
            minimumDistance = dist 

    return [numFood, minimumDistance]



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
      if closestEnemyDistance == 0:
        enemyClosenessScore = 1000.0

      else:
        enemyClosenessScore = 100.0 * (1.0/closestEnemyDistance)

    # otherwise it's not good to be close to enemies
    else:
      if closestEnemyDistance == 0:
        enemyClosenessScore = -1000.0

      else:
        enemyClosenessScore = -100.0 * (1.0/closestEnemyDistance)


    return enemyClosenessScore


  #################################
  ## helper methods for features ##
  #################################

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
    Weights = {"foodScore": 1000, "capsuleScore": 10, "numInvadersScore": 100, "enemyClosenessScore": 100000, "scoreOfGame": 1}

      
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
    # myPos = gameState.getAgentPosition(self.index)

    # [length of food, minFoodDistance]
    enemyPositions = self.getPositions(gameState, False)
    
    foodStats = self.getFoodStats(gameState, enemyPositions)
    
    numFood = foodStats[0]
    # this stores the closest distance between an enemy and a piece of food we're defending
    closestFoodDistance = foodStats[1]
    


    # a list of all accesible poisitions on our side of the map
    mySideList = self.getMySide(gameState)


    # if no food left in the game this is bad 
    if numFood == 0:
      foodScore = -1000000

    # otherwise there's still food left
    else:
      
              
      # punish states with less food 
      foodLeftScore = -100.0 * (1.0/numFood)

      # punish states that have food close to an enemy:
      # if food is right next to enemy this is really bad
      if closestFoodDistance == 1:
        closestFoodScore = -200.0

      # otherwise make it so the closer the food, the lower the score
      else: 
        closestFoodScore = -100.0 * (1.0/closestFoodDistance)

      # create a final food score
      foodScore = closestFoodScore + foodLeftScore 

    return foodScore

  def getFoodStats(self, gameState, enemyPositions):
    '''
    returns a list of [length of food, minFoodDistance]
    '''
    foodHalfGrid = self.getFoodYouAreDefending(gameState)
    numFood = 0
    minimumDistance = float('inf')


    for x in range(foodHalfGrid.width):
      for y in range(foodHalfGrid.height):
        if foodHalfGrid[x][y] == True:
          numFood += 1
      
          for enemy in enemyPositions:       
            dist = self.getMazeDistance((x,y), enemy)
            if  dist < minimumDistance:
              minimumDistance = dist 

    return [numFood, minimumDistance]



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
      if closestEnemyDistance == 0:
        enemyClosenessScore = 1000.0

      else:
        enemyClosenessScore = 100.0 * (1.0/closestEnemyDistance)

    # otherwise it's not good to be close to enemies
    else:
      if closestEnemyDistance == 0:
        enemyClosenessScore = -1000.0

      else:
        enemyClosenessScore = -100.0 * (1.0/closestEnemyDistance)

    #print "closestEnemyDistance: ", closestEnemyDistance
    #print "enemyClosenessScore: ", enemyClosenessScore

    return enemyClosenessScore

          


