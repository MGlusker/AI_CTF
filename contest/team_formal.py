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



#############################
#  PARTICLE FILTERING CLASS #
#############################

class JointParticleFilter:
  """
  JointParticleFilter tracks a joint distribution over tuples of all ghost
  positions.
  """

  def __init__(self, numParticles=300):
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

    indexOnJailPath =  len(self.jailPaths[opponentAgentIndex]) - self.jailTimer[opponentAgentIndex]
    # if indexOnJailPath == -1:
    #   indexOnJailPath == 

    whereOnJailPath = self.jailPaths[opponentAgentIndex][indexOnJailPath]

    for i in range(self.numParticles):
      self.particles[opponentAgentIndex].append(whereOnJailPath)

    if (currentAgentIndex == opponentAgentIndex+1) or (opponentAgentIndex == 3 and currentAgentIndex == 0):
      self.jailTimer[opponentAgentIndex] -= 1


    return whereOnJailPath


  def setParticlesToStart(self, gameState, opponentAgentIndex):
      #This should only be called at the very beginning of the game because thats the only time both particles are in jail
      #print "set to Start"
      self.particles[opponentAgentIndex] = []

      #Find the starting postion of this opponent  
      startPos = gameState.getInitialAgentPosition(opponentAgentIndex)

      #this turns the list of startPos into a tuple
      startPos = tuple(startPos)
      #print "startPos: ", startPos

      #Note which agents these are depend will vary basedd on which team we are
      for i in range(self.numParticles):
          self.particles[opponentAgentIndex].append(startPos)


      self.jailTimer[opponentAgentIndex] = len(self.jailPaths[opponentAgentIndex])-1


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
    hasBeenEatenList = []

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
      hasBeenEaten = self.hasBeenEaten(gameState, self.opponentAgents[i], currentAgentIndex, thisAgent)
      hasBeenEatenList.append(hasBeenEaten)

      if self.jailTimer[self.opponentAgents[i]] != 0 and enemyPosList[i] == None:
        whereOnJailPath = self.setParticlesToJailTimer(gameState, self.opponentAgents[i], currentAgentIndex) #returns where on jail path
        particleDictionary[self.opponentAgents[i]][whereOnJailPath] = 1

      #Has Been Eaten
      elif hasBeenEaten and enemyPosList[i] == None:
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
      
    return hasBeenEatenList


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

    #ourLegalActions[ourSuccessorsEvalScores.index(max(ourSuccessorsEvalScores))]

    opponentLegalActions = opponentPositionGameState.getLegalActions(opponentIndex)
    prob = float(1)/float(len(opponentLegalActions))
    
    
    for action in opponentLegalActions:
      try:
        successor = opponentPositionGameState.generateSuccessor(opponentIndex, action)
        pos = successor.getAgentState(opponentIndex).getPosition()
        dist[pos] = prob
      except:
        #print "MARTIN EXCEPTION"
        #print "original particle", particle 
        #print "action", action
        #print "all actions", opponentLegalActions
        #print "areWeOnOurSide", self.isOpponentOnTheirSide(particle, gameState)
        dist[particle] = prob

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
  
  # initially empty 
  enemyScaredTimes = util.Counter()

  offensiveBooleans = [True, False]

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
    enemyScaredTimes = self.initializeScaredTimes()
    


    BaseCaptureAgent.jointInference.initialize(self.ourTeamAgents, self.opponentAgents, gameState, legalPositions, self)

    self.start = gameState.getAgentPosition(self.index)

    if self.index == self.ourTeamAgents[0]:
      self.isOnOffense =  self.offensiveBooleans[0] #On Offense
    elif self.index == self.ourTeamAgents[1]:
      self.isOnOffense = self.offensiveBooleans[1] #On Defense

    

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    ######################
    # PARTICLE FILTERING #
    ######################  
   
    self.start = time.time()
  
    hasBeenEatenList = self.jointInference.observeState(gameState, self.index, self)
    
    #print "Observe Filtering time:", time.time() - self.start

    displayDist = self.jointInference.getBeliefDistribution()
    dists = self.jointInference.getBeliefDistribution()

    
    self.displayDistributionsOverPositions(displayDist)
   

    elapseStart = time.time()

    self.jointInference.elapseTime(gameState)

    #print "Elapse Filtering time:", time.time() - elapseStart
    ##########################
    # END PARTICLE FILTERING #
    ##########################

    #Displays my side List
    # sideDist = self.getMySideDist(self.mySideList)
    # self.displayDistributionsOverPositions([sideDist])

    #self.switch(gameState)
    
    action = self.getActionAlphaBeta(gameState, dists, self.index)
    

    

    #print "enemy scared Time: ", self.enemyScaredTimes

    self.updateScaredTimes(action, gameState, hasBeenEatenList)

    print "Total time:", time.time() - self.start
    #print ""

    return action



  def getActionAlphaBeta(self, gameState, dists, currentAgentIndex):
    #It's necessarily pacman's turn cause this is at the root 
    self.DEPTH = 4
    self.timesUp = False
    self.timesUpfeatureOn = False
    self.totalPrunes = 0
    

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
      v = max([v, self.minRecursiveHelper(child, 1, currentAgentIndex+1, alpha, beta, self.DEPTH)])
      ourSuccessorsEvalScores.append(v)

      if(v > beta):
        break

      alpha = max([alpha, v])

    #print ourSuccessorsEvalScores
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
        #print "Max in minimax - Martin Exception"
        continue
      v = max([v, self.minRecursiveHelper(child, depthCounter+1, currentAgentIndex+1, alpha, beta, DEPTH)])

      if(v > beta) or self.timesUp:
        self.totalPrunes += 1
        return v
        

      alpha = max([alpha, v])

    #print float(time.time() - self.start) > float(0.95)
    if float(time.time() - self.start) > float(0.95) and self.timesUpfeatureOn:
      
      self.timesUp = True

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
        #print "Max in minimax - Martin Exception"
        continue
      v = min([v, self.maxRecursiveHelper(child, depthCounter+1, currentAgentIndex+1, alpha, beta, DEPTH)])
      
      if(v < alpha) or self.timesUp:
        self.totalPrunes += 1
        return v
    
      beta = min([beta, v])

    if float(time.time() - self.start) > float(0.95) and self.timesUpfeatureOn:
      self.timesUp = True
    
    return v
     

  def evaluationFunction(self, currentGameState): 
    """
    Computes a linear combination of features and feature weights
    """
    isOnOffense = self.isOnOffense 

    if isOnOffense:
      features = self.getOffensiveFeatures(currentGameState)
      weights = self.getOffensiveWeights(currentGameState)
    else:
      features = self.getDefensiveFeatures(currentGameState)
      weights = self.getDefensiveWeights(currentGameState)

    return features * weights
 

  def initializeScaredTimes(self):
    self.enemyScaredTimes[self.opponentAgents[0]] = 0
    self.enemyScaredTimes[self.opponentAgents[1]] = 0


  def updateScaredTimes(self, action, gameState, hasBeenEatenList):
    #print "self.index", self.index
    currentState = gameState
    successor = gameState.generateSuccessor(self.index, action)

    currentNumCapsules = len(self.getCapsules(gameState))
    actionNumCapsules = len(self.getCapsules(successor))

    prevOpponent = self.index - 1 
    if prevOpponent == -1:
      prevOpponent = 3
  
    if actionNumCapsules < currentNumCapsules:
      
      self.enemyScaredTimes[self.opponentAgents[0]] = 40
      self.enemyScaredTimes[self.opponentAgents[1]] = 40
      return
    
    #decrement the correct opponent
    elif self.enemyScaredTimes[prevOpponent] > 0: 
      
      self.enemyScaredTimes[prevOpponent] -= 1
    elif self.enemyScaredTimes[prevOpponent] > 0: 
      
      self.enemyScaredTimes[prevOpponent] -= 1


    # if agent gets eaten set the scared timer of it to 0
    if hasBeenEatenList[0] == True and self.enemyScaredTimes[self.opponentAgents[0]] > 0:
      
      self.enemyScaredTimes[self.opponentAgents[0]] = 0 
    if hasBeenEatenList[1] == True and self.enemyScaredTimes[self.opponentAgents[1]] > 0:
      
      self.enemyScaredTimes[self.opponentAgents[1]] = 0 



  def switch(self, gameState):

    currentIsOnOffense = self.isOnOffense
    currentJailTimerOne = self.jointInference.jailTimer[self.opponentAgents[0]]
    currentJailTimerTwo = self.jointInference.jailTimer[self.opponentAgents[1]]

    numInvaders = self.getNumInvadersScore(gameState)
    print numInvaders
    
    if False:
      print "NO"
    if self.getScore(gameState) >= 8:#winning by a lot 
      print "both on DEFENSE"
      self.offensiveBooleans = [False, False]

    # elif self.getScore(gameState) <= -8: #losing by a lot
    #   print "both on offense"
    #   self.offensiveBooleans = [True, True]

    else: #somewhere in between 
      self.offensiveBooleans = [True, False]


      if currentJailTimerTwo > 0  or currentJailTimerOne > 0:
        print "JAIL so both offense"
        self.offensiveBooleans = [True, True]

    print self.offensiveBooleans
    if self.index == self.ourTeamAgents[0] and self.areWeOnOurSide(gameState):
      self.isOnOffense =  self.offensiveBooleans[0] #On Offense
    elif self.index == self.ourTeamAgents[1] and self.areWeOnOurSide(gameState):
      self.isOnOffense = self.offensiveBooleans[1]
    
    #print "offensiveBooleans", self.offensiveBooleans
 


  ##################
  # HELPER METHODS #
  ##################


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

        #print "setOpponentPositions opponentIndex", self.opponentAgents[index]
        if self.isOpponentOnTheirSide(pos, gameState):
          tempIsPacman = False
        else:
          tempIsPacman = True

        returnGameState.data.agentStates[self.opponentAgents[index]] = game.AgentState(conf, tempIsPacman)
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
    if myPos[0] > mySideX and gameState.isOnRedTeam(selfIndex):
      onMySide = False
    if myPos[0] < mySideX and not gameState.isOnRedTeam(selfIndex):
      onMySide = False

    return onMySide


  def getOffensiveFeatures(self, currentGameState):

    features = util.Counter()

    #ourPositions = self.getPositions(currentGameState, True)
    #enemyPositions = self.getPositions(currentGameState, False)
    
    
    
    #currentEnemyScaredTimes = [enemyState.scaredTimer for enemyState in enemyCurrentStates]

    #print currentEnemyScaredTimes

    offensiveFoodScore = self.getOffensiveFoodScore(currentGameState)
    offensiveCapsuleScore = self.getOffensiveCapsuleScore(currentGameState)
    offensiveEnemyClosenessScore = self.getOffensiveEnemyClosenessScore(currentGameState)

    features["offensiveFoodScore"] = offensiveFoodScore
    features["offensiveCapsuleScore"] = offensiveCapsuleScore
    features["offensiveEnemyClosenessScore"] = offensiveEnemyClosenessScore
    features["scoreOfGame"] = self.getScore(currentGameState)

    #print "FS: ", offensiveFoodScore
    #print "ECS: ", offensiveEnemyClosenessScore
    #print "CS: ", offensiveCapsuleScore
    #print "ACTUAL SCORE: ", self.getScore(currentGameState)
    
    print "FOOD: ", 1*offensiveFoodScore
    print "CAPSULE: ", 100*offensiveCapsuleScore
    print "ENEMY: ", 50*offensiveEnemyClosenessScore
    print "ACTUAL SCORE: ", 1000*self.getScore(currentGameState)
    print ""

    return features

  
  def getOffensiveWeights(self, gameState):
    # offensiveFoodScore is positive
    # capsuleScore is positive
    # offensiveEnemyClosenessScore is negative
    # socreOfGame is negative if losing, positive if winning
    Weights = {"offensiveFoodScore": 1, "offensiveEnemyClosenessScore": 50, "offensiveCapsuleScore": 100, "scoreOfGame": 1000}

      
    return Weights

  ###################################
  ## methods to get feature scores ##
  ###################################
  
  def getOffensiveFoodScore(self, gameState):
    """
    Returns a score based on how much food is on the board 
    and based on how close we are to food
    (less food is better and being closer to food is better)
    """


    offensiveFoodScore = 0.0
    
    # a list of all accesible positions on our side of the map
    mySideList = self.mySideList
    #[length of food, minFoodDistance]
    offensiveFoodStats = self.getOffensiveFoodStats(gameState, gameState.getAgentPosition(self.index))
    numFood = offensiveFoodStats[0]
    closestFoodDistance = offensiveFoodStats[1]

    myPos = gameState.getAgentPosition(self.index)
    minDistanceHome = min([self.getMazeDistance(myPos, position) for position in mySideList])

    realGameState = self.getCurrentObservation()
    thisMinimaxState = gameState


    thisMinimaxStateScore = self.getScore(thisMinimaxState)
    realScore = self.getScore(realGameState)

    
    currentJailTimerOne = self.jointInference.jailTimer[self.opponentAgents[0]]
    currentJailTimerTwo = self.jointInference.jailTimer[self.opponentAgents[1]]
  

    enemyScaredTimes = self.enemyScaredTimes
    maxEnemyScaredTime = -float("inf")
    minEnemyScaredTime = float("inf")
    maxEnemyScaredIndex = 0
    for key, value in enemyScaredTimes.items():
      if value < minEnemyScaredTime:
        minEnemyScaredTime = value
      if value > maxEnemyScaredTime:
        maxEnemyScaredIndex = key
        maxEnemyScaredTime = value





    # if we're home than this is really good 
    if myPos in mySideList and thisMinimaxStateScore > realScore:
      #print "We're home!"
      offensiveFoodScore = 2000000000.0

    # first check to see if our agent is carrying 2 food (or more) 
    # and there's no other food close by, then incentivize going home (to our side)


    # if no one has been sent to jail and no ghosts are scared
    elif (gameState.getAgentState(self.index).numCarrying >= 2) and (currentJailTimerOne == 0 and currentJailTimerTwo == 0) and (maxEnemyScaredTime == 0):  
      #and closestFoodDistance > 2:
      #print "CARRYING MORE THAN 2:", closestFoodDistance
      #print "HOLD Less"
      #print gameState.getAgentState(self.index).numCarrying
      offensiveFoodScore = 100000000.0 * (1.0/minDistanceHome)
      
    elif gameState.getAgentState(self.index).numCarrying >= 4 and (currentJailTimerOne > 0 or currentJailTimerTwo > 0):
      #print "HOLD more"
      #print gameState.getAgentState(self.index).numCarrying
      offensiveFoodScore = 100000000.0 * (1.0/minDistanceHome)

    elif gameState.getAgentState(self.index).numCarrying >= 6 and (maxEnemyScaredTime > 0):
      #print "HOLD MOST"
      #print gameState.getAgentState(self.index).numCarrying
      offensiveFoodScore = 100000000.0 * (1.0/minDistanceHome)

    # otherwise, we want to eat more food so reward states that are close to food
    else:  

      if numFood == 0:
        foodLeftScore = 10000000.0
      
      else: 
        # reward states with less food 
        foodLeftScore = 1000000.0 * (1.0/numFood)

      # reward states that have food that is close by:
      # if food is right next to us this is really good
      if closestFoodDistance == 1:
        closestFoodScore = 40.0

      # otherwise make it so the closer the food, the higher the score
      else: 
        closestFoodScore = 10.0 * (1.0/closestFoodDistance)

      # create a final food score
      offensiveFoodScore = closestFoodScore + foodLeftScore 

    #offensiveFoodScore = offensiveFoodScoree / 1000000.0

    #print offensiveFoodScore


    return offensiveFoodScore

  def getOffensiveFoodStats(self, gameState, myPos):
    '''
    returns a list of [length of food, minFoodDistance]
    '''
    foodHalfGrid = self.getFood(gameState)
    numFood = 0
    minimumDistance = float('inf')

    for x in range(foodHalfGrid.width):
      for y in range(foodHalfGrid.height):
        if foodHalfGrid[x][y] == True:
          numFood += 1
          dist = self.getMazeDistance((x,y), myPos)
          if  dist < minimumDistance:
            minimumDistance = dist 

    return [numFood, minimumDistance]



  def getOffensiveCapsuleScore(self, gameState):
    #this is meant as an offensive capsule score

    #This is for capsules we are trying to eat

    offensiveCapsuleScore = 0.0

    capsuleList = self.getCapsules(gameState)

    distanceToCapsules = []
   
    #minCapsuleDistance = None 
    
    
    for capsule in capsuleList:
      distanceToCapsules.append(self.getMazeDistance(gameState.getAgentPosition(self.index), capsule))

    # if no capsules left in game this is good
    if len(distanceToCapsules) == 0:

      #print "NO CAPSULES LEFT"
      numCapsulesLeftScore = 500.0
      offensiveCapsuleScore = 0.0

    # otherwise reward states with fewer capsules 
    else: 


      numCapsulesLeftScore = 250.0 * (1.0/len(distanceToCapsules))

      minCapsuleDistance = min(distanceToCapsules)
      #print "MIN CAPSULE DIST: ", minCapsuleDistance

      # reward being close to capsules
      if minCapsuleDistance == 1:
        offensiveCapsuleScore = 5.0
      
      else:
        offensiveCapsuleScore = 1.0 * (1.0/(minCapsuleDistance)) #+closestGhostDistance))
    

    return offensiveCapsuleScore + numCapsulesLeftScore
    
  def getOffensiveEnemyClosenessScore(self, gameState): 
    """
    punish our agent being close to enemies 
    (unless we're on our own side)
    """
    enemyScaredTimes = self.enemyScaredTimes
    maxEnemyScaredTime = -float("inf")
    minEnemyScaredTime = float("inf")
    maxEnemyScaredIndex = 0
    for key, value in enemyScaredTimes.items():
      if value < minEnemyScaredTime:
        minEnemyScaredTime = value
      if value > maxEnemyScaredTime:
        maxEnemyScaredIndex = key
        maxEnemyScaredTime = value

    ourScaredTimes = [gameState.getAgentState(us).scaredTimer for us in self.ourTeamAgents]

    # a boolean telling us if we're on our own side or not
    onMySide = self.areWeOnOurSide(gameState)

    # a list of the enemy positions (as determined by particleFiltering)
    enemyPositions = self.getPositions(gameState, False)
    
    distanceToEnemies = []
    # find distance to each enemy
    for enemy in enemyPositions:
      distanceToEnemies.append(self.getMazeDistance(gameState.getAgentPosition(self.index), enemy))

    closestEnemyDistance = min(distanceToEnemies)

    
    # REAL ENEMY POSITIONS
    myRealPosition = self.getCurrentObservation().getAgentPosition(self.index)
    realEnemyPositions = self.getPositions(self.getCurrentObservation(), False)
    realDistanceToEnemies = []
    
    for enemy in realEnemyPositions:
      realDistanceToEnemies.append(self.getMazeDistance(myRealPosition, enemy))

    closestRealEnemyDistance = min(realDistanceToEnemies)
    closestRealEnemyPosition = realEnemyPositions[realDistanceToEnemies.index(closestRealEnemyDistance)]
    closestRealEnemyIndex = realDistanceToEnemies.index(closestRealEnemyDistance)



    enemyClosenessScore = 0.0




    #onMySide = False
    # if we're on our side it's good to be close to enemies (UNLESS WE"RE SCARED)
    if onMySide:
      
      # if we're not scared of any enemies try to get close
      if max(ourScaredTimes) == 0:
        
        #print "new enemy closest Distance: ", distanceToEnemies[closestRealEnemyIndex]
        #print "closest real enemy distance: ", closestRealEnemyDistance

        if (distanceToEnemies[closestRealEnemyIndex]-closestRealEnemyDistance) > 6:
        #if (closestEnemyDistance-closestRealEnemyDistance) > 1:
          #print "NO FUCKING WAY *********************"
          enemyClosenessScore = 10.0
        
        elif closestEnemyDistance == 0:
          getEnemyClosenessScore = 9.0

        elif closestEnemyDistance == 1:
          enemyClosenessScore = 5.0
        else:
          enemyClosenessScore = 5.0 * (1.0/closestEnemyDistance)

      
      # otherwise we're scared so run away
      else: 
        if closestEnemyDistance == 0:
          enemyClosenessScore = 0 
        elif closestEnemyDistance == 1:
          enemyClosenessScore = 1
        elif closestEnemyDistance == 2:
          enemyClosenessScore = 2
        elif closestEnemyDistance == 3:
          enemyClosenessScore = 3
        
        else:
          enemyClosenessScore = 5


    # otherwise on enemy side
    else:
    

      # if either ghost is scared
      if maxEnemyScaredTime > 0 or minEnemyScaredTime > 0: 
        
        # then ignore ghosts all together 
        # then this is good!
        


        # if we eat a ghost the new enemy closest distance
        # should be greater than it was before you took that action

        # print "new enemy closest Distance: ", distanceToEnemies[closestRealEnemyIndex]
        # print "closest real enemy distance: ", closestRealEnemyDistance

        if (distanceToEnemies[closestRealEnemyIndex]-closestRealEnemyDistance) > 6:
          print "EAT"
          enemyClosenessScore = 1000000000

        else:
          enemyClosenessScore = 1000000
        #  enemyClosenessScore = 1000.0
        #elif closestEnemyDistance == 0:
        #  enemyClosenessScore = 100.0
        #elif closestEnemyDistance == 1:
        #  enemyClosenessScore = 50.0
        #else:
        #  enemyClosenessScore = 50.0 * (1.0/closestEnemyDistance)
        
      
      # otherwise lets be a little bit scared of ghosts
      else:  
        if closestEnemyDistance == 0:
          enemyClosenessScore = 0 
        elif closestEnemyDistance == 1:
          enemyClosenessScore = 1
        elif closestEnemyDistance == 2:
          enemyClosenessScore = 2
        elif closestEnemyDistance == 3:
          enemyClosenessScore = 3
        #elif closestEnemyDistance > 3 and closestEnemyDistance < 7:      
        #  enemyClosenessScore = 5
        else:
          enemyClosenessScore = 5
        



    return enemyClosenessScore


  #################################
  ## helper methods for features ##
  #################################




#######################
##  Defensive Agent  ##
#######################


  def getDefensiveFeatures(self, currentGameState):

    features = util.Counter()

    #ourPositions = self.getPositions(currentGameState, True)
    #enemyPositions = self.getPositions(currentGameState, False)
    onMySide = self.areWeOnOurSide(self.getCurrentObservation())
    
    
    #currentEnemyScaredTimes = [enemyState.scaredTimer for enemyState in enemyCurrentStates]

    #print currentEnemyScaredTimes

    defensiveFoodScore = self.getDefensiveFoodScore(currentGameState)
    defensiveCapsuleScore = self.getDefensiveCapsuleScore(currentGameState)
    numInvadersScore = self.getNumInvadersScore(currentGameState)
    defensiveEnemyClosenessScore = self.getDefensiveEnemyClosenessScore(currentGameState)
    #goHomeScore = self.getGoHomeScore(currentGameState)

      
    features["defensiveFoodScore"] = defensiveFoodScore
    features["defensiveCapsuleScore"] = defensiveCapsuleScore
    features["numInvadersScore"] = numInvadersScore
    features["defensiveEnemyClosenessScore"] = defensiveEnemyClosenessScore
    features["scoreOfGame"] = self.getScore(currentGameState)
    

    #print "FOOD: ", defensiveFoodScore
    #print "CAPSULE: ", defensiveCapsuleScore
    #print "NUM INVADERS: ", numInvadersScore
    #print "ENEMY: ", defensiveEnemyClosenessScore
    #print "ACTUAL SCORE: ", self.getScore(currentGameState)
    
    return features

  
  def getDefensiveWeights(self, gameState):
    Weights = {"defensiveFoodScore": 1000000, "defensiveEnemyClosenessScore": 1000, "numInvadersScore": -25000,  "defensiveCapsuleScore": 1, "scoreOfGame": 1000}

      
    return Weights

  #######################################
  ## helper methods for feature scores ##
  #######################################
  

  
  def getDefensiveFoodScore(self, gameState):
    """
    Returns a score based on how much food is on the board 
    and based on how close enemies are to food
    (more food is better and being closer to food is worse)
    """

    defensiveFoodScore = 0.0
    # myPos = gameState.getAgentPosition(self.index)

    

    enemyPositions = self.getPositions(gameState, False)
    
    # [length of food, minFoodDistance]
    defensiveFoodStats = self.getDefensiveFoodStats(gameState, enemyPositions)
    
    # number of food pelets we're defending in this gamestate
    numFood = defensiveFoodStats[0]
    #print "numFood", numFood
    # this stores the closest distance between an enemy and a piece of food we're defending
    closestFoodDistance = defensiveFoodStats[1]
    


    # a list of all accesible poisitions on our side of the map
    mySideList = self.getMySide(gameState)


    # if no food left in the game then we lose 
    if numFood == 0:
      defensiveFoodScore = -float('inf')

    # otherwise there's still food left
    else:
      
              
      # punish states with less food 
      foodLeftScore = -10000.0 * (1.0/numFood)

      # more points for more food
      #defensiveFoodScore = numFood * 5.0

      # punish states that have food close to an enemy:
      # if food is right next to enemy this is really bad
      #if closestFoodDistance == 0:
      #  closestFoodScore = -200.0

      # otherwise make it so the closer the food, the lower the score
      #else: 
      #  closestFoodScore = -100.0 * (1.0/closestFoodDistance)

      # create a final food score
      #defensiveFoodScore = closestFoodScore + foodLeftScore 

    return defensiveFoodScore

  def getDefensiveFoodStats(self, gameState, enemyPositions):
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



  def getDefensiveCapsuleScore(self, gameState):
    #this is meant as an defensive capsule score

    #This is for capsules we are trying to eat

    defensiveCapsuleScore = 0.0

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
      defensiveCapsuleScore = -50.0

    # otherwise reward states with more capsules 
    else: 
      minCapsuleDistance = min(distanceToCapsules)
      
      # punish enemy being very close to capsule
      if minCapsuleDistance == 0:
        defensiveCapsuleScore = -500.0
      
      # punish enemies being close to capsules
      else:
        defensiveCapsuleScore = -100.0 * (1.0/minCapsuleDistance) #+closestGhostDistance))
    

    return defensiveCapsuleScore
    
  
  def getNumInvadersScore(self, gameState):
    """
    counts how many invaders are on our side and returns
    a lower score for more invaders
    """

    enemyPositions = self.getPositions(gameState, False)

    # count how many invaders are on our side
    numInvaders = 0

    mapWidth = gameState.getWalls().width
    mapHeight = gameState.getWalls().height
    allPosList = []

    #red is always on the left; blue always on right 
    # if we're on the RED team 
    #print gameState.isOnRedTeam(self.index)
    if gameState.isOnRedTeam(self.index):
      width = (mapWidth/2)-1
      #mySideX = x
      for y in range(mapHeight):
        for x in range(width):
          if not gameState.hasWall(x,y):
            allPosList.append((x,y))

    # if we're on the BLUE team
    #print not gameState.isOnRedTeam(self.index)
    if not gameState.isOnRedTeam(self.index):
      width = (mapWidth/2)
      #mySideX = x
      #print "BLUE"
      for y in range(mapHeight):
        for x in range(width):
          if not gameState.hasWall(x,y):
            allPosList.append((x,y))

    
    #print allPosList

    for enemy in enemyPositions:
      for position in allPosList:
        if enemy == position:
          #print "ENEMYTYYYYYY"
          numInvaders += 1


    return numInvaders 

    

  def getDefensiveEnemyClosenessScore(self, gameState): 
    """
    reward our agent being close to invaders 
    (unless we're on their side)
    """

    #enemyScaredTimes = self.enemyScaredTimes
    ourScaredTimes = [gameState.getAgentState(us).scaredTimer for us in self.ourTeamAgents]
    # a boolean telling us if we're on our own side or not
    onMySide = self.areWeOnOurSide(gameState)

    myRealPosition = self.getCurrentObservation().getAgentPosition(self.index)
    realEnemyPositions = self.getPositions(self.getCurrentObservation(), False)
    realDistanceToEnemies = []
    
    for enemy in realEnemyPositions:
      realDistanceToEnemies.append(self.getMazeDistance(myRealPosition, enemy))

    closestRealEnemyDistance = min(realDistanceToEnemies)
    closestRealEnemyPosition = realEnemyPositions[realDistanceToEnemies.index(closestRealEnemyDistance)]
    closestRealEnemyIndex = realDistanceToEnemies.index(closestRealEnemyDistance)
    #print closestRealEnemyDistance, "closestrealenemyDist"
    #print closestRealEnemyPosition, " Position"
    #print closestRealEnemyIndex, "index "
    

    myPos = gameState.getAgentPosition(self.index)
    enemyPositions = self.getPositions(gameState, False)  
    distanceToEnemies = []
    # find distance to each invader
    for enemy in enemyPositions:
      distanceToEnemies.append(self.getMazeDistance(myPos, enemy))

    closestEnemyDistance = min(distanceToEnemies)
    closestEnemyPosition = enemyPositions[distanceToEnemies.index(closestEnemyDistance)]

    
    defensiveEnemyClosenessScore = 0.0


    # if we're on our side it's good to be close to enemies (UNLESS WE"RE SCARED)
    if onMySide:
    #print "ONMYSIDE"
    
    # if we're not scared of any enemies try to get close
      #if max(ourScaredTimes) == 0:
      if True:
        #print "EAT GHOST"
        # if we eat the enemy
        #print "distance that will happen", distanceToEnemies[closestRealEnemyIndex]
        #print "actual current distance", closestRealEnemyDistance, "index: ", closestRealEnemyIndex
        #if distanceToEnemies[closestRealEnemyIndex] == 0:
        

        if (distanceToEnemies[closestRealEnemyIndex]-closestRealEnemyDistance) > 4:
        #if (closestEnemyDistance-closestRealEnemyDistance) > 1:
          #print "NO FUCKING WAY *********************"
          #enemyClosenessScore = 100.0
      
          #if closestRealEnemyDistance == 1 and gameState.getAgentPosition(closestRealEnemyIndex) == None:
          #print "ghost eaten"
          defensiveEnemyClosenessScore = 1000000000.0

        elif closestEnemyDistance == 0:
          defensiveEnemyClosenessScore = 1000.0
        else:
          defensiveEnemyClosenessScore = 100.0 * (1.0/closestEnemyDistance)


      
      # otherwise we're scared so run away
      else: 
        #print "wErE sCaReD"
        if closestEnemyDistance == 0:
          defensiveEnemyClosenessScore = -1000.0

        else:
          defensiveEnemyClosenessScore = -100.0 * (1.0/closestEnemyDistance)


    # otherwise we're on the other side so go home
    """
    else:
      
      minDistanceHome = min([self.getMazeDistance(myPos, position) for position in self.mySideList])
      
      if myPos in self.mySideList:
        #print "We're home!"
        defensiveEnemyClosenessScore = 2000000000.0

    
      else:
        defensiveEnemyClosenessScore = 100000000.0 * (1.0/minDistanceHome)
        """
    """
    # if we're on our side it's good to be close to enemies
    if onMySide:
      # if were on the edge of our side and enemy right next to us BAD
      if myRealPos in self.mySideList and closestEnemyDistance == 1 and myPos == myRealPos:
        defensiveEnemyClosenessScore = -1000.0

      elif closestEnemyDistance == 0:
        defensiveEnemyClosenessScore = 1000.0

      else:
        defensiveEnemyClosenessScore = 100.0 * (1.0/closestEnemyDistance)

    # otherwise it's not good to be close to enemies
    else:
      if closestEnemyDistance == 0:
        defensiveEnemyClosenessScore = -1000.0

      else:
        defensiveEnemyClosenessScore = -100.0 * (1.0/closestEnemyDistance)

    #print "closestEnemyDistance: ", closestEnemyDistance
    #print "defensiveEnemyClosenessScore: ", defensiveEnemyClosenessScore
    """
    return defensiveEnemyClosenessScore

          


