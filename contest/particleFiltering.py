######################
# Particle Filtering #
######################

class JointParticleFilter:
  """
  JointParticleFilter tracks a joint distribution over tuples of all ghost
  positions.
  """

  def __init__(self, numParticles=600):
      self.setNumParticles(numParticles)

  def setNumParticles(self, numParticles):
      self.numParticles = numParticles

  #def initialize(self, gameState, legalPositions):
  def initialize(self, gameState):
      "Stores information about the game, then initiaself.numGhosts = gameState.getNumAgents() - 1"

      #self.legalPositions = legalPositions
      self.initializeParticles(gameState)


  def initializeParticles(self, gameState):
      #This should only be called at the very beginning of the game because thats the only time
     
      #Initialize new array of particles
      self.particles = []
    
      #Iterate through the agents and find their starting postion
      startPos = []
      for opponentAgentIndex in self.opponentAgents:
        startPos.append(gameState.getInitialAgentPosition(self, opponentAgentIndex))

      #this turns the list of startPos into a tuple
      #!!!!not sure we need this!!!!
      startPos = tuple(startPos)

      #Each particle is of the format: [enemy agent 0 location, enemy agent 1 location]
      #Note which agents these are depend will vary basedd on which team we are
      for i in range(self.numParticles):
          self.particles.append(startPos)

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
        if pos[index] == None:
          unknownParticleIndices.append(index)

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
        for i in unknownParticleIndices:

          # find the true distance from pacman to the current ghost that we're iterating through
          trueDistance = util.manhattanDistance(p[i], myPos)

          # weight each particle by the probability of getting to that position (use emission model)
          # account for our current belief distribution (evidence) at this point in time
          listOfLocationWeights.append(gameState.getDistanceProb(trueDistance, noisyDistances[i]))
            
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

      #for i in range(self.numParticles): 
      for index, p in enumerate(self.particles):
          #particleDictionary[self.particles[i]] += particleWeights[i]
          particleDictionary[p] += particleWeights[index]


      particleDictionary.normalize() 
      
      if particleDictionary.totalCount() == 0:
         
          #self.initializeParticles()
          #I'm not sure it makes sense to reinitialize the particles here
          
      # otherwise, go ahead and resample based on our new beliefs 
      else:
          
          keys = []
          values = []

          # find each key, value pair in our counter
          keys, values = zip(*particleDictionary.items())
          
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
        ourPostDist = getOpponentDist(gameState, newParticle, opponentIndex)

        temp.append(util.sample(ourPosDist))
          

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

#HELPER METHODS FOR ELAPSE TIME


def getOpponentDist(gameState, particle, opponentIndex):

  #create a gamestate that corresponds to the particle 
  opponentPositionGameState = setOpponentPositions(gameState, particle)

  dist = util.Counter()

  opponentLegalActions = opponentPositionGameState.getLegalActions(opponentIndex)
  prob = 1/len(opponentLegalActions)

  for action in opponentLegalActions:
    successor = opponentPositionGameState.generateSuccessor(opponentIndex, action)
    pos = successor.getAgentState(opponentIndex).getPosition()
    dist[pos] = prob

  return dist

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


def setOpponentPositions(gameState, opponentPositions):
  "Sets the position of all opponent to the values in the particle"

  for index, pos in enumerate(opponentPositions):
      conf = game.Configuration(pos, game.Directions.STOP)
      gameState.data.agentStates[self.opponentAgents[index]] = game.AgentState(conf, False)
  return gameState

##########################
# END PARTICLE FILTERING #
##########################
