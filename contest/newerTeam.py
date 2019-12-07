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
from baselineTeam import ReflexCaptureAgent
from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveQLearner', second='DefensiveQLearner'):
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


class EvalAgent(CaptureAgent):

    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing=.1)

        # all purpose fields used in all the subclasses below
        self.start = 1
        self.ghostStartPositions = []
        self.otherTeammate = 0
        self.ghostScaredTimer = [0, 0]
        self.opponents = {}

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

        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.ghostStartingPositions = [gameState.getAgentPosition(opp) for opp in self.getOpponents(gameState)]
        self.opponents = {opp: i for i, opp in enumerate(self.getOpponents(gameState))}
        #print self.ghostStartingPositions
        self.jailPath = self.computeJailDistance(gameState)

        # computes the index of the other "team member"
        self.generateOtherTeamMember(gameState)

    def generateOtherTeamMember(self, gameState):
        team = self.getTeam(gameState)
        other_member = [i for i in team if i != self.index]
        self.otherTeammate = other_member[0]

    def newLine(self):
        return '------------------------------------'

    ########################################
    ##  METHODS THAT COMPUTE SCARED TIMES ##
    ########################################

    def isScared(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer > 0

    def isTeammateScared(self, gameState):
        return gameState.getAgentState(self.otherTeammate).scaredTimer > 0

    def agentScaredTimeRemaining(self, gameState, agent=0):
        return gameState.getAgentState(agent).scaredTimer

    def teammateCaptured(self, gameState, agent=0):
        '''
        This method will only work on teammates, not on enemies.
        '''
        if len(self.observationHistory) < 2: return False
        prevState = self.observationHistory[-2]
        prevPos, curPos = prevState.getAgentPosition(agent), gameState.getAgentPosition(agent)
        if self.distancer.getDistance(prevPos, curPos) > 1:
            return True
        else:
            return False

    def computeJailDistance(self, gameState):
        pathFound = False
        startPos = self.ghostStartingPositions[1]
        agentIndex = self.getOpponents(gameState)[1]
        prevAction = Directions.STOP
        path = []
        while not pathFound:
            legalActions = gameState.getLegalActions(agentIndex)
            legalActions.remove('Stop')
            if prevAction != Directions.STOP:
                rev = Directions.REVERSE[prevAction]
                legalActions.remove(rev)
            if len(legalActions) == 1: # still in jail cell
                agentPos = gameState.getAgentPosition(agentIndex)
                gameState = gameState.generateSuccessor(agentIndex, legalActions[0])
                prevAction = legalActions[0]
                path.append((agentPos, prevAction))
            else: # out of jail cell - actions are no longer deterministic
                pathFound = True
        # print path
        return path

    def isPacman(self, gameState, id):
        midpoint = gameState.data.layout.width / 2

        estimatedPosition = gameState.getAgentPosition(id)
        if gameState.isOnRedTeam(self.index):
            if estimatedPosition[0] <= midpoint:
                return True
            else: return False
        else:
            if estimatedPosition[0] > midpoint:
                return True
            else:
                return False


###########################
###########################
#### PARTICLE FILTERING ###
###########################
###########################


class ParticleFiltering(EvalAgent):
    def __init__(self, index, timeForComputing=.1):
        EvalAgent.__init__(self, index, timeForComputing=0.1)

        # variables needed for particle filtering
        self.particleGhostOne, self.particleGhostTwo = [], []
        self.numParticles = 500
        self.posInPath = None
        self.legalPositions = []
        self.ghostAgents = []
        self.observedGhost1, self.observedGhost2 = None, None

        # tracks the ghost index for alpha beta later
        self.ghost = 0

    def registerInitialState(self, gameState):
        EvalAgent.registerInitialState(self, gameState)
        self.posInPath = self.ghostStartingPositions
        # self.particleList = self.initializeParticles(gameState)
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        # print self.legalPositions
        self.ghostAgents = self.getOpponents(gameState)
        self.observedPositions = self.ghostStartingPositions
        self.particleGhostOne = [self.ghostStartingPositions[0]] * self.numParticles
        self.particleGhostTwo = [self.ghostStartingPositions[1]] * self.numParticles


    def checkCaptured(self, gameState):
        if self.observedGhost1 is None and self.observedGhost2 is None: return [False, False]
        return [True if self.observedPositions[i] - gameState.getAgentPosition(self.ghostAgents[i]) > 1 else False for i in range(2)]

    def runParticleFiltering(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        print self.checkCaptured(gameState)
        # print [self.newLine() for i in range(5)]
        for (i, ghost) in enumerate(self.ghostAgents):
            # print ghost
            if enemies[i].configuration is None:
                # print 'elapsing time'
                self.observe(gameState, ghost)
                self.elapseTime(gameState, ghost)
            else:
                if self.opponents[ghost] == 0:
                    self.observedPositions[0] = gameState.getAgentPosition(ghost)
                    self.particleGhostOne = [enemies[i].getPosition() for p in range(self.numParticles)]
                else:
                    self.observedPositions[1] = gameState.getAgentPosition(ghost)
                    self.particleGhostTwo = [enemies[i].getPosition() for p in range(self.numParticles)]
        '''
        if withinFive:
            captured = [self.checkGhostCaptured(gameState, ghostIndex) for ghostIndex in self.getOpponents(gameState) \
                        if gameState.getAgentState(ghostIndex).isPacman]
            for (i, capture) in enumerate(captured):
                if capture:
                    for j in range(10):
                        print self.newLine()
                    if i == 0:
                        self.particleGhostOne = [self.ghostStartingPositions[opponents[0]]] * self.numParticles
                    else:
                        self.particleGhostTwo = [self.ghostStartingPositions[opponents[1]]] * self.numParticles
        '''

    def updateGameState(self, gameState):
        opponents = self.getOpponents(gameState)
        curPosition = gameState.getAgentPosition(self.index)
        beliefDistributionOne, beliefDistributionTwo = self.getBeliefDistribution(
            opponents[0]), self.getBeliefDistribution(opponents[1])
        self.displayDistributionsOverPositions([beliefDistributionOne, beliefDistributionTwo])
        # print beliefDistributionTwo, beliefDistributionOne
        bestvals1, bestvals2 = beliefDistributionOne.sortedKeys(), beliefDistributionTwo.sortedKeys()
        # print bestvals1[0], bestvals2[0]
        # print bestvals1, bestvals2
        # print bestvals1, bestvals2
        dist_ghost1, dist_ghost2 = self.distancer.getDistance(curPosition, bestvals1[0]), self.distancer.getDistance(
            curPosition, bestvals2[0])
        newState = self.setGhostPosition(gameState, opponents[0], bestvals1[0])
        newState = self.setGhostPosition(newState, opponents[1], bestvals2[0])
        if dist_ghost1 < dist_ghost2:  # highest prob position is for ghost 1
            self.ghost = opponents[0]
            newState = self.setGhostPosition(gameState, opponents[0], bestvals1[0])
        else:
            self.ghost = opponents[1]
            newState = self.setGhostPosition(gameState, opponents[1], bestvals2[0])
        return newState

    def checkGhostCaptured(self, gameState, ghostIndex):
        if len(self.observationHistory) < 2: return False
        prevState = self.observationHistory[-2]
        ghostPrevPos, ghostCurPos = prevState.getAgentPosition(ghostIndex), gameState.getAgentPosition(ghostIndex)
        pacmanPrevPos, pacmanCurPos = prevState.getAgentPosition(self.index), gameState.getAgentPosition(
            self.index)
        prevDistance = self.distancer.getDistance(ghostPrevPos, pacmanPrevPos)
        curDistance = self.distancer.getDistance(ghostCurPos, pacmanCurPos)
        if prevDistance == 1 and curDistance > 2:
            return True
        else:
            return False

    def onSide(self, gameState, index, pos):
        midpoint = gameState.data.layout.width / 2
        if gameState.isOnRedTeam(index):
            if pos[0] <= midpoint: return True
            else: return False
        else:
            if pos[0] > midpoint: return True
            else: return False

    def setGhostPosition(self, gameState, ghostIndex, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)  # creates anticipated position for ghost
        if self.onSide(gameState, ghostIndex, ghostPosition):
            gameState.data.agentStates[ghostIndex] = game.AgentState(conf, False)  # creates agentState object for the enemy
        else:
            gameState.data.agentStates[ghostIndex] = game.AgentState(conf, True)
        # print 'after update in setPosition:', gameState
        return gameState

    def initializeUniformly(self, gameState, ghost, posList=None):
        """
        Initializes a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        particles = []
        num = self.numParticles
        if posList is None:
            positions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        else:
            positions = posList
        counter = 0
        # assuming the number of particles is always larger than the number of states
        while (counter < num):
            for p in positions:
                if (counter < num):
                    particles.append(p)
                counter = counter + 1
        if self.opponents[ghost] == 0:  # this is the smaller ghost
            self.particleGhostOne = particles  # enables access to other methods that don't have gameState attribute
        else:
            self.particleGhostTwo = particles  # enables access to other methods that don't have gameState attribute
        # print 'particles initialized', self.particleGhostOne, self.particleGhostTwo

    def observe(self, gameState, ghost):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.getJailPosition()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeUniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattanDistance to calculate the
        distance between a particle and Pacman's position.
        """
        pacmanPosition = gameState.getAgentPosition(self.index)
        noisyDistance = gameState.getAgentDistances()[ghost]
        if self.opponents[ghost] == 0:  # we're dealing with the first ghost
            weightDictionary = util.Counter()  # local weights dictionary
            curDistribution = self.getBeliefDistribution(ghost)
            for p in self.particleGhostOne:  # create empirical weights distribution
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                weightDictionary[p] = gameState.getDistanceProb(trueDistance, noisyDistance) * curDistribution[p]
            if weightDictionary.totalCount() == 0:  # check for all weights == 0 after reweighting distribution
                self.initializeUniformly(gameState, ghost, posList=weightDictionary.keys())
            else:  # sample keys (positions) from our empirical distribution
                keys, values = [], []
                for key, value in weightDictionary.items():
                    keys.append(key), values.append(value)
                resampledDistribution = util.nSample(values, keys, self.numParticles)
                self.particleGhostOne = resampledDistribution
        else:
            #print 'second ghost'
            weightDictionary = util.Counter()  # local weights dictionary
            curDistribution = self.getBeliefDistribution(ghost)
            for p in self.particleGhostTwo:  # create empirical weights distribution
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                weightDictionary[p] = gameState.getDistanceProb(trueDistance, noisyDistance) * curDistribution[p]
            if weightDictionary.totalCount() == 0:  # check for all weights == 0 after reweighting distribution
                self.initializeUniformly(gameState, ghost, posList=weightDictionary.keys())
            else:  # sample keys (positions) from our empirical distribution
                keys, values = [], []
                for key, value in weightDictionary.items():
                    keys.append(key), values.append(value)
                resampledDistribution = util.nSample(values, keys, self.numParticles)
                self.particleGhostTwo = resampledDistribution

    def elapseTime(self, gameState, ghost):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        temp = []
        # print 'in elapseTime', gameState
        if self.opponents[ghost] == 0:
            # print 'updating first ghost'
            for oldPos in self.particleGhostOne:
                # for every particle in list, create a new distribution, sample uniformly, append to temp list
                newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, ghost, oldPos), ghost)
                # print oldPos, newPosDist
                newPos = util.sample(newPosDist)
                temp.append(newPos)
            self.particleGhostOne = temp
        else:
            # print 'updating second ghost'
            for oldPos in self.particleGhostTwo:
                # for every particle in list, create a new distribution, sample uniformly, append to temp list
                newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, ghost, oldPos), ghost)
                newPos = util.sample(newPosDist)
                temp.append(newPos)
            self.particleGhostTwo = temp

    def getBeliefDistribution(self, ghost):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        # take particle in list and weight by frequency, then normalize
        beliefDistribution = util.Counter()
        if self.opponents[ghost] == 0:
            for particle in self.particleGhostOne:
                beliefDistribution[particle] += 1
        else:
            for particle in self.particleGhostTwo:
                beliefDistribution[particle] += 1
        beliefDistribution.normalize()
        return beliefDistribution

    def getPositionDistribution(self, gameState, index):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getAgentPosition(index)
        actionDist = self.getDistribution(gameState, index)
        dist = util.Counter()
        try:
            for action, prob in actionDist.items():
                successorPosition = game.Actions.getSuccessor(ghostPosition, action)
                dist[successorPosition] = prob
            return dist
        except KeyError:
            return actionDist

    def getDistribution(self, state, ghost):
        # Read variables from state
        from util import manhattanDistance
        distribution = util.Counter()
        jailCheck = [item for item in self.jailPath if state.getAgentPosition(ghost) in item]
        if jailCheck != []:
            # we think hes in the jail area
            item = jailCheck[0]
            successorPos = Actions.getSuccessor(item[0], item[1])
            distribution[successorPos] = 1.0
        else:
            # computes action that will bring agent closest to pacman
            distanceToInvaders= [(self.distancer.getDistance(state.getAgentPosition(self.index), Actions.getSuccessor(state.getAgentPosition(ghost), action)), action) for action in state.getLegalActions(ghost)]
            # teammateDistToInvaders = [(self.distancer.getDistance(state.getAgentPosition(self.otherTeammate), Actions.getSuccessor(state.getAgentPosition(ghost), action)), action) for action in state.getLegalActions(ghost)]
            # if not state.getAgentState(ghost).isPacman:
            #     bestTeammate, bestIndex = min(teammateDistToInvaders), min(distanceToInvaders)
            #     # print bestTeammate, bestTeammate
            #     best_action = min(bestIndex, bestTeammate)[1]
            # else:
            #     bestTeammate, bestIndex = max(teammateDistToInvaders), max(distanceToInvaders)
            #     best_action = max(bestIndex, bestTeammate)[1]
            best_action = min(distanceToInvaders)[1]
            bestProb = 0.7
            # print best_action
            distribution[best_action] = bestProb
            legalActions = state.getLegalActions(ghost)
            for a in legalActions: distribution[a] += (1 - bestProb) / len(legalActions)

        # print distribution
        distribution.normalize()
        return distribution

    def eatCapsule(self, gameState, curPos, capsules):
        if curPos[0] in capsules or curPos[1] in capsules:
            return True
        else:
            return False

    def scaredTime(self):
        return [max(self.ghostScaredTimer[i] - 1, 0) for i in range(2)]

    def checkScaredTimer(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)

        if gameState.isOnRedTeam(self.index):
            capsules = gameState.getBlueCapsules()
            if capsules == [] and len(self.observationHistory) > 1 and self.observationHistory[
                -2].getBlueCapsules() != []:
                capsules = self.observationHistory[-2].getBlueCapsules()
        else:
            capsules = gameState.getRedCapsules()
            if capsules == [] and len(self.observationHistory) > 1 and self.observationHistory[
                -2].getRedCapsules() != []:
                capsules = self.observationHistory[-2].getRedCapsules()
        if self.eatCapsule(successor,
                           [successor.getAgentPosition(self.index), successor.getAgentPosition(self.otherTeammate)],
                           capsules): self.ghostScaredTimer = [41, 41]
        self.ghostScaredTimer = self.scaredTime()
        # print 'scaredTime:', self.ghostScaredTimer
        return successor


################################
### ALPHA BETA PRUNING #########
################################


class AlphaBetaAgent(ParticleFiltering):
    def __init__(self, index, timeForComputing=.1):
        ParticleFiltering.__init__(self, index, timeForComputing=.1)

        self.prevState = None

    def max_value(self, state, alpha, beta, agents=0, isTeammate=False, cur_depth=2):
        if cur_depth <= 0 or state.isOver(): return self.betterEvaluationFunction(state, agents), Directions.STOP
        value = (-float('inf'), Directions.STOP)
        # print agents, state.getLegalActions(agents)
        self.prevState = state
        for action in state.getLegalActions(agents):
            if isTeammate:  # end of ply, circling back to start agent
                try:
                    nextState = state.generateSuccessor(agents, action)
                    min_value = self.max_value(nextState, alpha, beta, agents=self.index, cur_depth=cur_depth - 1)
                except Exception:
                    continue
            else:  # start of ply, go to first ghost
                try:
                    nextState = state.generateSuccessor(agents, action)
                    min_value = self.min_value(nextState, alpha, beta, agents=self.ghost, cur_depth=cur_depth)
                except Exception:
                    continue
            # print min_value
            value = max(value, (min_value[0], action))
            if value[0] > beta: return value[0], value[1]
            alpha = max(alpha, value[0])
        # print 'about to return', value[0], value[1]
        return value[0], value[1]

    def min_value(self, state, alpha, beta, agents=0, cur_depth=2):
        if cur_depth <= 0 or state.isOver(): return self.betterEvaluationFunction(state, agents), Directions.STOP
        value = (float('inf'), Directions.STOP)
        self.prevState = state
        for action in state.getLegalActions(agents):  # at ghost position, next value is the other teammate
            try:
                nextState = state.generateSuccessor(agents, action)
                value = min(value,
                            (self.max_value(nextState, alpha, beta, self.otherTeammate, True, cur_depth)[0], action))
            except Exception:
                continue
            if value[0] < alpha: return value[0], value[1]
            beta = min(beta, value[0])
        return value[0], value[1]

    def betterEvaluationFunction(self, gameState, agents):
        """
        Our evaluation function that uses Q-learning to find the best weights for its features
        """
        # gets action that we took to get to this gameState
        # print agents, gameState.isOnRedTeam(agents) == gameState.isOnRedTeam(self.index)
        if self.prevState is None:
            action = Directions.STOP
        else:  # alpha beta passes in the next game state but the observation history doesnt consider the previous gameState
            prevPos = self.prevState.getAgentPosition(agents)
            curPos = gameState.getAgentPosition(agents)
            from game import Actions
            action = Actions.vectorToDirection((prevPos[0] - curPos[0], prevPos[1] - curPos[1]))
        return self.evaluate(gameState, action)

    def evaluate(self, gameState, action):
        return 0


#######################
## GENERAL Q LEARNER ##
#######################

class QLearner(AlphaBetaAgent):
    def __init__(self, index, timeForComputing=.1):
        AlphaBetaAgent.__init__(self, index, timeForComputing=.1)

        # general q-learning instance variables
        self.alpha, self.discount = 0.3, 0.9
        self.weights = util.Counter()

    def getFeatures(self, gameState, action):
        '''
        This method is meant to be overwritten
        '''
        return [None]

    def evaluate(self, gameState, action):
        #if self.getFeatures(gameState, action)['successorScore'] > 1: print self.getFeatures(gameState, action) * self.weights
        return self.getFeatures(gameState, action) * self.weights

    def initializeWeights(self, feature_list=None):
        '''
        Converts our dict of baseline weights to a counter to be used
        :return:
        '''
        #print feature_list
        # all initial weights need to be <1 for this to work
        for feature in feature_list:
            self.weights[feature] = 1.0

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        val = 0
        for feature in self.getFeatures(state, action):
            feature_weight = self.getFeatures(state, action)[feature]
            weight = self.weights[feature]
            val += feature_weight * weight
        return val

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = state.getLegalActions(self.index)
        if len(legal_actions) == 0: return 0.0
        best = -float('inf')
        for action in legal_actions:
            temp = self.getQValue(state, action)
            if temp > best: best = temp
        # print 'bestQValue', best
        return best

    def getReward(self, gameState):
        curFood = self.getFood(gameState).asList()
        if curFood == 0: return 400  # a greater reward than anything else!
        return -len(curFood) * 5

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        featureDict = self.getFeatures(state, action)
        difference = (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))
        for feature in featureDict:
            feature_weight = featureDict[feature]
            self.weights[feature] += self.alpha * difference * feature_weight
        print self.index, self.weights


#####################
## SPECIFIC AGENTS ##
#####################


############################################
#### AGENT 1: OFFENSIVE LEARNER ############
############################################


class OffensiveQLearner(QLearner):
    def __init__(self, index, timeForComputing=.1):
        QLearner.__init__(self, index, timeForComputing=.1)

        # defines offensive q learning specific features
        self.weights = util.Counter()
        print 'test'
        self.initializeWeights(
            feature_list=['distanceToFood', 'distanceBack', 'numActions', 'successorScore', 'scaredDistance1', 'numCapsules',
                          'closestCapsule', 'stop', 'reverse'])

    def getFeatures(self, gameState, action):
        '''
                A function that returns the features that will be used to compute an optimal balance of an offensive and defensive agent
                :param gameState:
                :param action:
                :return: features that correspond to the
                '''
        # print gameState
        features = util.Counter()

        # incorporates bias
        # features['bias'] = 1.0

        # print action, gameState.getAgentPosition(self.index)
        # if action not in gameState.getLegalActions(self.index): return features
        # print agentIndex, action, gameState.getLegalActions(agentIndex)
        successor = gameState.generateSuccessor(self.index, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        myState = successor.getAgentState(self.index)
        numCarrying = 0
        if myState.isPacman:
            if (myState.numCarrying > 0):
                numCarrying = myState.numCarrying

        features['successorScore'] = -len(foodList) * 100

        # prevents distances from diverging
        # normalizing_constant = float(gameState.data.layout.height * gameState.data.layout.width)

        # compute distance to middle
        # add feature has food
        # create list of all states on our side of the board
        if gameState.isOnRedTeam(self.index):
            ourSide = [(float(gameState.data.layout.width / 2), float(i)) for i in
                       range(gameState.data.layout.height) if
                       not gameState.hasWall(gameState.data.layout.width / 2, i)]
        else:
            ourSide = [(float(gameState.data.layout.width / 2 + 1), float(i)) for i in
                       range(gameState.data.layout.height) if
                       not gameState.hasWall(gameState.data.layout.width / 2 + 1, i)]

        minWayBack = min([self.getMazeDistance(myPos, spot) for spot in ourSide])
        # if myState.isPacman: print myState.isPacman
        if (myState.isPacman):
            # print 'finding distanceBack'
            # if numCarrying == 0: features['distanceBack'] = -minWayBack * 2
            features['distanceBack'] = -minWayBack * numCarrying * 5  # / normalizing_constant
        else:
            features['toCenter'] = -minWayBack  # /normalizing_constant

            # we want to slightly incentivize getting capsules
            if gameState.isOnRedTeam(self.index):
                capsule_dist = [self.distancer.getDistance(myPos, capsule) for capsule in successor.getBlueCapsules()]
            else:
                capsule_dist = [self.distancer.getDistance(myPos, capsule) for capsule in successor.getRedCapsules()]
            features['numCapsules'] = -len(capsule_dist)  # should be negative --> less is more
            if len(capsule_dist) > 0:
                features['closestCapsule'] = -min(capsule_dist) * 50
                if min(capsule_dist) == 0: features['closestCapsule'] = 1500
            else:
                features['closestCapsule'] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # isScared = [a.scaredTimer > 0 for a in enemies]

        # need method that determines enemy state scared times
        # if any([self.ghostScaredTimer[i] > 0 for i in range(len(self.ghostScaredTimer))]):
        # dists = [self.distancer.getDistance(myPos, a.getPosition()) for i, a in enumerate(enemies)]
        # features['scaredDistance2'] = min(dists)
        # if min(dists) == 1:
        #     features['scaredDistance2'] = 100
        # elif min(dists) == 0:
        #     features['scaredDistance2'] = 200

        # computes distances to invaders that we can see

        invaders = [gameState.getAgentPosition(i) for i in self.ghostAgents if self.isPacman(gameState, i)]
        if len(invaders) > 0:
            dists = [self.distancer.getDistance(myPos, a) for a in invaders]
            features['invaderDistance'] = -min(dists) * 10
            if min(dists) == 1:
                features['invaderDistance'] = -100
            elif min(dists) == 0:
                features['invaderDistance'] = -200

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = -minDistance * 6  # / normalizing_constant
            if minDistance == 1:
                features['distanceToFood'] = 250
            elif minDistance == 0:
                features['distanceToFood'] = 500

        if action == Directions.STOP: features['stop'] = -100
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        features.divideAll(100000.0)
        # print features

        return features

    def chooseAction(self, gameState):
        start_time = time.time()
        self.runParticleFiltering(gameState)
        newState = self.updateGameState(gameState)
        best_action = self.max_value(newState, -float('inf'), float('inf'), agents=self.index)
        successor = self.checkScaredTimer(gameState, best_action[1])
        self.update(newState, best_action[1], successor, self.getReward(newState))
        print("--- %s seconds ---" % (time.time() - start_time))
        return best_action[1]


############################################
#### AGENT 2: DEFENSIVE LEARNER ############
############################################


class DefensiveQLearner(QLearner):
    def __init__(self, index, timeForComputing=.1):
        QLearner.__init__(self, index, timeForComputing=.1)

        # defines offensive q learning specific features
        self.weights = util.Counter()
        self.initializeWeights(
            feature_list=['invaderDistance', 'toCenter', 'numInvaders', 'scaredDistance2', 'distanceToFood', 'numActions', 'stop', 'reverse', 'onDefense', 'closestCapsule'])


    def getReward(self, gameState):
        '''
        # Computes reward for a defensive agent: get as close to a Pacman as possible
        # :param gameState:
        # :return:
        '''
        print "in getReward"
        print " "
        print " "
        print " "
        print " "
        print " "
        print " "
        curPos = gameState.getAgentPosition(self.index)
        updatedPositions = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if self.isPacman(gameState, i)]
        #updatedPositions = [e.getPosition() for e in enemies if e.isPacman]

        print updatedPositions
        print "**********************************"
        if len(updatedPositions) == 0: # no pacman in sight
            if gameState.isOnRedTeam(self.index):
                ourSide = [(float(gameState.data.layout.width / 2), float(i)) for i in
                           range(gameState.data.layout.height) if
                           not gameState.hasWall(gameState.data.layout.width / 2, i)]
            else:
                ourSide = [(float(gameState.data.layout.width / 2 + 1), float(i)) for i in
                           range(gameState.data.layout.height) if
                           not gameState.hasWall(gameState.data.layout.width / 2 + 1, i)]
            minWayBack = min([self.getMazeDistance(curPos, spot) for spot in ourSide])
            return -2 * minWayBack
        dists = [self.distancer.getDistance(curPos, pos) for pos in updatedPositions]
        bestReward = -min(dists)
        #print "DISTANCES:", dists
        if 1 < abs(bestReward) < 2:
            print"EHH"
            bestReward = 10000
        elif (abs(bestReward) <= 1):
            print "AHHH"
            bestReward = 20000
        multiple = -len(updatedPositions) * 20
        print multiple
        return bestReward + multiple


    def getFeatures(self, gameState, action):
        '''
                A function that returns the features that will be used to compute an optimal balance of an offensive and defensive agent
                :param gameState:
                :param action:
                :return: features that correspond to the
                '''
        # print gameState
        features = util.Counter()

        # incorporates bias
        # features['bias'] = 1.0

        # print action, gameState.getAgentPosition(self.index)
        # if action not in gameState.getLegalActions(self.index): return features
        # print agentIndex, action, gameState.getLegalActions(agentIndex)
        successor = gameState.generateSuccessor(self.index, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        myState = successor.getAgentState(self.index)
        numCarrying = 0
        if myState.isPacman:
            if (myState.numCarrying > 0):
                numCarrying = myState.numCarrying

        features['successorScore'] = -len(foodList) * 100

        # prevents distances from diverging
        # normalizing_constant = float(gameState.data.layout.height * gameState.data.layout.width)

        # compute distance to middle
        # add feature has food
        # create list of all states on our side of the board
        if gameState.isOnRedTeam(self.index):
            ourSide = [(float(gameState.data.layout.width / 2), float(i)) for i in
                       range(gameState.data.layout.height) if
                       not gameState.hasWall(gameState.data.layout.width / 2, i)]
        else:
            ourSide = [(float(gameState.data.layout.width / 2 + 1), float(i)) for i in
                       range(gameState.data.layout.height) if
                       not gameState.hasWall(gameState.data.layout.width / 2 + 1, i)]

        minWayBack = min([self.getMazeDistance(myPos, spot) for spot in ourSide])
        # if myState.isPacman: print myState.isPacman
        if (myState.isPacman):
            # print 'finding distanceBack'
            # if numCarrying == 0: features['distanceBack'] = -minWayBack * 2
            features['distanceBack'] = -minWayBack * numCarrying * 5  # / normalizing_constant
        else:
            features['toCenter'] = -minWayBack  # /normalizing_constant

            # we want to slightly incentivize getting capsules
            if gameState.isOnRedTeam(self.index):
                capsule_dist = [self.distancer.getDistance(myPos, capsule) for capsule in successor.getBlueCapsules()]
            else:
                capsule_dist = [self.distancer.getDistance(myPos, capsule) for capsule in successor.getRedCapsules()]
            features['numCapsules'] = -len(capsule_dist)  # should be negative --> less is more
            if len(capsule_dist) > 0:
                features['closestCapsule'] = -min(capsule_dist) * 12
                if min(capsule_dist) == 0: features['closestCapsule'] = 1500
            else:
                features['closestCapsule'] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # isScared = [a.scaredTimer > 0 for a in enemies]

        invaders = [i for i in self.ghostAgents if self.isPacman(gameState, i)]
        features['numInvaders'] = -len(invaders) * 5000
        if (len(invaders) == 0):
            features['numInvaders'] = 2500

        # need method that determines enemy state scared times
        # if any([self.ghostScaredTimer[i] > 0 for i in range(len(self.ghostScaredTimer))]):
        # dists = [self.distancer.getDistance(myPos, a.getPosition()) for i, a in enumerate(enemies)]
        # features['scaredDistance2'] = min(dists)
        # if min(dists) == 1:
        #     features['scaredDistance2'] = 100
        # elif min(dists) == 0:
        #     features['scaredDistance2'] = 200

        # computes distances to invaders that we can see

        # print action, gameState
        curPos = gameState.getAgentPosition(self.index)
        updatedPositions = [(successor.getAgentPosition(i), self.isPacman(gameState, i)) for i in self.ghostAgents]
        dists = [self.distancer.getDistance(curPos, pos[0]) for pos in updatedPositions if pos[1]]
        distsOther = [self.distancer.getDistance(curPos, pos[0]) for pos in updatedPositions if not pos[1]]
        if len(dists) == 0:
            features['invaderDistance'] = -min(distsOther) * 5
        else:
            features['invaderDistance'] = -min(dists) * 100
            if min(dists) == 1:
                features['invaderDistance'] = 1000
                # print action, 'almost eating!! '  # / normalizing_constant
            if min(dists) == 0:
                # print action, 'nom! nom! nom! eat! ghost!'
                features['invaderDistance'] = 10000

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = -minDistance * 6  # / normalizing_constant
            if minDistance == 1:
                features['distanceToFood'] = 250
            elif minDistance == 0:
                features['distanceToFood'] = 500

        if action == Directions.STOP: features['stop'] = -100
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        features.divideAll(100000.0)
        # print features

        return features

    def chooseAction(self, gameState):
        start_time = time.time()
        self.runParticleFiltering(gameState)
        newState = self.updateGameState(gameState)
        best_action = self.max_value(newState, -float('inf'), float('inf'), agents=self.index)
        successor = self.checkScaredTimer(gameState, best_action[1])
        self.update(newState, best_action[1], successor, self.getReward(newState))
        print("--- %s seconds ---" % (time.time() - start_time))
        return best_action[1]

    '''
        A function that returns the features that will be used to compute an optimal balance of an offensive and defensive agent
        :param gameState:
        :param action:
        :return: features that correspond to the
        # print gameState
        features = util.Counter()

        # incorporates bias
        # features['bias'] = 1.0

        # print action, gameState.getAgentPosition(self.index)
        # if action not in gameState.getLegalActions(self.index): return features
        # print agentIndex, action, gameState.getLegalActions(agentIndex)
        successor = gameState.generateSuccessor(self.index, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        myState = successor.getAgentState(self.index)

        # computes successor score --> score for defensive agent based on distance to opponent

        # prevents distances from diverging
        # normalizing_constant = float(gameState.data.layout.height * gameState.data.layout.width)

        # compute distance to middle
        # add feature has food
        # create list of all states on our side of the board
        if gameState.isOnRedTeam(self.index):
            ourSide = [(float(gameState.data.layout.width / 2), float(i)) for i in
                       range(gameState.data.layout.height) if
                       not gameState.hasWall(gameState.data.layout.width / 2, i)]
        else:
            ourSide = [(float(gameState.data.layout.width / 2 + 1), float(i)) for i in
                       range(gameState.data.layout.height) if
                       not gameState.hasWall(gameState.data.layout.width / 2 + 1, i)]

        minWayBack = min([self.getMazeDistance(myPos, spot) for spot in ourSide])
        features['toCenter'] = -minWayBack

        # highly value getting closer to the ghost
        curPos = gameState.getAgentPosition(self.index)
        updatedPositions = [(successor.getAgentPosition(i), self.isPacman(gameState, i)) for i in self.ghostAgents]
        dists = [self.distancer.getDistance(curPos, pos[0]) for pos in updatedPositions if pos[1]]
        distsOther = [self.distancer.getDistance(curPos, pos[0]) for pos in updatedPositions if not pos[1]]
        if len(dists) == 0: features['successorScore'] = -min(distsOther) * 5
        else:
            features['successorScore'] = -min(dists) * 100
            if min(dists) == 1:
                features['successorScore'] = 1000
                #print action, 'almost eating!! '  # / normalizing_constant
            if min(dists) == 0:
                #print action, 'nom! nom! nom! eat! ghost!'
                features['successorScore'] = 10000
        # isScared = [a.scaredTimer > 0 for a in enemies]

        # need method that determines enemy state scared times

        # if any([self.ghostScaredTimer[i] > 0 for i in range(len(self.ghostScaredTimer))]):
        #     dists = [self.distancer.getDistance(myPos, a.getPosition()) for i, a in enumerate(enemies)]
        #     features['scaredDistance2'] = -min(dists)

        # computes distances to invaders that we can see
        invaders = [i for i in self.ghostAgents if self.isPacman(gameState, i)]
        features['numInvaders'] = -len(invaders) * 4000
        if (len(invaders) == 0):
            features['numInvaders'] = 2000

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = -minDistance * 2 # / normalizing_constant
            if minDistance == 1:
                features['distanceToFood'] = 25
            elif minDistance == 0:
                features['distanceToFood'] = 50
        #
        # # # incentivizes states that have more potential actions
        # # legalActions = successor.getLegalActions(self.index)
        # # features['numActions'] = len(legalActions) * 5
        #
        # # we want to slightly incentivize getting capsules
        # if gameState.isOnRedTeam(self.index):
        #     capsule_dist = [self.distancer.getDistance(myPos, capsule) for capsule in
        #                     successor.getBlueCapsules()]
        # else:
        #     capsule_dist = [self.distancer.getDistance(myPos, capsule) for capsule in
        #                     successor.getRedCapsules()]
        # features['numCapsules'] = -len(capsule_dist)  # should be negative --> less is more
        # if len(capsule_dist) > 0:
        #     features['closestCapsule'] = -min(capsule_dist) * 20
        #     if min(capsule_dist) == 0: features['closestCapsule'] = 1500
        # else:
        #     features['closestCapsule'] = 0

        if action == Directions.STOP: features['stop'] = -100
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        if myState.isPacman: features['onDefense'] = -1500
        else: features['onDefense'] = 100

        features.divideAll(10000.0)

        return features
    '''
