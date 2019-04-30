# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        # currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor)
        # currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        distanceToGhost = manhattanDistance(newPos, newGhostStates[0].getPosition())

        if distanceToGhost > 0:
            score -= 10.0 / distanceToGhost
        if newFood.asList:
            distanceToFood = [manhattanDistance(newPos,x) for x in newFood.asList()]
        else: distanceToFood = [manhattanDistance(newPos, x) for x in newCapsules.asList()]
        if len(distanceToFood):
            score += 10.0 / min(distanceToFood)
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    # is the game over?
    def isTerminal(self, state, depth, agent):
        return depth == self.depth or \
               state.isWin() or \
               state.isLose() or \
               state.getLegalActions(agent) == 0

    def isPacman(self, state, agent):
        return agent % state.getNumAgents() == 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def findSuccessors(state, agent, depth):
            successors = (
                minimaxDecision(state.generateSuccessor(agent, action), depth, agent + 1)
                for action in state.getLegalActions(agent)
            )
            return successors

        def minimaxDecision(state, depth, agent):
            if agent == state.getNumAgents():  # pacman
                return minimaxDecision(state, depth + 1, 0)
            if self.isTerminal(state, depth, agent):
                return self.evaluationFunction(state)
            successors = findSuccessors(state, agent, depth)
            if self.isPacman(state, agent):
                return max(successors)
            else:
                return min(successors)
        return max(gameState.getLegalActions(0), key=lambda x: minimaxDecision(gameState.generateSuccessor(0, x), 0, 1))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """
        #  aima code on github looks very similar to the pseudocode in the textbook
        def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_search:
    best_score = -infinity
    beta = infinity
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

        if CUTOFF-TEST(s,d) maxa Actions(s) H-MINIMAX(RESULT(s,a),d+ 1) if PLAYER(s)=MAX
                            mina Actions(s) H-MINIMAX(RESULT(s,a),d+ 1) if PLAYER(s)=MIN
        """
        def AlphaBetaSearch(state, depth, agent, A=float("-inf"), B=float("inf")):
            #  alpha and beta initialized to respective infinity
            if self.isPacman(state, agent):
                return getValue(state, depth, agent, A, B, float('-inf'), max)
            else:
                return getValue(state, depth, agent, A, B, float('inf'), min)

        #  combined max-value and min-value function
        #  b/c very minimal difference between the two, I pass in negOrPos(infinity value) and the Max or Min function
        def getValue(state, depth, agent, A, B, negOrPos, maxOrMin):
            if agent == state.getNumAgents():
                agent = 0
                depth += 1

            if self.isTerminal(state, depth, agent):
                return self.evaluationFunction(state), None

            v = negOrPos
            theAction = None
            #  so actions don't need to be calculated twice
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                score, otherValue = AlphaBetaSearch(successor, depth, agent + 1, A, B)
                v, theAction = maxOrMin((v, theAction), (score, action))
                if self.isPacman(state, agent):
                    if v > B:
                        return v, theAction
                    A = maxOrMin(A, v)
                else:
                    if v < A:
                        return v, theAction
                    B = maxOrMin(B, v)

            return v, theAction
        otherValue, action = AlphaBetaSearch(gameState, 0, 0)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        """
         --- wikipedia
        function expectiminimax(node, depth)
    if node is a terminal node or depth = 0
        return the heuristic value of node
    if the adversary is to play at node
        // Return value of minimum-valued child node
        let A := infinity
        foreach child of node
            A := min(A, expectiminimax(child, depth-1))
    else if we are to play at node
        // Return value of maximum-valued child node
        let A := -infinity
        foreach child of node
            A := max(A, expectiminimax(child, depth-1))
    else if random event at node
        // Return weighted average of all child nodes' values
        let A := 0
        foreach child of node
            A := A + (Probability[child] * expectiminimax(child, depth-1))
    return A
        """
        def findSuccessors(state, depth, agent):
            successors = [ExpectimaxAgent(state.generateSuccessor(agent, action), depth, agent + 1)
                          for action in state.getLegalActions(agent)]
            return successors

        # variation of minimax
        def ExpectimaxAgent(state, depth, agent):
            if agent == state.getNumAgents():
                return ExpectimaxAgent(state, depth + 1, 0)  # when isPacman, go to next depth

            if self.isTerminal(state, depth, agent):
                return self.evaluationFunction(state)

            successors = findSuccessors(state, depth, agent)

            if self.isPacman(state, agent):  # do the best move
                return max(successors)
            else:  # find the average move to take
                return sum(successors) / len(successors)

        # give best move for Pacman
        return max(gameState.getLegalActions(0), key=lambda x: ExpectimaxAgent(gameState.generateSuccessor(0, x), 0, 1))



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      --

      I wanted to keep the parameters relatively simple as I did for the original
      evaluation function. So, I used the simply the scared ghost, distance to ghost, and distance to food as
      the factor of score.
      I made the incentive to chase after a scared ghost very high.
      Otherwise, keep going for food unless ghost is very close to you.
      I did not use any fancy equations - just the weight / distance.
      Summary:
      extremely aggressive towards the scared ghost.
      otherwise just go for the food unless the ghost is right next to you.
      using manhattan distance to find the location of food and ghost per state.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # currentFood = currentGameState.getFood() #food available from current state
    # newFood = successorGameState.getFood()  # food available from successor state (excludes food@successor)
    currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    newCapsules = currentGameState.getCapsules()  # capsules available from successor (excludes capsules@successor)
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()


    score = currentGameState.getScore()

    # ghost distance
    ghostValue = 0
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # kill the ghost!
                ghostValue += 110.0 / distance  # arbitrary extreme weight to scared ghost
            else:  # run if near
                ghostValue -= 0.3 / distance  # only run if super close
    score += ghostValue
    # food distance
    if newFood.asList:
        distanceToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
    else:
        distanceToFood = [manhattanDistance(newPos, x) for x in currentCapsules.asList()]
    if len(distanceToFood):
        score += 1.5 / min(distanceToFood)

    return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

