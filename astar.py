from collections import *
from heapq import *
import pdb
from math import sqrt, pow

class AStar:
    def __init__(self, graph):
        self.graph = graph
        self.startNode = graph.getStart()
        self.bestNode = self.startNode
        self.goalNode = graph.getGoal()

        self.openList = deque()
        self.closed = set()

        self.newNode = self.startNode
        self.newNode.estimateDistance(self.goalNode)
        self.openNode(self.newNode)
        #self.pathLength = 1
        self.failed = False


    def extractMin(self):
        li = self.openList
        sortedlist = sorted(list(li), key=lambda x: x.f, reverse=True)
        n = sortedlist[ len(sortedlist) - 1 ]
        nodesLowesF = [ n ]
        tie_n = n.tieBreaking(self.goalNode)
        for node in sortedlist:
            if node.f == n.f:
                nodesLowesF.append(node)

        if len(nodesLowesF) == 1:
            self.openList.remove(n)
            return n

        for x in nodesLowesF:
            tie_x = x.tieBreaking(self.goalNode)
            if  tie_x  < tie_n:
                n = x
                tie_n = tie_x
        self.openList.remove(n)
        return n


    def openNode(self, node):
        self.openList.append(node)
        node.update('open')


    def closeNode(self, node):
        self.closed.add(node)
        node.update('closed')


    def isOpen(self, node):
        for n in self.openList:
            if n.getID() == node.getID():
                return True
        return False


    def isClosed(self, node):
        for n in self.closed:
            if n.getID() == node.getID() :
                return True
        return False


    def updatePath(self):
        for node in self.bestPath:
            node.update('path')


    def attachAndEval(self, child, parent):
        child.setParent(parent)
        child.estimateDistance(self.goalNode)


    def backtrackPath(self):
        self.bestPath = []
        pathNode = self.newNode
        #self.pathLength = 1

        while pathNode.getParent() != None:
            #self.pathLength += 1
            self.bestPath.append(pathNode)
            pathNode = pathNode.getParent()
        self.bestPath.append(pathNode)

        self.updatePath()
        return self.goalNode


    def betterPathFound(self, new, old):
        if new.getG() + 1 < old.getG():
            return True
        else:
            return False


    def iterateAStar(self):
        if self.newNode.state != 'goal':
            if len(self.openList) == 0:
                self.failed = True
                return self.newNode

            self.newNode = self.extractMin()
            self.closeNode(self.newNode)

            if self.newNode.h < self.bestNode.h:
                self.bestNode = self.newNode

            if self.newNode.getState() == 'goal':
                return self.backtrackPath()

            succ = self.graph.generateSucc(self.newNode)

            for s in succ:
                if self.isClosed(s):
                    if self.betterPathFound(self.newNode, s):
                        self.newNode.improvePath(s)
                elif self.isOpen(s):
                    if self.betterPathFound(self.newNode, s):
                        self.attachAndEval(s, self.newNode)
                else:
                    self.attachAndEval(s, self.newNode)
                    self.openNode(s)
                    #self.countNodes += 1
                self.newNode.addChild(s)
        # self.nofExpandedNodes += 1
        return self.newNode


# init(x,y)
class Node:

    def __init__(self, x, y):
        self.g = float('inf')
        self.f = float('inf')
        self.h = float('inf')
        self.parent = None #pointer to best parent node
        self.children = [] #list of succesors
        self.state = 'unvisited'
        self.x = x
        self.y = y


    def __repr__(self):
        return 'node(id=%s, fValue=%s, h=%s, g=%s state=%s)' %(self.getID(), self.f, self.h, self.g, self.state)


    def update(self, state):
        if self.state is not 'goal' and self.state is not 'start':
            self.state = state


    def getID(self):
        return ((self.x, self.y), self.state)


    def getF(self):
        self.f = self.g + self.h
        return self.f


    def getG(self):
        return self.g


    def setG(self, gValue):
        self.g = gValue
        self.f = self.g + self.h


    def getState(self):
        return self.state


    def setParent(self, parentNode):
        parentG = parentNode.getG()
        if self.g + self.cost(parentNode) > parentG:
            self.setG(parentG + self.cost(parentNode))
            self.parent = parentNode


    def getParent(self):
        return self.parent


    def getChildren(self):
        return self.children


    def addChild(self, node):
        self.children.append(node)


    def improvePath(self, node):
        cost = self.cost(node)
        for child in self.children:
            gNew = self.g + cost
            if gNew < child.g:
                child.setParent(self)
                # child.setG(gNew)
                child.improvePath(node)


    def tieBreaking(self, goal=None):
        return sqrt( pow((self.x - goal.x), 2) + pow((self.y - goal.y), 2) )


    def cost(self, node):
        nodeID = node.getID()
        if self.getID() == nodeID:
            return 0
        return 1


    def estimateDistance(self, goal):
        # Manhatan distance
        self.h = (abs(goal.x - self.x) + abs(goal.y - self.y))
        self.f = self.g + self.h


# init(width,height)
class Graph:
    def __init__(self, width, height, image):
        self.startNode = None
        self.goalNode = None
        # self.limit = 2000
        self.width = width
        self.height = height
        # self.celltype = ['start', 'goal', 'unvisited', 'closed', 'open', 'blocked', 'path']
        self.grid = []
        for x in range(width):
            self.grid.append([])
            for y in range(height):
                self.grid[x].append( Node(x,y) )
                if image[x,y,1] == 255:
                    self.grid[x][y].state = 'blocked'


    def update_cell(self, x, y, state):
        self.grid[x][y].update(state)
        if state=='start':
            self.startNode = self.grid[row][column]
            self.startNode.g = 0
        if state=='goal':
            self.goalNode = self.grid[row][column]


    def getStart(self):
        return self.startNode


    def getGoal(self):
        return self.goalNode


    def isGoal(self, node):
        print("isGoal should not be run")
        return


    def generateSucc(self, node):
        listToCheck = []
        ((x,y), s) = node.getID()
        directions = [[-1, 0], [0,-1], [1,0], [0,1]]
        for i in range(len(directions)):
            k = x + directions[i][0]
            l = y + directions[i][1]

            if  k < self.width and  l < self.height:
                neighbornode = self.grid[k][l]
                listToCheck.append( neighbornode )

        neighbors = []
        for neighbornode in listToCheck:
            if neighbornode and neighbornode.getState()!= 'blocked':
                if neighbornode.getG() > node.getG() + 1 :
                    neighbornode.setG( node.getG() + 1 )
                neighbors.append( neighbornode )
            # else:
            #     neighbornode.setG(100)
            #     neighbors.append( neighbornode )
        return neighbors
