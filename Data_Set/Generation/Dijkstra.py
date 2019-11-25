import math


class Dijkstra(object):

    def __init__(self, g, wt="weight", cap = "capacity"):
        self.dist = {}  # A map from nodes to their labels (float)
        self.predecessor = {}  # A map from a node to a node
        self.g = g
        self.wt = wt
        edges = g.edges()
        # Set the value for infinite distance in the graph
        self.inf = float('inf') #0.0  math.inf
        #print g
        """
        for e in edges:
            self.inf += abs(g[e[0]][e[1]][cap])
        self.inf += 1.0
        #print("Inf set to:",self.inf)
        """
    def getPath(self, source, dest, as_nodes=False):
        """
        Computes the shortest path in the graph between the given *source* and *dest*
        node (strings).  Returns the path as a list of links (default) or as a list of
        nodes by setting the *as_nodes* keyword argument to *True*.
        """
        self.dist = {}  # A map from nodes to their labels (float)
        self.predecessor = {}  # A map from a node to a node

        # Initialize the distance labels to "infinity"
        vertices = self.g.nodes()
        for vertex in vertices:
            self.dist[vertex] = self.inf
            self.predecessor[vertex] = source

        # Further set up the distance from the source to itself and
        # to all one hops away.
        self.dist[source] = 0.0
        if self.g.is_directed():
            outEdges = self.g.out_edges([source])
        else:
            outEdges = self.g.edges([source])

        for edge in outEdges:
            self.dist[edge[1]] = self.g[edge[0]][edge[1]][self.wt]
        s = set(vertices)
        s.remove(source)
        currentMin = self._findMinNode(s)
        #print("CurrentMin node",currentMin)
        if currentMin == None:
            return None
        s.remove(currentMin)
        while currentMin != dest and (len(s) != 0) and currentMin != None:
            if self.g.is_directed():
                outEdges = self.g.out_edges([currentMin])
            else:
                outEdges = self.g.edges([currentMin])
            for edge in outEdges:
                opposite = edge[1]
                if self.dist[currentMin] + self.g[edge[0]][edge[1]][self.wt] < self.dist[opposite]:
                    self.dist[opposite] = self.dist[currentMin] + self.g[edge[0]][edge[1]][self.wt]
                    self.predecessor[opposite] = currentMin
                    s.add(opposite)

            currentMin = self._findMinNode(s)
            # print "Current min node {}, s = {}".format(currentMin, s)
            if currentMin == None:
                return None
            s.remove(currentMin)

        # Compute the path as a list of edges
        currentNode = dest
        predNode = self.predecessor.get(dest)

        node_list = [dest]
        done = False
        path = []
        while not done:
            path.append((predNode, currentNode))
            currentNode = predNode
            predNode = self.predecessor[predNode]
            node_list.append(currentNode)
            done = currentNode == source
        node_list.reverse()
        if as_nodes:
            return node_list
        else:
            return path

    def _findMinNode(self, s):
        """
        Finds the vertex with the minimum distance label in the set "s".
        returns the minimum vertex
        """
        minNode = None
        minVal = self.inf
        for vertex in s:
            if self.dist[vertex] < minVal:
                minVal = self.dist[vertex]
                minNode = vertex
        #print("MinVal",minVal)
        return minNode



