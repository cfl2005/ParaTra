# !/usr/bin/python
# -*- coding: utf-8 -*-


from collections import defaultdict

class DiGraph(object):

    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices


    def addEdge(self, u, v):
        if v is None or v == '':
            self.graph[u] = []
        else:
            self.graph[u].append(v)

    def toposort_helper(self, v, visited, stack):

        visited[v] = True
        for i in self.graph[v]:
            if visited[i] == False:
                self.toposort_helper(i, visited, stack)

        stack.insert(0, v)

    def recursion_toposort(self):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.toposort_helper(i, visited, stack)

        return stack

    def loop_toposort(self):

        in_degrees = dict((u, 0) for u in self.graph)
        vertext_num = len(in_degrees)

        for u in self.graph:
            for v in self.graph[u]:
                in_degrees[v] += 1

        Q = [u for u in in_degrees if in_degrees[u] == 0]

        Seq = []

        while Q:
            u = Q.pop()
            Seq.append(u)
            for v in self.graph[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    Q.append(v)
        if len(Seq) == vertext_num:
            return Seq

        else:
            print("Graph exists a cricle")

def main():

    """
    G = {
        'a': 'bce',
        'b': 'd',
        'c': 'd',
        'd': '',
        'e': 'cd'
    }
    """

    g = DiGraph(5)
    g.addEdge('a', 'b')
    g.addEdge('a', 'c')
    g.addEdge('a', 'e')
    g.addEdge('b', 'd')
    g.addEdge('c', 'd')
    g.addEdge('e', 'c')
    g.addEdge('e', 'd')
    g.addEdge('d', None)
    stack = g.loop_toposort()
    print(stack)

def toposort(graph):
    in_degrees = dict((u, 0) for u in graph)
    vertex_num = len(in_degrees)

    for u in graph:
        for v in graph[u]:
            in_degrees[v] += 1
    Q = [u for u in in_degrees if in_degrees[u] == 0]

    Seq = []

    while Q:
        u = Q.pop()
        Seq.append(u)
        for v in graph[u]:
            in_degrees[v] -= 1   
            if in_degrees[v] == 0:
                Q.append(v)

    if len(Seq) == vertex_num:
        return Seq
    else:
        print("exists a circle.")



def loop_toposort():
    G = {
        'a': 'bce',
        'b': 'd',
        'c': 'd',
        'd': '',
        'e': 'cd'
    }

    print(toposort(G))

if __name__ == "__main__":
    main()
    # loop_toposort()
