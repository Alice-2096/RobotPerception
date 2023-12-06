# build a Node class to store images and actions as edges
class Node:
    def __init__(self, image, action):
        self.image = image
        self.action = action
        self.neighbors = []  # adj list
        self.visited = False
        self.parent = None  # for tracking the path

    def add_edge(self, node):
        self.neighbors.append(node)

    def get_edges(self):
        return self.edges

    def get_image(self):
        return self.image

    def get_action(self):
        return self.action

    def set_visited(self, visited):
        self.visited = visited

    def get_visited(self):
        return self.visited

    def set_parent(self, parent):
        self.parent = parent


class Graph:  # build a Graph class to store the node
    def __init__(self):
        self.nodes = []
        self.root = None
        self.current = None

    def add_node(self, node):
        self.nodes.append(node)

    def get_nodes(self):
        return self.nodes

    def set_root(self, root):
        self.root = root

    def get_root(self):
        return self.root

    def set_current(self, current):
        self.current = current

    def get_current(self):
        return self.current
