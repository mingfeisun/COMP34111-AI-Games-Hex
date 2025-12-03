class Node:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}      # Move -> Node
        self.visits = 0         # N
        self.value = 0          # W (total value)
        self.prior = prior      # P (probability from neural network)

    @property
    def value(self):
        return self.value / self.visits if self.visits != 0 else 0