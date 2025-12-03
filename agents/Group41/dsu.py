class DSU:
    """Implements the Disjoint Set Union (Union-Find) data structure."""
    
    def __init__(self, size):
        # Initializes 'size' elements, each in its own set.
        self.parent = list(range(size))
        # Stores the height/rank of each tree for Union by Rank optimization.
        self.rank = [0] * size

    def find(self, x):
        # Finds the representative (root) of the set containing element x.
        while self.parent[x] != x:
            # Path Compression: Flattens the tree by making nodes point directly to their grandparent.
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        # Merges the sets containing elements a and b, if they are not already in the same set.
        a = self.find(a)
        b = self.find(b)
        
        if a == b:
            return
        
        # Union by Rank: Attaches the smaller tree (lower rank) to the root of the larger tree.
        if self.rank[a] < self.rank[b]:
            self.parent[a] = b
        elif self.rank[a] > self.rank[b]:
            self.parent[b] = a
        else:
            # If ranks are equal, attach one to the other and increase the new root's rank.
            self.parent[b] = a
            self.rank[a] += 1