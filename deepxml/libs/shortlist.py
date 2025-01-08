from typing import List
from numpy import ndarray
from xclib.utils.ann import ClusteringIndex as _ClusteringIndex


class ClusteringIndex(_ClusteringIndex):
    """Clustering Index suitable for:
    - when we just need the items in a clusters while quering
    - it is constantly updated (maintains a state with step and num_clusters)
    - Useful for in-batch sampling or ngame sampling

    Args:
        num_instances (int): number of items to cluster
        num_clusters (int): create these many clusters
        num_threads (int): number of threads to use while clustering
        curr_steps (List[int]): increase complexity at these steps
    """
    def __init__(
            self,
            num_items: int,
            num_clusters: int,
            num_threads: int,
            curr_steps: List[int],
            space: str='cosine') -> None:
        self.num_items = num_items
        super().__init__(
            num_clusters,
            efS=-1,
            num_neighbours=-1,
            num_threads=num_threads,
            space=space)
        self.curr_steps = curr_steps
        self.step = 0

    def update_state(self) -> None:
        """A state is defined by:
        - step: number of epochs 
        - num_clusters: cluster into these many documents

        The structure may be updated based on the state
        """
        if sum([i-1==self.step for i in self.curr_steps]) > 0:
            print(f"Doubling cluster size at: {self.step} to {2*self.num_items/self.num_clusters}")
            # larger clusters => harder negatives
            self.num_clusters /= 2
        self.step += 1

    def _init(self) -> None:
        """Each item is an cluster in itself
        """
        self.index = []
        for i in range(self.num_items):
           self.index.append([i])

    def update(self, X: ndarray, num_clusters: int=None) -> None:
        """Update the index based on given representations

        Args:
            X (ndarray): item embeddings
            num_clusters (int, optional): cluster into these many clusters.
              Defaults to None.
              - the num_clusters argument will override the stored value
        """
        assert self.num_instances == len(X)
        self.fit(X, num_clusters)

    def query(self, idx: int) -> ndarray:
        """Get the item induces in idx-th cluster

        Args:
            idx (int): cluster index

        Returns:
            ndarray: indices of items in given cluster index
        """
        return self.index[idx]
