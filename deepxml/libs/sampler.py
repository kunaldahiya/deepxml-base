from typing import List
from numpy import ndarray

import numpy as np
from xclib.utils.shortlist import Shortlist as _Shortlist
#from .clustering import ClusteringIndex as _ClusteringIndex
from xclib.utils.ann import ClusteringIndex as _ClusteringIndex


class ANNIndex(_Shortlist):
    """ANN Index suitable for:
    - when we just need explicit hard negatives
    - it is constantly updated (maintains a state)
    - Useful for explicit negative sampling
    """
    def __init__(
            self,
            num_items: int,
            update_steps: List[int]=[-1],
            method: str='hnswlib', 
            num_neighbours=100, 
            M: int=50, 
            efC: int=50, 
            num_threads: int=12, 
            space: str='cosine') -> None:
        """
        Args:
            num_items (int): number of items
            method (str, optional): ANN method or lib. Defaults to 'hnswlib'.
            num_neighbours (int, optional): fetch these many neighbors.
                Defaults to 100.
            M (int, optional): M in hnsw. Defaults to 50.
            efC (int, optional): construction parameter in hnsw. Defaults to 50.
            num_threads (int, optional): thread in train/inference. Defaults to 12.
            space (str, optional): Build in this space. Defaults to 'cosine'.
        """
        super().__init__(
            method=method,
            efS=num_neighbours,
            num_neighbours=num_neighbours,
            M=M,
            efC=efC,
            num_threads=num_threads,
            space=space)
        self.update_steps = update_steps
        self.ind, self.sim = None, None
        self.num_items = num_items
        self.n_step = 0

    def step(self, X_i: ndarray=None, X_o: ndarray=None) -> None:
        self._update_state()
        if self.n_step in self.update_steps:
            assert X_i is not None and X_o is not None, \
                "Input and output embeddings can't be none "\
                "when you want to update the negative sampler"
            self.update_index(X_i, X_o)

    def _init(self) -> None:
        """TODO
        """
        pass

    def update_state(self) -> None:
        self.n_step += 1

    def update(self, X_i: ndarray, X_o: ndarray) -> None:
        """Update the index based on given representations

        Args:
            X_i (ndarray): input item embeddings (search using these)
            X_o (ndarray): output item embeddings (search from these)
        """
        # TODO: Memory requirements for large datasets 
        # Keep indices and similarity as memmap file?
        self.fit(X_o)
        self.ind, self.sim = super().query(X_i)

    def query(self, index: int) -> tuple:
        return self.ind[index], self.sim[index]

    def __getitem__(self, index: int) -> tuple:
        # TODO: Fix it when the index is not initialized
        # Return random?
        return self.ind[index], self.sim[index]


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
            update_steps: List[int],
            curr_steps: List[int],
            space: str='cosine') -> None:
        self.num_items = num_items
        super().__init__(
            num_clusters,
            efS=-1,
            num_neighbours=-1,
            num_threads=num_threads,
            space=space)
        self.update_steps = update_steps
        self.curr_steps = curr_steps
        self.n_step = 0

    def step(self, X, *args, **kwargs):
        if self.n_step in self.update_steps and X is not None:
            self.update_index(X)
        self._update_state()

    def _update_state(self) -> None:
        """A state is defined by:
        - step: number of epochs 
        - num_clusters: cluster into these many documents

        The structure may be updated based on the state
        """
        if sum([i-1==self.n_step for i in self.curr_steps]) > 0:
            print(f"Doubling cluster size at: {self.n_step} to {2*self.num_items/self.num_clusters}")
            # larger clusters => harder negatives
            self.num_clusters //= 2
        self.n_step += 1

    def _init(self) -> None:
        """Each item is an cluster in itself
        """
        self.index = []
        for i in range(self.num_items):
           self.index.append([i])

    def update_index(self, X: ndarray, num_clusters: int=None) -> None:
        """Update the index based on given representations

        Args:
            X (ndarray): item embeddings
            num_clusters (int, optional): cluster into these many clusters.
              Defaults to None.
              - the num_clusters argument will override the stored value
        """
        assert self.num_items == len(X)
        print("Fitting the clusters!")
        self.fit(X, num_clusters)

    def __getitem__(self, idx: int) -> ndarray:
        """Get the item induces in idx-th cluster"""
        return self.index[idx]

    def query(self, idx: int) -> ndarray:
        """Get the item induces in idx-th cluster

        Args:
            idx (int): cluster index

        Returns:
            ndarray: indices of items in given cluster index
        """
        return self.index[idx]

    def indices_permutation(self, shuffle=True):
        c_indices = np.arange(len(self.index))
        if shuffle:
            np.random.shuffle(c_indices)
        return np.concatenate([self.index[i] for i in c_indices])

