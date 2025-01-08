import math


def compute_depth_of_tree(n: int, s: int) -> int:
    """Get depth of tree 

    Args:
        n (int): Total number of items at root node 
        s (int): Cluster size at the leaf node 

    Returns:
        int: Depth of tree
    """
    return int(math.ceil(math.log(n / s) / math.log(2)))
