import numpy as np


class BaseNode:
    def render(self, num_samples: int, **kwargs) -> np.ndarray:
        """
        Renders the node for a given number of samples.
        Any kwargs passed to this function should be forwarded to the node's children,
        The node can use consume_kwargs to consume and remove some of the kwargs before
        passing them on to the children, e.g. an oscillator node might want to consume
        frequency multiplier and amplitude multiplier kwargs but we don't want to pass
        them to the child oscillator nodes.
        """
        raise NotImplementedError