def flatten(lst):
    """
    Recursively flattens a nested list of arbitrary depth.

    Parameters
    ----------
    lst : list
        A (possibly nested) list, e.g. [1, [2, [3, 4]], 5].

    Yields
    ------
    element
        Each non-list element contained in `lst`, in depth-first order.

    Examples
    --------
    >>> list(flatten([1, [2, [3, 4]], 5]))
    [1, 2, 3, 4, 5]
    """
    for item in lst:
        if isinstance(item, list):
            # If the current item is a list, recursively flatten it
            yield from flatten(item)
        else:
            # Otherwise, yield the item directly
            yield item
