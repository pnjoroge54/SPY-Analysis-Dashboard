def getIndexOfTuple(lst, index, value):
    for pos, t in enumerate(lst):
        if t[index] == value:
            return pos + 1
            
    # Matches behavior of list.index
    raise ValueError("list.index(x): x not in list")