def readLabels() -> list[str]:
    """
    Reads all the labels from the MobileNet model into a list.
    """
    with open("ImageNetLabels.txt") as f:
        elems = f.readlines()
    
    return elems

def readFlowerLabels() -> list[str]:
    """
    Reads all the flower labels into a list.
    """
    with open("flowerLabels.txt") as f:
        elems = f.readlines()
    
    return elems

def getLabel(labels: list[str], pos: int) -> str:
    """
    Returns the label name from a specific position in the label list.
    """
    return labels[pos].strip()

