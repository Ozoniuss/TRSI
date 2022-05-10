with open("ImageNetLabels.txt") as f:
    elems = f.readlines()

# reads all the classification labels
def readLabels() -> list[str]:
    with open("ImageNetLabels.txt") as f:
        elems = f.readlines()
    
    return elems

# find the label at a specific position
def getLabel(labels: list[str], pos: int) -> str:
    return labels[pos].strip()

def readFlowerLabels() -> list[str]:
    with open("flowerLabels.txt") as f:
        elems = f.readlines()
    
    return elems