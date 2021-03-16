
def winner(x1,x2,dataset):
    (_, c1), (_, c2)  = dataset[x1], dataset[x2]
    if c1 == "paper":
        return 1
    return 0
