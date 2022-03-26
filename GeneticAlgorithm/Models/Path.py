class Path:
    def __init__(self, source, dest, cost, amount):
        self.source = source
        self.dest = dest
        self.cost = cost
        self.amount = amount

    def __str__(self):
        return 'Cost -- source: ' + str(self.source) + ' -- dest: ' + str(self.dest) + ' -- cost: ' + str(self.cost) \
               + ' -- flow: ' + str(self.amount)