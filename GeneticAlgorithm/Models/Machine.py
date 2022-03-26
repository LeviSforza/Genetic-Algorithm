class Machine:
    def __init__(self, number, x, y):
        self.number = number
        self.x = x
        self.y = y

    def __str__(self):
        return 'Machine -- number: ' + str(self.number) + ' -- x: ' + str(self.x) + ' -- y: ' + str(self.y)
