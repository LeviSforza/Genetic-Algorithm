class Instance:
    def __init__(self, dimX, dimY, name, machinesNumb, flow_path, cost_path):
        self.dimX = dimX
        self.dimY = dimY
        self.name = name
        self.machinesNumb = machinesNumb
        self.flow_path = flow_path
        self.cost_path = cost_path

    def __str__(self):
        return 'Instance -- dimX: ' + str(self.dimX) + ' -- dimY: ' + str(self.dimY) + ' -- name: ' + self.name \
               + ' -- machinesNumb: ' + str(self.machinesNumb)
