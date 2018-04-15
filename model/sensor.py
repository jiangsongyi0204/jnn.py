class Sensor:

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.inputData = []

    def scan(self, data):
        self.inputData = list(data)
    
    def debug(self):
        print(self.name + ":" + ''.join(self.inputData))