'''
Layer 1:
    The input layer that receive all kind of outside world data.
    Like IMAGE,Audio,....
'''
class InputLayer:

    def __init__(self, opt):
        self.opt = opt
        self.layer_type = "input"
    
    def forward(self,v):
        self.in_act = v
        self.out_act = v
        return self.out_act
    
    def backward(self):
        return
    
'''
Layer 2:
    The classify layer.
'''
class ClassifyLayer:

    def __init__(self):
        self.classifications = []
        self.layer_type = "Classify"
    
    def forward(self,v):
    
