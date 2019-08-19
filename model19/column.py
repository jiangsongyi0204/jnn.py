import random
import numpy as np
import math

class Link:
    CHANGE_STEP = 0.01    
    def __init__(self, name, lType, fromSynapse, toSynapse):
        self.name = name
        self.lType = lType
        self.fromSynapse = fromSynapse
        self.toSynapse = toSynapse
        self.weight = 0.5 + random.uniform(-0.2, 0.2)

    def update(self):
        local_change_step = 0
        if self.lType == "l00":
            if self.fromSynapse.value == 0 and self.toSynapse.value == 0:
                local_change_step = Link.CHANGE_STEP
            else:
                local_change_step = -Link.CHANGE_STEP
        elif self.lType == "l01":
            if self.fromSynapse.value == 0 and self.toSynapse.value == 1:
                local_change_step = Link.CHANGE_STEP
            else:
                local_change_step = -Link.CHANGE_STEP
        elif self.lType == "l10":
            if self.fromSynapse.value == 1 and self.toSynapse.value == 0:
                local_change_step = Link.CHANGE_STEP
            else:
                local_change_step = -Link.CHANGE_STEP
        elif self.lType == "l11":
            if self.fromSynapse.value == 1 and self.toSynapse.value == 1:
                local_change_step = Link.CHANGE_STEP
            else:
                local_change_step = -Link.CHANGE_STEP
        self.weight += local_change_step

class Synapse:
    def __init__(self, name, pos, cell):
        self.name = name
        self.pos = pos
        self.cell = cell
        self.links = []
        self.value = 0.0

    def makeLinks(self):
        link = Link("test","l00",self,self)
        self.links.append(link)

class Cell:
    def __init__(self, name, pcell, minicolumn):
        self.name = name
        self.pcell = pcell
        self.minicolumn = minicolumn
        self.synapses_prediction = []
        self.synapses_forword = []
        self.initSynapses()

    def initSynapses(self):
        for x in range(9):
            self.synapses_prediction.append(Synapse("S"+str(x),x,self))
            self.synapses_forword.append(Synapse("S"+str(x),x,self))
    
    def linkTo(self,cell):
        print("linkTo("+self.name+","+cell.name+")")

    def forword(self,inputData):
        print(inputData)

class Minicolumn:
    def __init__(self, name, direction, column):
        self.name = name
        self.direction = direction
        self.column = column
        self.cells = []
        self.links = []
        self.initCells()

    def initCells(self):
        self.cells.append(Cell(self.name + "-l0", None, self))
        for x in range(1,int(math.log(self.column.size,3))+1):
            self.cells.append(Cell(self.name + "-l" + str(x), self.cells[x-1], self))
        for x in range(1,len(self.cells)):
            self.cells[x-1].linkTo(self.cells[x])

    def forword(self,inputData):
        (size,t)= inputData.shape
        center = int((size - 1)/2) 
        cener_x = center
        cener_y = center
        self.cells[0].forword(inputData[cener_x:cener_x+1,cener_y:cener_y+1])
        for x in range(1,len(self.cells)):
            idx = x -1
            dx = 3**idx
            if self.direction == 1:
                cener_x = center - dx
                cener_y = center - dx
            elif self.direction == 2:
                cener_x = center - dx
                cener_y = center
            elif self.direction == 3:
                cener_x = center - dx
                cener_y = center + dx
            elif self.direction == 4:
                cener_x = center
                cener_y = center - dx
            elif self.direction == 5:
                cener_x = center
                cener_y = center + dx
            elif self.direction == 6:
                cener_x = center + dx
                cener_y = center - dx
            elif self.direction == 7:
                cener_x = center + dx
                cener_y = center
            elif self.direction == 8:
                cener_x = center + dx
                cener_y = center + dx
            r = int(dx/2)    
            self.cells[x].forword(inputData[cener_x-r:cener_x+r+1,cener_y-r:cener_y+r+1])
        #TODO

class Column:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.inputData = np.zeros((self.size,self.size))
        self.isLearning = False
        self.minicolumns = []
        self.initMiniColumns()

    def initMiniColumns(self):
        for x in range(1,8):
            self.minicolumns.append(Minicolumn("MC"+str(x),x,self))

    def forword(self,inputData):
        for mc in self.minicolumns:
            mc.forword(inputData)
        #TODO