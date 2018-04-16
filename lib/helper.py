import matplotlib.pyplot as plt

class Helper:
    
    @staticmethod
    def draw(fc):
        fig, ax = plt.subplots(nrows=2, ncols=5)
        plt.subplots_adjust(left=0.01, bottom=0.01, right=1, top=1, wspace=0.05, hspace=0.05)
        i = 0
        for row in ax:
            for col in row:
                col.imshow(fc[i].getFeatureImg())
                i=i+1
        plt.show()