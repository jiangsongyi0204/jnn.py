import matplotlib.pyplot as plt
import random

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
    
    @staticmethod
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    @staticmethod
    def pos_random_sample(size, min_rate, max_rate):
        posarr = [m for m in range(0,size)]
        picksize = random.randrange(round(size*min_rate), round(size*max_rate))
        return random.sample(posarr,picksize)

    @staticmethod
    def arr_random_sample(vec, min_rate, max_rate):
        size = len(vec)
        picksize = random.randrange(round(size*min_rate), round(size*max_rate))
        return random.sample(vec,picksize)
