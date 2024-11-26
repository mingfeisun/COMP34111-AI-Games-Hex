import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore

class trainingModel:
    _boardShape = 11
    lastNstates = 5
    model=None

    def __init__(self):
        print("define layers here")

        inputShape = layers.Input(shape=(self.boardShape*self.boardShape,self.lastNstates))

        #TODO tweak layers
        valueHead = layers.Conv2D(1, 1, activation='relu')(inputShape) #as only want one input
        valueHead = layers.Conv2D(32, 32, activation='relu')(valueHead) # what does 32 define? picked random value for now
        valueHead = layers.Flatten()(valueHead)
        valueHead = layers.Dense(1, activation='tanh', name='value')(valueHead)

        policyHead = layers.Conv2D(32, 32, activation='relu')(inputShape)
        policyHead - layers.Flatten()(policyHead)
        policyHead = layers.Dense(self._boardShape*self._boardShape, activation='softmax', name='policy')(policyHead)
        
        model = models.Model(inputs=inputShape ,outputs=[valueHead, policyHead])

    def getHeads(self,boardState):

        #tmp placeholders for jonah
        value = 1
        policy = [0]*(self._boardShape*self._boardShape)
        return(value,policy)


    def trainModel():
        print("TODO")









