import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore


# get board state representation(from alans part)
boardShape = None

# define model structure 

#todo get board shape
boardShape = 11
lastNstates = 5

model = models.Sequential([
    layers.Input(shape= (boardShape,boardShape,lastNstates)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1 + boardShape*boardShape, activation='softmax') 
    #the first entry is the value head (0-1 chance of winning) all others correspond to probability distribution of the moves.
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO seperate layer sequence to predict value head and policy head


policy = layers.Conv2D(2, 1, activation='relu')(x)
policy = layers.BatchNormalization()(policy)
policy = layers.Flatten()(policy)
policy = layers.Dense(boardShape * boardShape, activation='softmax', name='policy')(policy)



inputShape = layers.Input(shape=(boardShape,boardShape,lastNstates))
#model = models.Model(inputs=inputShape ,outputs=[valueHead, policyHead])




