import numpy as np

# Training data
x = np.array([1,2,3,4,5])
y = np.array([40,45,50,55,60])


# Model parameters
m=0
b=0

# Learning rate
learning_rate = 0.01

# Number of times to Learn
epochs = 2000

for i in range(epochs):

    # Make prediction
    prediction = m*x+b

    # Calculate errors
    errors = prediction - y

    # Calculate gradients
    dm = (2 / len(x)) * np.sum(errors * x)
    db = (2 / len(x)) * np.sum(errors)

    # Update parameters
    m = m-learning_rate * dm
    b = b-learning_rate * db

    # Predict score for 7 hours
    hours = 7
    prediction = m * hours + b
    
    print("Predicted score:", prediction)