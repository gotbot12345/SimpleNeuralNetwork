import numpy as np

inputs = [1, 2, 3, 4, 5, 6]
targets = [1, 2, 3, 4, 5, 6]

w = np.random.rand()
b = np.random.rand()

def Y(inputs, weight, bias):
    preds = []
    for i in range(len(inputs)):
        preds.append(inputs[i] * weight + bias)
    return preds  

def dcdw(inputs, preds, targets):
    dcdw = 0
    for i in range(len(preds)):
        dcdw = 2 * (preds[i] - targets[i]) * inputs[i]
    return dcdw

def dcdb(preds, targets):
    dcdb = 0
    for i in range(len(preds)):
        dcdb = 2 * (preds[i] - targets[i])
    return dcdb

for i in range(1000):          
    output = Y(inputs, w, b)
    dw = dcdw(inputs, output, targets)
    db = dcdb(output, targets)
    w = w + -0.005 * dw
    b = b + -0.005 * db
    print("output: " + str(output) + "\nderivative of w: " + str(dw) + "\nderivative of b: " + str(db) + "\nw: " + str(w) + "\nb: " + str(b))

w = np.round(w, 2)
b = np.round(b, 2)

print("Final Parameters: " + "\nw: " + str(w) + "\nb: " + str(b))
    