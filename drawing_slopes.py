import matplotlib
import matplotlib.pyplot as plt

weight = 0.5
input = 0.5
goal_prediction = 0.8

def y(a, x, b):
    x_vals = []
    x_vals.append(x - 0.1*x)
    x_vals.append(x)
    x_vals.append(x + 0.1*x)
    xy = [[],[]]
    
    for x in x_vals:
        xy[0].append(x)
        xy[1].append(a*x+b)
    return xy

test = y(0, 1.6, 0)

plt.plot(test[0], test[1])

a = input * (input * weight - goal_prediction) 
test2 = y(a,0.5, 0)
plt.plot(test2[0], test2[1])

def calc_error(input, weight):
    pred = input * weight
    error = (pred - goal_prediction) ** 2

errors = []
weights = []
for i in range(30):
    weight = i/10
    error = calc_error(input, weight)
    derivative = input * (input * weight - goal_prediction)
    errors.append(error)
    weights.append(weight)
    #print("weight, error, derivative: {}, {}, {}".format(weight, error, derivative))

plt.plot(weights, errors)
#plt.axis([0, 3, 0, 1])
plt.xlabel('weight')
plt.ylabel('error')
plt.show()
