input = 2
weight = 0.5
goal_prediction = 0.8
alpha = 0.1

for i in range(20):
	pred = input * weight
	error = (pred - goal_prediction) ** 2
	derivative = input * (pred - goal_prediction)
	weight = weight - (alpha * derivative)
	print("Error: {:12.10f}, derivative: {:10.6f}, prediction: {:10.5f}".format(error, derivative, pred))
