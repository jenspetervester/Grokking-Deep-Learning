weight = 0.5
goal_prediction = 0.8
input = 2
alpha = 0.1

for i in range(20):
	pred = input * weight
	error = (pred - goal_prediction) ** 2
	#derivative = input * (pred - goal_prediction)
	derivative = 2 * input * (-goal_prediction + input * weight)
	weight = weight - (alpha * derivative)
	print("prediction: " + str(pred) + " error: " + str(error))
