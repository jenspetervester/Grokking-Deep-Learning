def elementwise_multiplication(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))
    res = []
    for ix, item in enumerate(vec_a):
        res.append(vec_a[ix] * vec_b[ix])
    return res
	
def elementwise_addition(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))
    res = []
    for i in range(len(vec_a)):
        res.append(vec_a[i] + vec_b[i])
    return res
	
def vector_sum(vec_a):
    res = 0
    for elem in vec_a:
        res += elem
    return res
	
def vector_average(vec_a):
    return vector_sum(vec_a) / len(vec_a)
	
# dot product / weighted sum
def w_sum(vec_a, vec_b):
	#print("len vec_a {} \n len vec_b {}".format(len(vec_a), len(vec_b)))
	assert(len(vec_a) == len(vec_b))
	multiplied_elements = elementwise_multiplication(vec_a, vec_b)
	return vector_sum(multiplied_elements)

def vector_of_zeroes(len):
	return [0] * len

def matrix_of_zeroes(rows, columns):
	res = []
	for r in range(rows):
		res.append([0] * columns)
	return res
	
def vect_mat_mul(vect, matrix):
	#print("vect{} \n matrix {}".format(vect, matrix))
	#print("len vect {} \n len matrix {}".format(len(vect), len(matrix)))
	assert(len(vect) == len(matrix))
	output = vector_of_zeroes(len(vect))
	for i in range(len(vect)):
		#print("** i={}, len(vect)={}".format(i, len(vect)))
		output[i] = w_sum(vect,matrix[i])
		#print("output[i]={}".format(output[i]))
	return output
	
def outer_prod(vec_a, vec_b):
	#print("vec_a: {}, vec_b: {}".format(vec_a, vec_b))
	res = matrix_of_zeroes(len(vec_a), len(vec_b))
	for i in range(len(vec_a)):
		for j in range(len(vec_b)):
			res[i][j] = vec_a[i] * vec_b[j]
	return res