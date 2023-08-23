import numpy as np
import matplotlib.pyplot as mat
import scipy.special as sp
import network as nw

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2

network = nw.network(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()


i = 0
for record in training_data_list:
	all_values = record.split(",")
	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

	targets = np.zeros(output_nodes) + 0.01
	targets[int(all_values[0])] = 0.99

	network.train(inputs, targets)

	i+=1
	rate = i / len(training_data_list) * 100

	if rate % 5 < 0.001:
		print(f"Нейросеть обучилась на {int(rate)}%")

print("--------------------------------")

while True:
	inputs = np.zeros(output_nodes) + 0.01
	print("Введите желаемую цифру: ", end="")
	inputs[int(input())] = 0.99

	outputs = network.back_query(inputs) * 255
	image_array = outputs.reshape((28, 28))
	mat.imshow(image_array, cmap="Greys", interpolation="None")

	mat.show()

	print("Закончить? ", end="")
	if int(input()):
		break