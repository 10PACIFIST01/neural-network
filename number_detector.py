import numpy as np
import matplotlib.pyplot as mat
import network as nw

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

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

test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
	all_values = record.split(',')
	correct_label = int(all_values[0])
	#print(correct_label, "<-- должно быть так")

	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	outputs = network.query(inputs)
	label = np.argmax(outputs)

	#print(label, "<-- так думает нейросеть")

	if (label == correct_label):
		scorecard.append(1)
	else:
		scorecard.append(0)

	#print("--------------------------------")

print(f"эффективность равна {sum(scorecard) / len(scorecard) * 100}%")