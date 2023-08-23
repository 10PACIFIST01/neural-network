import numpy as np
import scipy.special as sp

class network:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_grade, function=lambda x: sp.expit(x)):
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes

		self.rate = learning_grade

		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		self.activation_function = function

	def train(self, inputs_list, target_list):
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(target_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		self.who += self.rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
		self.wih += self.rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

	def query(self, inputs_list):
		inputs = np.array(inputs_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

	def back_query(self, inputs_list):
		inputs = np.array(inputs_list, ndmin=2)
		inputs = sp.logit(inputs)

		hidden_inputs = np.dot(inputs, self.who)
		hidden_inputs -= np.min(hidden_inputs)
		hidden_inputs /= np.max(hidden_inputs)
		hidden_inputs *= 0.98
		hidden_inputs += 0.01
		hidden_outputs = sp.logit(hidden_inputs)

		final_inputs = np.dot(hidden_outputs, self.wih)
		final_inputs -= np.min(final_inputs)
		final_inputs /= np.max(final_inputs)
		final_inputs *= 0.98
		final_inputs += 0.01

		return final_inputs