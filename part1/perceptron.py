import numpy as np

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model.

		This function will initialize a feature_dim weight vector,
		for each class.

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS]
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example
		"""
		self.num_class = num_class
		self.w = np.zeros((feature_dim+1,num_class))
		self.w[-1,:] = 1
		print(self.w)

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset.

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""
		total_num = train_label.shape[0]
		print("total_num",total_num)
		dim = train_set.shape[1]
		for i in range(total_num):
			true_label = train_label[i]
			w0 = np.dot(self.w[:-1,0], train_set[i])
			w0 += self.w[-1,0]
			pred = 0
			for j in range(1,self.num_class):
				w1 = np.dot(self.w[:-1,j], train_set[i])
				w1 += self.w[-1,j]
				if w1 > w0 :
					w0 = w1
					pred = j
			eta = (i/400+1)
			if pred != true_label:
				self.w[:-1,true_label] += train_set[i] / eta
				self.w[-1,true_label] += 1 / eta
				self.w[:-1,pred] -= train_set[i] / eta
				self.w[-1,pred] -= 1 / eta
		# YOUR CODE HERE
		pass

	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset.
			The accuracy is computed as the average of correctness
			by comparing between predicted label and true label.

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE
		accuracy = 0
		pred_label = np.zeros((len(test_set)))
		total_num = test_label.shape[0]
		dim = test_set.shape[1]
		for i in range(total_num):
			w0 = np.dot(test_set[i],self.w[:-1,0])
			w0 += self.w[-1,0]
			pred = 0
			for j in range(1,self.num_class):
				w1 = np.dot(self.w[:-1,j], test_set[i])
				w1 += self.w[-1,j]
				if w1 > w0 :
					w0 = w1
					pred = j
			pred_label[i] = pred
			if pred == test_label[i]:
				accuracy += 1
		accuracy = accuracy / total_num
		print(accuracy)
		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters
		"""

		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters
		"""

		self.w = np.load(weight_file)
