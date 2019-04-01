import numpy as np

class NaiveBayes(object):
    def __init__(self,num_class,feature_dim,num_value):
        """Initialize a naive bayes model.

        This function will initialize prior and likelihood, where
        prior is P(class) with a dimension of (# of class,)
            that estimates the empirical frequencies of different classes in the training set.
        likelihood is P(F_i = f | class) with a dimension of
            (# of features/pixels per image, # of possible values per pixel, # of class),
            that computes the probability of every pixel location i being value f for every class label.

        Args:
            num_class(int): number of classes to classify
            feature_dim(int): feature dimension for each example
            num_value(int): number of possible values for each pixel
        """
        print("num_class", num_class)
        print("num_value", num_value)
        print("feature_dim", feature_dim)
        self.num_value = num_value
        self.num_class = num_class
        self.feature_dim = feature_dim

        self.prior = np.zeros((num_class))
        self.likelihood = np.zeros((feature_dim,num_value,num_class))
        
    def train(self,train_set,train_label):
        """ Train naive bayes model (self.prior and self.likelihood) with training dataset.
            self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
            self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class).
            You should apply Laplace smoothing to compute the likelihood.

        Args:
            train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
            train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
        """
        print("line 40, begin to train")
        total = train_label.shape[0]
        dim   = train_set.shape[1]
        k = 0.1 ## need to change
        
        for i in range(total):
            label = train_label[i]
            self.prior[label] += 1
        print('self.prior assign finished')
        for i in range(total):#iterate through all the test case
            label = train_label[i]
            denominator = self.prior[label] + k*self.num_value
            addingcomponent = 1.0 / denominator
            for j in range(dim):
                value = train_set[i][j]
                self.likelihood[j][value][label] += addingcomponent
        
        print('likelihood assign finished')
        for i in range(self.num_class): #iterate through all the class
            denominator = self.prior[i] + k*self.num_value
            self.likelihood[:,:,i] += k / denominator
            self.prior[i] = self.prior[i] / total
            self.prior[i] = np.log(self.prior[i])
        
        print('likelihood modify finished')
        self.likelihood = np.log(self.likelihood)
        
        # YOUR CODE HERE
        print(self.prior)
        pass


    def test(self,test_set,test_label):
        """ Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
            by performing maximum a posteriori (MAP) classification.
            The accuracy is computed as the average of correctness
            by comparing between predicted label and true label.

        Args:
            test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
            test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

        Returns:
            accuracy(float): average accuracy value
            pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
        """
        print("line 79, begin to test")
        accuracy = 0
        pred_label = np.zeros((len(test_set)))

        # YOUR CODE HERE
        total = test_label.shape[0]
        dim = test_set.shape[1]
        num_class = self.num_class
        highest= {}
        lowest = {}
        for i in range(num_class):
            highest[i]=(0,-1000000)
            lowest[i] =(0,1000000)
        for i in range(total):
            #calculate for the first class
            pred = 0
            map = self.prior[0]
            for d in range(dim):
                value = test_set[i][d]
                map += self.likelihood[d][value][0]
            for label in range(1,num_class):
                probability = self.prior[label]
                for d in range(dim):
                    value = test_set[i][d]
                    probability += self.likelihood[d][value][label]
                if probability > map:
                    map = probability
                    pred = label
                if test_label[i]==pred and probability > highest[test_label[i]][1]:
                    highest[test_label[i]]=(i,probability)
                if test_label[i]==pred and probability < lowest[test_label[i]][1]:
                    lowest[test_label[i]]=(i,probability)
            pred_label[i] = pred
            if test_label[i] == pred:
                accuracy+=1
        accuracy = accuracy / total
        print(accuracy)
        print('highest:',highest)
        print('lowest:',lowest)
        return accuracy, pred_label

        pass




    def save_model(self, prior, likelihood):
        """ Save the trained model parameters
        """

        np.save(prior, self.prior)
        np.save(likelihood, self.likelihood)

    def load_model(self, prior, likelihood):
        """ Load the trained model parameters
        """

        self.prior = np.load(prior)
        self.likelihood = np.load(likelihood)

    def intensity_feature_likelihoods(self, likelihood):
        """
        Get the feature likelihoods for high intensity pixels for each of the classes,
            by sum the probabilities of the top 128 intensities at each pixel location,
            sum k<-128:255 P(F_i = k | c).
            This helps generate visualization of trained likelihood images.

        Args:
            likelihood(numpy.ndarray): likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class)
        Returns:
            feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
                (# of features/pixels per image, # of class)
        """
        # YOUR CODE HERE

        feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2])) # dim by num of class
        d = self.feature_dim
        c = self.num_class
        for dim in range(d):
            for label in range(c):
                for k in range(128,256):
                    feature_likelihoods[dim][label] += self.likelihood[dim][k][label]
        return feature_likelihoods
