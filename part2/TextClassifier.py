# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0
        self.word_freq = []    #label, {word, freq}.    (a list of dictionaries)
        self.label_freq = []

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # TODO: Write your code here

        for i in range(15):
            self.word_freq.append({})
            self.label_freq.append(1)

        #set up label_freq (checked correct, sum = 1.0)
        for label in train_label:
            self.label_freq[label] += 1
        for i in range(15):
            self.label_freq[i] /= (len(train_label) + 15)

        #set up word_freq
        for i in range(len(train_label)):
            label = train_label[i]
            text = train_set[i]
            for word in text:
                if word not in self.word_freq[label].keys():
                    self.word_freq[label][word] = 1
                else:
                    self.word_freq[label][word] += 1



        print(self.word_freq)
        print(self.label_freq)
        return


    def predict(self, x_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []

        # TODO: Write your code here
        total_words = 0
        for i in range(15):
            total_words += len(self.word_freq[i])

        for i in range(len(dev_label)):

            ans = dev_label[i]      #the correct answer
            text = x_set[i]

            #a list for the probabilities of 15 labels
            prob_list = []
            for label_num in range(15):
                prob_list.append(-1)    #default value

            #calculate prob list
            for label_num in range(15):         #for every possible label
                #calculate the probability that the text is label-x (with unigram model formula)
                prob = (self.label_freq[label_num])
                for word in text:
                    if word in self.word_freq[label_num]:
                        prob *= (self.word_freq[label_num][word] / total_words)
                    else:
                        prob *= (1 / total_words)
                prob_list[label_num] = prob

            #choose max prob label
            prediction = prob_list.index(max(prob_list))
            result.append(prediction)
            if prediction == ans:
                accuracy += 1

        accuracy /= len(dev_label)

        return accuracy,result

