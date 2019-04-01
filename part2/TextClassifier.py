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
import math
from collections import OrderedDict

class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification
        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0
        self.word_freq = []  # label, {word, freq}.    (a list of dictionaries)
        self.label_freq = []
        self.bi_word_freq = []


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
        #initalization
        for i in range(15):
            self.word_freq.append({})
            self.label_freq.append(0)     # no laplace smoothing
            self.bi_word_freq.append({})  # initialized with smoothing with factor 1

        # set up label_freq(prior) (checked correct, sum = 1.0)
        for label in train_label:
            self.label_freq[label] += 1

        for i in range(15):
            self.label_freq[i] /= (len(train_label) + 15)       #to prob
            #self.label_freq[i] = math.log(self.label_freq[i])   #to log()

        # set up word_freq
        for i in range(len(train_label)):  # iterate through all 14 labels
            label = train_label[i] - 1  # match label with index
            text = train_set[i]
            for word in text:  # read current line of text and count freq
                if word not in self.word_freq[label].keys():
                    self.word_freq[label][word] = 1
                else:
                    self.word_freq[label][word] += 1



        # set up bi-word_freq
        for i in range(len(train_label)):  # iterate through all 15 labels
            label = train_label[i] - 1  # match label with index
            text = train_set[i]
            for j in range(1, len(text)):  # read current line of text and count freq
                if (text[j - 1], text[j]) not in self.bi_word_freq[label].keys():
                    self.bi_word_freq[label][(text[j - 1], text[j])] = 1
                else:
                    self.bi_word_freq[label][(text[j - 1], text[j])] += 1

        #testing
        #print("word_freq:")
        #print(self.word_freq)
        #print("bi_word_freq:")
        #print(self.bi_word_freq)
        #print("label_freq:")
        #print(self.label_freq)

        return



    def predict(self, x_set, dev_label, lambda_mix=0.0):
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
        #calculate total num of words as denominator
        total_words = 0
        for i in range(15):
            total_words += len(self.word_freq[i])

        for i in range(len(dev_label)):

            ans = dev_label[i]  # the correct answer
            text = x_set[i]

            # initialize a list for the probabilities of 15 labels (prior)
            prob_list = []
            for label_num in range(15):
                prob_list.append(float("-inf"))  # default value

            # calculate prob list
            for label_num in range(15):  # for every possible label
                # calculate the probability that the text is label-x (with unigram model formula)
                prob = (self.label_freq[label_num])     #prior
                for word in text:
                    if word in self.word_freq[label_num]:
                        prob *= (self.word_freq[label_num][word] / total_words)
                    else:
                        prob *= (1 / total_words)

                prob_list[label_num] = prob
                #print(prob_list)

            # choose max prob label
            prediction = prob_list.index(max(prob_list)) + 1  # match index with label
            result.append(prediction)
            if prediction == ans:
                accuracy += 1

        #print(result)

        #get the 20 feature words
        #for i in range(15):
            #sorted_d = sorted(self.word_freq[i].items(), key=lambda x:x[1], reverse = True)
            #print("Word Frequencies for Label ", i)
            #print(sorted_d)


        accuracy /= len(dev_label)

        return accuracy, result






    def predict_bi(self, x_set, dev_label, lambda_mix=0.0):  # prediction using uni-bi-gram combined
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
        # calculate total num of words as denominator
        total_words = 0
        total_biwords = 0

        for i in range(15):
            total_words += len(self.word_freq[i])
            total_biwords += len(self.bi_word_freq[i])

        for i in range(len(dev_label)):

            ans = dev_label[i]  # the correct answer
            text = x_set[i]

            # initialize a list for the probabilities of 15 labels (prior)
            prob_list = []
            for label_num in range(15):
                prob_list.append(float("-inf"))  # default value

            # calculate prob list
            for label_num in range(15):  # for every possible label
                # calculate the probability that the text is label-x (with unigram model formula)
                prob = (self.label_freq[label_num])  # prior
                prob2 = self.label_freq[label_num]

                for word in text:
                    if word in self.word_freq[label_num]:
                        prob *= (self.word_freq[label_num][word] / total_words)
                    else:
                        prob *= (1 / total_words)

                for j in range(1, len(text)):
                    if (text[j - 1], text[j]) in self.bi_word_freq[label_num]:
                        prob2 *= (self.bi_word_freq[label_num][(text[j - 1], text[j])] / total_biwords)
                    else:
                        prob2 *= (1 / total_biwords)

                prob_list[label_num] = self.lambda_mixture * prob + (1 - self.lambda_mixture) * prob2
                # print(prob_list)

            # choose max prob label
            prediction = prob_list.index(max(prob_list)) + 1  # match index with label
            result.append(prediction)
            if prediction == ans:
                accuracy += 1

        # print(result)
        accuracy /= len(dev_label)

        return accuracy, result
