import numpy as np
from collections import defaultdict


class Bayes(object):
    '''
    Implements Naïve Bayes model for the given dataset
    '''

    def __init__(self, weight=1):
        '''
        Initializes the Naïve Bayes object
        '''
        self.weight = weight
        pass

    def train(self, dataset, outcome):
        self.possible_outcomes = set(outcome)
        self.total_count = len(outcome)

        self.vocabulary = set([word for text in dataset for word in text])
        self.probability_priori = {item: np.log(len([None for o in outcome if o == item])/self.total_count) for item in self.possible_outcomes}

        self.possible_outcomes_count = {possible_outcome: len(
            [item for item in outcome if item == possible_outcome]) for possible_outcome in self.possible_outcomes}
        self.word_counter = defaultdict(lambda: defaultdict(lambda: 0))
        # populate tables
        for i, text in enumerate(dataset):
            for word in text:
                self.word_counter[outcome[i]][word] += 1
        self.likelyhood = defaultdict(lambda: defaultdict(lambda: 0))
        for word in self.vocabulary:
            for possible_outcome in self.possible_outcomes:
                self.likelyhood[word][possible_outcome] = np.log((self.word_counter[possible_outcome][word]+1)/(sum([self.word_counter[possible_outcome][word_count] for word_count in self.word_counter[possible_outcome]])+len(self.vocabulary)))

    def predict(self, values, print_probabilities=False):
        possible_outcomes = {
            possible_outcome: 0 for possible_outcome in self.possible_outcomes}
        for value in values:
            for possible_outcome in possible_outcomes:
                possible_outcomes[possible_outcome] += self.likelyhood[value][possible_outcome]

        if print_probabilities:
            print([f"Likelyhood {possibility}: {possible_outcomes[possibility]}" for possibility in possible_outcomes])
        return max(possible_outcomes, key=possible_outcomes.get)
