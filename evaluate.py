import numpy as np
from scipy.special import softmax
import random


class Evaluator:
    def name(self):
        """
        Return name of evaluator.
        """
        raise NotImplementedError()

    def similarity(self, s1, s2):
        """
        Evaluate similarity between two strings.
        Should return value between 0 and 1.
        """
        raise NotImplementedError()


class RandomEvaluator(Evaluator):
    def name(self):
        return "RandomEvaluator"

    def similarity(self, s1, s2):
        return random.random()


class LabeledData:
    def __init__(self, json):
        self._json = json

        self.intent = json["intent"]
        self.prompt = json["prompt"]
        self.choices = [item["text"] for item in json["choices"]]

        for i, choice in enumerate(json["choices"]):
            if choice["nextSectionId"] == json["choice"]["nextSectionId"]:
                self.label = i
                break

    def __str__(self):
        return f"LabeledData(intent={self.intent}, prompt={self.prompt}, choices={self.choices}, label={self.label})"

    def __repr__(self):
        return str(self)

    def cross_entropy_loss(self, evaluator):
        # Caculate softmax(similarity(intent, choice)) for each choice
        # Then calculate cross entropy loss
        similarity = [
            evaluator.similarity(self.intent, choice) for choice in self.choices
        ]
        softmax_similarity = softmax(similarity)
        return -np.log(softmax_similarity[self.label])


def load_label_data(filename):
    import json

    with open(filename, "r") as f:
        data = json.load(f)
        return [LabeledData(item) for item in data]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <data_filename>")
        exit(1)

    data_filename = sys.argv[1]

    data = load_label_data(data_filename)

    # Stats for data
    print(f"Total data: {len(data)}")

    evaluator = RandomEvaluator()

    # Calculate cross entropy loss
    loss = 0
    for item in data:
        loss += item.cross_entropy_loss(evaluator)

    print(f"Cross entropy loss for {evaluator.name()}: {loss}")
