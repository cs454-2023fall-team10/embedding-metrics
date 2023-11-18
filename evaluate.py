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

    def safe_similarity(self, s1, s2):
        value = self.similarity(s1, s2)
        if value < 0 or value > 1:
            raise ValueError(
                f"Similarity value should be between 0 and 1, but {value} is given."
            )
        return value


class RandomEvaluator(Evaluator):
    def name(self):
        return "RandomEvaluator"

    def similarity(self, s1, s2):
        return random.random()


class FastTextEvaluator(Evaluator):
    def __init__(self, model_name):
        from gensim.models import fasttext

        self._model = fasttext.load_facebook_vectors(model_name)

    def name(self):
        return "FastTextEvaluator"

    def similarity(self, s1, s2):
        return self._model.wv.n_similarity(s1.split(), s2.split())


class PororoEvaluator(Evaluator):
    def __init__(self):
        from pororo import Pororo

        self._model = Pororo(task="similarity", lang="ko")

    def name(self):
        return "PororoEvaluator"

    def similarity(self, s1, s2):
        return self._model(s1, s2)


class SentenceBertEvaluator(Evaluator):
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def name(self):
        return "SentenceBertEvaluator"

    def similarity(self, s1, s2):
        from sentence_transformers import util

        s1_embedding = self._model.encode(s1)
        s2_embedding = self._model.encode(s2)

        return util.pytorch_cos_sim(s1_embedding, s2_embedding)[0][0]


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

    def cross_entropy_loss(self, evaluator, print_details=False):
        similarity = [
            evaluator.safe_similarity(self.intent, choice) for choice in self.choices
        ]
        softmax_similarity = softmax(similarity)
        loss = -np.log(softmax_similarity[self.label])
        
        if print_details:
            print(self.intent, self.choices, similarity, self.label, loss)
            
        return loss
    
    def cosine_similarity_loss(self, evaluator, print_details=False):
        similarity = [
            evaluator.safe_similarity(self.intent, choice) for choice in self.choices
        ]
        
        loss = 0
        for i, sim in enumerate(similarity):
            if i == self.label:
                loss += 1 - sim
            else:
                loss += sim
        
        if print_details:
            print(self.intent, self.choices, similarity, self.label, loss)
        
        return loss


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

    evaluators = [
        RandomEvaluator(),
        # FastTextEvaluator("./vector_embedding/fasttext/models/cc.ko.300.bin"),
        PororoEvaluator(),
        # SentenceBertEvaluator("jhgan/ko-sroberta-multitask"),
    ]

    # Calculate loss
    for evaluator in evaluators:
        loss = 0
        for i, item in enumerate(data):
            loss += item.cross_entropy_loss(evaluator, print_details=i % 10 == 0)

        print(f"Loss for {evaluator.name()}: {loss}")
