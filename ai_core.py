# ai_core.py

import os
import joblib
from collections import defaultdict, Counter, deque


# ==============================
# Pattern Memory
# ==============================

class PatternMemory:

    def __init__(self, max_order=7):
        self.max_order = max_order
        self.memory = defaultdict(Counter)


    def learn(self, history):

        if len(history) < 2:
            return

        for size in range(1, self.max_order + 1):

            for i in range(len(history) - size):

                pattern = tuple(history[i:i+size])
                nxt = history[i+size]

                self.memory[pattern][nxt] += 1



    def predict(self, history):

        for size in range(
            min(self.max_order, len(history)),
            0,
            -1
        ):

            pattern = tuple(history[-size:])


            if pattern in self.memory:

                results = self.memory[pattern]

                value, count = results.most_common(1)[0]

                confidence = count / sum(results.values())


                return {
                    "value": value,
                    "confidence": confidence,
                    "name": "PatternMemory"
                }


        return {
            "value": None,
            "confidence": 0,
            "name": "PatternMemory"
        }



# ==============================
# Frequency / Statistical Brain
# ==============================

class PatternAnalyzer:


    def __init__(self):

        self.counter = Counter()



    def learn(self, history):

        self.counter.clear()

        self.counter.update(history)



    def predict(self):

        if not self.counter:

            return {
                "value": None,
                "confidence":0,
                "name":"Frequency"
            }


        value,count = self.counter.most_common(1)[0]


        confidence = (
            count /
            sum(self.counter.values())
        )


        return {
            "value":value,
            "confidence":confidence,
            "name":"Frequency"
        }




# ==============================
# Markov Transition Brain
# ==============================

class TransitionAnalyzer:


    def __init__(self):

        self.markov1 = defaultdict(Counter)

        self.markov2 = defaultdict(Counter)



    def learn(self,history):


        self.markov1.clear()
        self.markov2.clear()


        for a,b in zip(history,history[1:]):

            self.markov1[a][b]+=1



        for i in range(len(history)-2):

            key=(
                history[i],
                history[i+1]
            )

            self.markov2[key][history[i+2]]+=1




    def predict(self,history):


        # Markov 2 first

        if len(history)>=2:

            key=(
                history[-2],
                history[-1]
            )


            if key in self.markov2:


                data=self.markov2[key]

                value,count=data.most_common(1)[0]


                return {
                    "value":value,
                    "confidence":
                        count/sum(data.values()),
                    "name":"Markov2"
                }



        # Markov 1

        if history:

            last=history[-1]


            if last in self.markov1:

                data=self.markov1[last]

                value,count=data.most_common(1)[0]


                return {
                    "value":value,
                    "confidence":
                        count/sum(data.values()),
                    "name":"Markov1"
                }



        return {
            "value":None,
            "confidence":0,
            "name":"Markov"
        }



# ==============================
# Short Term Analyzer
# ==============================

class ShortTermAnalyzer:


    def __init__(self,window):

        self.window=window



    def predict(self,history):


        if len(history)<self.window+1:

            return {
                "value":None,
                "confidence":0,
                "name":
                f"Short{self.window}"
            }



        pattern=tuple(
            history[-self.window:]
        )


        matches=Counter()


        for i in range(
            len(history)-self.window
        ):

            if tuple(
                history[i:i+self.window]
            )==pattern:

                matches[
                    history[i+self.window]
                ]+=1



        if matches:


            value,count=matches.most_common(1)[0]


            return {
                "value":value,
                "confidence":
                count/sum(matches.values()),
                "name":
                f"Short{self.window}"
            }



        return {
            "value":None,
            "confidence":0,
            "name":
            f"Short{self.window}"
        }# ==============================
# Performance Tracker
# ==============================

class PerformanceTracker:

    def __init__(self):

        self.results = defaultdict(
            lambda: deque(maxlen=100)
        )


    def update(self, model, correct):

        self.results[model].append(
            1 if correct else 0
        )


    def accuracy(self, model):

        data = self.results[model]

        if not data:
            return 0.5

        return sum(data) / len(data)



# ==============================
# Drift Detector
# ==============================

class DriftDetector:

    def __init__(self, window=50):

        self.window = window
        self.history = deque(
            maxlen=window
        )


    def update(self, correct):

        self.history.append(
            1 if correct else 0
        )


    def detected(self):

        if len(self.history) < self.window:
            return False


        accuracy = (
            sum(self.history)
            /
            len(self.history)
        )


        return accuracy < 0.45



# ==============================
# Decision Manager
# ==============================

class DecisionManager:


    def __init__(self):

        self.weights = defaultdict(
            lambda:1.0
        )

        self.prediction_streak = 0
        self.last_prediction = None



    def update_weights(
        self,
        performance
    ):

        for model in performance.results:

            score = performance.accuracy(
                model
            )


            # keep weights balanced

            self.weights[model] = (
                0.5 + score
            )



    def decide(
        self,
        predictions,
        performance
    ):


        self.update_weights(
            performance
        )


        votes = defaultdict(float)

        explanation=[]



        for item in predictions:

            value=item["value"]


            if value is None:
                continue


            name=item["name"]

            confidence=item["confidence"]


            reliability = (
                self.weights[name]
            )


            score = (
                confidence *
                reliability
            )


            votes[value]+=score


            explanation.append(
                f"{name} supports {value} "
                f"({confidence*100:.1f}%)"
            )



        if not votes:

            return {
                "prediction":None,
                "confidence":0,
                "explanation":
                [
                    "Not enough data"
                ]
            }



        prediction=max(
            votes,
            key=votes.get
        )


        confidence = (
            votes[prediction]
            /
            sum(votes.values())
        )


        # prevent fake certainty

        confidence=min(
            confidence,
            0.85
        )


        # prediction repetition penalty

        if prediction == self.last_prediction:

            self.prediction_streak += 1

        else:

            self.prediction_streak=1



        self.last_prediction=prediction



        if self.prediction_streak > 3:

            confidence *= 0.85



        return {

            "prediction":prediction,

            "confidence":confidence,

            "explanation":explanation

        }



# ==============================
# Adaptive Engine
# ==============================

class AdaptiveEngine:


    def __init__(self):


        # empty history

        self.history=[]


        self.pattern_memory = PatternMemory()


        self.pattern_analyzer = PatternAnalyzer()


        self.transition_analyzer = TransitionAnalyzer()


        self.short3 = ShortTermAnalyzer(3)

        self.short5 = ShortTermAnalyzer(5)

        self.short7 = ShortTermAnalyzer(7)



        self.performance = PerformanceTracker()


        self.drift = DriftDetector()


        self.manager = DecisionManager()



        self.last_prediction_models=[]

        self.last_prediction=None



    def predict(self):


        models=[

            self.pattern_analyzer.predict(),

            self.transition_analyzer.predict(
                self.history
            ),

            self.pattern_memory.predict(
                self.history
            ),

            self.short3.predict(
                self.history
            ),

            self.short5.predict(
                self.history
            ),

            self.short7.predict(
                self.history
            )

        ]


        self.last_prediction_models=models



        result=self.manager.decide(
            models,
            self.performance
        )


        self.last_prediction = (
            result["prediction"]
        )


        return result



    def learn(self,value):


        # evaluate previous prediction

        if self.last_prediction is not None:


            correct = (
                self.last_prediction == value
            )


            self.performance.update(
                "ensemble",
                correct
            )


            self.drift.update(
                correct
            )



        # add only user input

        self.history.append(
            value
        )


        # update brains

        self.pattern_memory.learn(
            self.history
        )


        self.pattern_analyzer.learn(
            self.history
        )


        self.transition_analyzer.learn(
            self.history
        )



    def visible_history(self):

        """
        Only display last 10.
        Full history remains stored.
        """

        return self.history[-10:]



    def save(
        self,
        filename="brain_state.joblib"
    ):

        joblib.dump(
            self,
            filename
        )



    @staticmethod
    def load(
        filename="brain_state.joblib"
    ):


        if os.path.exists(filename):

            return joblib.load(
                filename
            )


        return AdaptiveEngine()