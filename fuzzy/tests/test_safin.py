import unittest
import numpy as np

from sklearn.datasets import fetch_california_housing

from fuzzy.self_adaptive.safin import SaFIN

class SaFINTestCase(unittest.TestCase):
    def test_california_housing(self):
        california = fetch_california_housing()
        NUM_DATA = 40
        train_X = california.data[:NUM_DATA]
        train_Y = np.array([california.target]).T[:NUM_DATA]
        test_X = california.data[NUM_DATA:]
        test_Y = np.array([california.target]).T[NUM_DATA:]
        safin = SaFIN(alpha=0.2, beta=0.6)
        rmse_before_prune = safin.fit(train_X, train_Y, batch_size=50, epochs=10, verbose=False, rule_pruning=False)
        predicted_Y = safin.predict(train_X)

        expected_output = np.array([4.2652532 , 3.40350801, 3.3249243 , 2.62624563, 2.5731876 ,
                                       2.5731876 , 2.61270487, 2.56706278, 2.55425098, 2.62469887,
                                       2.57281928, 2.57281928, 2.57281928, 2.44164476, 2.48825806,
                                       2.3938432 , 2.57213973, 2.54372859, 2.50043997, 2.55249994,
                                       2.46904895, 2.48544908, 2.41401713, 2.40191396, 2.5662137 ,
                                       2.47833507, 2.51422226, 2.18670187, 2.38259013, 2.39539047,
                                       2.48156144, 2.36039846, 2.2148645 , 2.30798991, 2.56609274,
                                       2.3406255 , 2.18601593, 2.30737073, 2.56667218, 2.56667218])
        self.assertEqual(35, len(safin.rules))  # make sure the rule generation procedure has not changed
        self.assertEqual(np.around(0.9644943548896574, 6), np.around(rmse_before_prune, 6))  # make sure the overall performance has not changed
        self.assertTrue((np.around(expected_output, 6) == np.around(predicted_Y.flatten(), 6)).all())  # make sure the fuzzy inference has not changed

        #  make sure the weight matrices are the correct shapes
        self.assertEqual((8, 25), safin.W_1.shape)
        self.assertEqual((25, 35), safin.W_2.shape)
        self.assertEqual((35, 5), safin.W_3.shape)
        self.assertEqual((5, 1), safin.W_4.shape)


if __name__ == '__main__':
    unittest.main()
