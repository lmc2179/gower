import gower
import unittest
import numpy as np

class GowerTest(unittest.TestCase):
    def test_real(self):
        X = np.array([[0],
                      [1],
                      [2]])
        T = ['R']
        S = gower.similarity(X, T)
        expected_sim = [[1, 0.5, 0],
                        [0.5, 1, 0.5],
                        [0, 0.5, 1]]
        self.assertEqual(S.tolist(), expected_sim)

    def test_categorical(self):
        X = np.array([[0],
                      [1],
                      [2]])
        T = ['C']
        S = gower.similarity(X, T)
        expected_sim = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
        self.assertEqual(S.tolist(), expected_sim)
        
    def test_gower_similarity(self):
        X = np.array([[0, 0, 1],
                      [1, 1, 1],
                      [2, 0, 2]])
        T = ['R', 'B', 'C']
        S = gower.similarity(X, T)
        expected_sim = [[1, 1./2, 0],
                        [1./2, 1, 1./6],
                        [0, 1./6, 1]]
        self.assertEqual(S.tolist(), expected_sim)
        
    def test_gower_distance(self):
        X = np.array([[0, 0, 1],
                      [1, 1, 1],
                      [2, 0, 2]])
        T = ['R', 'B', 'C']
        D = gower.distance(X, T)
        expected_dis = [[0, 1./2, 1.],
                        [1./2, 0, 5./6],
                        [1., 5./6, 0]]
        self.assertEqual(D.tolist(), expected_dis)
        
        
if __name__ == '__main__':
    unittest.main()
