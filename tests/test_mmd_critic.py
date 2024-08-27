import unittest
from mmd_critic import MMDCritic
from mmd_critic.kernels import RBFKernel
import random

class TestMMDCritic(unittest.TestCase):
    def test_labels_match(self):
        X = [[elem] for elem in list(range(50))]
        
        # We'll set the labels here to be the same as our dataset so we know what belongs to what easily
        critic = MMDCritic(X, RBFKernel(1), labels=[item[0] for item in X])
        
        protos, proto_labels = critic.select_prototypes(5)
        criticisms, criticism_labels = critic.select_criticisms(5, protos)
        
        for x, y in zip(protos, proto_labels):
            self.assertEqual(x[0], y)
            
        for x, y in zip(criticisms, criticism_labels):
            self.assertEqual(x[0], y)
        
    def test_labels_none_when_none_set(self):
        critic = MMDCritic([[elem] for elem in list(range(50))], RBFKernel())
        
        protos, proto_labels = critic.select_prototypes(5)
        criticisms, criticism_labels = critic.select_criticisms(5, protos)
        
        self.assertIsNone(proto_labels)
        self.assertIsNone(criticism_labels)
        self.assertIsNotNone(protos)
        self.assertIsNotNone(criticisms)