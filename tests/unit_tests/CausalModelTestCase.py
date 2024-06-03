import unittest
import random
import torch
from pyvene import CausalModel
random.seed(42)


class CasualModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: CausalModelTestCase ===")
        self.variables = ['A', 'B', 'C']
        self.values = {
            'A': [False, True],
            'B': [False, True],
            'C': [False, True]
        }

        self.parents = {
            'A': [],
            'B': [],
            'C': ['A', 'B']
        }

        self.functions = {
            "A": lambda: True,
            "B": lambda: True,
            "C": lambda a, b: a and b
        }

        self.causal_model = CausalModel(
            self.variables, 
            self.values, 
            self.parents, 
            self.functions
        )
        self.causal_model.generate_equiv_classes()

    def test_initialization(self):
        inputs = ['A', 'B']
        outputs = ['C']
        timesteps = {
            'A': 0,
            'B': 0,
            'C': 1
        }
        equivalence_classes = {
            'C': {
                False: [
                    {'A': False, 'B': False},
                    {'A': False, 'B': True},
                    {'A': True, 'B': False}
                ],
                True: [
                    {'A': True, 'B': True}
                ]
            }
        }

        self.assertEqual(set(self.causal_model.inputs), set(inputs))
        self.assertEqual(set(self.causal_model.outputs), set(outputs))
        self.assertEqual(self.causal_model.timesteps, timesteps)
        self.assertEqual(self.causal_model.equiv_classes, equivalence_classes)
    
    def test_run_forward(self):
        # test run forward with default values (A and B set to True)
        self.assertEqual(
            self.causal_model.run_forward(),
            {'A': True, 'B': True, 'C': True}
        )

        # test run forward on all possible input values
        for a in [False, True]:
            for b in [False, True]:
                input_setting = {
                    'A': a,
                    'B': b
                }
                output_setting = {
                    'A': a,
                    'B': b,
                    'C': a and b
                }
                self.assertEqual(self.causal_model.run_forward(input_setting), output_setting)
        
        # test run forward on fully specified setting
        output_setting = {'A': False, 'B': False, 'C': True}
        self.assertEqual(self.causal_model.run_forward(output_setting), output_setting)
    
    def test_run_interchange(self):
        # interchange intervention on input
        base = {'A': True, 'B': False}
        source = {'A': False, 'B': True}
        self.assertEqual(self.causal_model.run_forward(base)['C'], False)
        self.assertEqual(self.causal_model.run_forward(source)['C'], False)
        self.assertEqual(
            self.causal_model.run_interchange(base, {'B': source})['C'],
            True
        )

        # interchange intervention on output
        base = {'A': False, 'B': False}
        source = {'A': True, 'B': True}
        self.assertEqual(self.causal_model.run_forward(base)['C'], False)
        self.assertEqual(
            self.causal_model.run_interchange(base, {'B': source})['C'],
            False
        )
        self.assertEqual(
            self.causal_model.run_interchange(base, {'C': source})['C'],
            True
        )

    def test_sample_input_tree_balanced(self):
        # NOTE: not quite sure how to test a function with random behavior
        # right now, fixing seed and assuming approximate behavior 
        # (taking balanced to be less than 30-70 split)

        K = 100
        # test sampling by output value
        outputs = []
        for _ in range(K):
            sample = self.causal_model.sample_input_tree_balanced()
            output = self.causal_model.run_forward(sample)
            outputs.append(output['C'])
        self.assertGreaterEqual(sum(outputs), 30)
        self.assertLessEqual(sum(outputs), 70)

        # test sampling by input value
        inputs = []
        for _ in range(K):
            sample = self.causal_model.sample_input_tree_balanced()
            inputs.append(sample['A'])
        self.assertGreaterEqual(sum(outputs), 30)
        self.assertLessEqual(sum(outputs), 70)
    
    def test_generate_factual_dataset(self):
        def sampler():
            return {'A': False, 'B': False}

        size = 4
        factual_dataset = self.causal_model.generate_factual_dataset(
            size=size,
            sampler=sampler,
            return_tensors=False
        )
        self.assertEqual(len(factual_dataset), size)

        self.assertEqual(factual_dataset[0]['input_ids'], {'A': False, 'B': False})
        self.assertEqual(factual_dataset[0]['labels']['C'], False)

        factual_dataset_tensors = self.causal_model.generate_factual_dataset(
            size=size,
            sampler=sampler,
            return_tensors=True
        )
        self.assertEqual(len(factual_dataset_tensors), size)
        X = torch.stack([example['input_ids'] for example in factual_dataset_tensors])
        y = torch.stack([example['labels'] for example in factual_dataset_tensors])
        self.assertEqual(X.shape, (size, 2))
        self.assertEqual(y.shape, (size, 1))
        self.assertTrue(torch.equal(X[0], torch.tensor([0., 0.])))
        self.assertTrue(torch.equal(y[0], torch.tensor([0.])))
    
    def test_generate_counterfactual_dataset(self):
        def sampler(*args, **kwargs):
            if kwargs.get('output_var', None):
                return {'A': True, 'B': True}

            return {'A': True, 'B': False}
        
        def intervention_sampler(*args, **kwargs):
            return {'B': True}
        
        def intervention_id(*args, **kwargs):
            return 0

        size = 4
        counterfactual_dataset = self.causal_model.generate_counterfactual_dataset(
            size=size,
            batch_size=1,
            intervention_id=intervention_id,
            sampler=sampler,
            intervention_sampler=intervention_sampler,
            return_tensors=False
        )
        self.assertEqual(len(counterfactual_dataset), size)
        example = counterfactual_dataset[0]
        self.assertEqual(example['input_ids'], {'A': True, 'B': False})
        self.assertEqual(example['source_input_ids'][0]['B'], True)
        self.assertEqual(example['intervention_id'], [0])
        self.assertEqual(example['base_labels']['C'], False) # T and F
        self.assertEqual(example['labels']['C'], True) # T and T


def suite():
    suite = unittest.TestSuite()
    suite.addTest(CasualModelTestCase("test_initialization"))
    suite.addTest(CasualModelTestCase("test_run_forward"))
    suite.addTest(CasualModelTestCase("test_run_interchange"))
    suite.addTest(CasualModelTestCase("test_sample_input_tree_balanced"))
    suite.addTest(CasualModelTestCase("test_generate_factual_dataset"))
    suite.addTest(CasualModelTestCase("test_generate_counterfactual_dataset"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
