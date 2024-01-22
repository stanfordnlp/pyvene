import unittest
from ..utils import *

class ModelUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass
    
    def test_gather_neurons_positive(self):
        tensor_input = torch.rand((5, 3, 2)) # batch_size, seq_len, emb_dim
        tensor_output = gather_neurons(tensor_input, "pos", [[0,1]] * 5)
        self.assertTrue(torch.allclose(tensor_output, tensor_input[:, 0:2, :]))
        tensor_output = gather_neurons(tensor_input, "h", [[0,1]] * 5)
        self.assertTrue(torch.allclose(tensor_output, tensor_input[:, 0:2, :]))
        
    def test_gather_neurons_pos_h_positive(self):
        tensor_input = torch.rand((5, 4, 3, 2)) # batch_size, #heads, seq_len, emb_dim
        tensor_output = gather_neurons(tensor_input, "h.pos", ([[1,2]] * 5, [[0,1]] * 5))
        self.assertTrue(torch.allclose(tensor_output, tensor_input[:, 1:3, 0:2, :]))
    
    def _test_gather_neurons_negative(self, name, unit, expected_error_msg):
        tensor_input = torch.rand((5, 3, 2))
        with self.assertRaisesRegex(AssertionError, expected_error_msg):
            gather_neurons(tensor_input, unit, [[0,1]] * 5)
    
    def test_gather_neurons_negative(self):
        self._test_gather_neurons_negative("dim", "dim", "Not Implemented Gathering with Unit = dim",)
        self._test_gather_neurons_negative("pos.dim", "pos.dim", "Not Implemented Gathering with Unit = pos.dim",)
        self._test_gather_neurons_negative("h.dim", "h.dim", "Not Implemented Gathering with Unit = h.dim")
        self._test_gather_neurons_negative("h.pos.dim", "h.pos.dim", "Not Implemented Gathering with Unit = h.pos.dim")
        

def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        ModelUtilsTestCase(
            "test_gather_neurons_pos_h_positive"
        )
    )
    suite.addTest(
        ModelUtilsTestCase(
            "test_gather_neurons_positive"
        )
    )
    suite.addTest(
        ModelUtilsTestCase(
            "test_gather_neurons_negative"
        )
    )
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
