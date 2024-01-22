import unittest
from ..utils import *
from pyvene.models.modeling_utils import *


class ModelUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_gather_neurons_positive(self):
        tensor_input = torch.rand((5, 3, 2))  # batch_size, seq_len, emb_dim
        tensor_output = gather_neurons(tensor_input, "pos", [[0, 1]] * 5)
        self.assertTrue(torch.allclose(tensor_output, tensor_input[:, 0:2, :]))
        tensor_output = gather_neurons(tensor_input, "h", [[0, 1]] * 5)
        self.assertTrue(torch.allclose(tensor_output, tensor_input[:, 0:2, :]))

    def test_gather_neurons_pos_h_positive(self):
        tensor_input = torch.rand((5, 4, 3, 2))  # batch_size, #heads, seq_len, emb_dim
        tensor_output = gather_neurons(
            tensor_input, "h.pos", ([[1, 2]] * 5, [[0, 1]] * 5)
        )
        self.assertTrue(torch.allclose(tensor_output, tensor_input[:, 1:3, 0:2, :]))

    def _test_gather_neurons_negative(self, name, unit, expected_error_msg):
        tensor_input = torch.rand((5, 3, 2))
        with self.assertRaisesRegex(AssertionError, expected_error_msg):
            gather_neurons(tensor_input, unit, [[0, 1]] * 5)

    def test_gather_neurons_negative(self):
        self._test_gather_neurons_negative(
            "dim",
            "dim",
            "Not Implemented Gathering with Unit = dim",
        )
        self._test_gather_neurons_negative(
            "pos.dim",
            "pos.dim",
            "Not Implemented Gathering with Unit = pos.dim",
        )
        self._test_gather_neurons_negative(
            "h.dim", "h.dim", "Not Implemented Gathering with Unit = h.dim"
        )
        self._test_gather_neurons_negative(
            "h.pos.dim", "h.pos.dim", "Not Implemented Gathering with Unit = h.pos.dim"
        )

    def test_scatter_neurons_positive(self):
        # batch_size, seq_len, emb_dim
        tensor_input = torch.arange(60).view(2, 5, 6)
        # batch_size, #heads, seq_len, emb_dim
        replacing_tensor_input = torch.arange(60, 96).view(2, 3, 3, 2)

        # Replace the heads 1, 2 at positions 0, 1 with the first
        golden_output = tensor_input.clone().view((2, 5, 3, 2))
        golden_output[:, 0:2, 1:3, :] = replacing_tensor_input[:, 0:2, 0:2, :].permute(
            0, 2, 1, 3
        )

        tensor_output = scatter_neurons(
            tensor_input,
            replacing_tensor_input,
            "head_attention_value_output",
            "h.pos",
            ([[1, 2]] * 2, [[0, 1]] * 2),
            hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel,
            GPT2Config(
                n_embd=6,
                n_head=3,
                attn_pdrop=0.0,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
                summary_first_dropout=0.0,
                n_layer=4,
                bos_token_id=0,
                eos_token_id=0,
                n_positions=20,
                vocab_size=10,
            ),
            False,
        )
        tensor_output = tensor_output.view((2, 5, 3, 2))
        self.assertTrue(torch.allclose(tensor_output, golden_output))

    def test_scatter_gathered_neurons_positive(self):
        # batch_size, #heads, seq_len, emb_dim
        replacing_tensor_input = torch.rand((2, 3, 5, 2))
        gathered_replacing_tensor_input = gather_neurons(
            replacing_tensor_input, "h.pos", ([[1, 2]] * 2, [[0, 1]] * 2)
        )
        # batch_size, seq_len, emb_dim
        tensor_input = torch.rand((2, 5, 6))
        golden_output = tensor_input.clone().view((2, 5, 3, 2))
        golden_output[:, 0:2, 1:3, :] = gathered_replacing_tensor_input.permute(
            0, 2, 1, 3
        )
        tensor_output = scatter_neurons(
            tensor_input,
            gathered_replacing_tensor_input,
            "head_attention_value_output",
            "h.pos",
            ([[1, 2]] * 2, [[0, 1]] * 2),
            hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel,
            GPT2Config(
                n_embd=6,
                n_head=3,
                attn_pdrop=0.0,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
                summary_first_dropout=0.0,
                n_layer=4,
                bos_token_id=0,
                eos_token_id=0,
                n_positions=20,
                vocab_size=10,
            ),
            False,
        )
        tensor_output = tensor_output.view((2, 5, 3, 2))
        self.assertTrue(torch.allclose(tensor_output, golden_output))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ModelUtilsTestCase("test_gather_neurons_pos_h_positive"))
    suite.addTest(ModelUtilsTestCase("test_gather_neurons_positive"))
    suite.addTest(ModelUtilsTestCase("test_gather_neurons_negative"))
    suite.addTest(ModelUtilsTestCase("test_scather_neurons_positive"))
    suite.addTest(ModelUtilsTestCase("test_scatter_gathered_neurons_positive"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
