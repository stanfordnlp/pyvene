import unittest
from ..utils import *


class IntervenableConfigUnitTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: IntervenableConfigUnitTestCase ===")
        self.config, self.tokenizer, self.gpt2 = create_gpt2_lm(
            config=GPT2Config(
                n_embd=24,
                attn_pdrop=0.0,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
                summary_first_dropout=0.0,
                n_layer=4,
                bos_token_id=0,
                eos_token_id=0,
                n_positions=128,
                vocab_size=10,
            )
        )

    def test_initialization_positive(self):
        config = IntervenableConfig(
            model_type=type(self.gpt2),
            representations=[
                RepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervention_types=VanillaIntervention,
        )

        assert config.model_type == type(self.gpt2)
        assert len(config.representations) == 1
        assert (
            config.intervention_types == VanillaIntervention
        )

        assert (
            config.representations[0].layer == 0
        )
        assert (
            config.representations[
                0
            ].component
            == "block_output"
        )
        assert (
            config.representations[0].unit
            == "pos"
        )
        assert (
            config.representations[0].max_number_of_units == 1
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(IntervenableConfigUnitTestCase("test_initialization_positive"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
