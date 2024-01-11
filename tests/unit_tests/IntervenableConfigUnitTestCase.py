import unittest
from tests.utils import *


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
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervenable_interventions_type=VanillaIntervention,
        )

        assert intervenable_config.intervenable_model_type == type(self.gpt2)
        assert len(intervenable_config.intervenable_representations) == 1
        assert (
            intervenable_config.intervenable_interventions_type == VanillaIntervention
        )

        assert (
            intervenable_config.intervenable_representations[0].intervenable_layer == 0
        )
        assert (
            intervenable_config.intervenable_representations[
                0
            ].intervenable_representation_type
            == "block_output"
        )
        assert (
            intervenable_config.intervenable_representations[0].intervenable_unit
            == "pos"
        )
        assert (
            intervenable_config.intervenable_representations[0].max_number_of_units == 1
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(IntervenableConfigUnitTestCase("test_initialization_positive"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
