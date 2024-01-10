import unittest
from tests.utils import *


class AlignableConfigUnitTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: AlignableConfigUnitTestCase ===")
        self.config, self.tokenizer, self.gpt2 = create_gpt2_lm(
            config = GPT2Config(
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
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            alignable_interventions_type=VanillaIntervention,
        )
        
        assert alignable_config.alignable_model_type == \
            type(self.gpt2)
        assert len(alignable_config.alignable_representations) == 1
        assert alignable_config.alignable_interventions_type == \
            VanillaIntervention
    
        assert alignable_config.alignable_representations[0].alignable_layer == 0
        assert \
            alignable_config.alignable_representations[0].alignable_representation_type == \
            "block_output"
        assert alignable_config.alignable_representations[0].alignable_unit == "pos"
        assert alignable_config.alignable_representations[0].max_number_of_units == 1
    
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(AlignableConfigUnitTestCase(
        'test_initialization_positive'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())