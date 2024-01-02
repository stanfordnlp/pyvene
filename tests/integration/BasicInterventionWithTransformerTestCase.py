import unittest
from utils import *

class BasicInterventionWithTransformerTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: BasicInterventionWithTransformerTestCase ===")
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
        self.vanilla_block_output_alignable_config = AlignableConfig(
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.nonhead_streams = [
            "block_output", "block_input", 
            "mlp_activation", "mlp_output", "mlp_input",
            "attention_value_output", "attention_output", "attention_input",
            "query_output", "key_output", "value_output"
        ]
        
    def test_clean_run_positive(self):
        """
        Positive test case to check whether vanilla forward pass work
        with our object.
        """
        alignable = AlignableModel(
            self.vanilla_block_output_alignable_config, self.gpt2)
        alignable.set_device(self.device)
        base = {"input_ids": torch.randint(0, 10, (10, 5)).to(self.device)}
        golden_out = self.gpt2(**base).logits
        our_output = alignable(base)[0][0]        
        self.assertTrue(torch.allclose(
            golden_out, our_output))
        # make sure the toolkit also works
        self.assertTrue(torch.allclose(
            GPT2_RUN(self.gpt2, base["input_ids"], {}, {}), golden_out))
        
    def _test_with_single_position_intervention(self, _intervention_type):
        """
        Cover different streams here.
        """
        pass
    
    def _test_with_position_intervention(
        self,
        intervention_layer, intervention_stream, intervention_type, 
        positions=[0],
    ):
        max_position = np.max(np.array(positions))
        b_s = 10
        base = {"input_ids": torch.randint(0, 10, (b_s, max_position+1)).to(self.device)}
        source = {"input_ids": torch.randint(0, 10, (b_s, max_position+2)).to(self.device)}
        
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    intervention_layer,
                    intervention_stream,
                    "pos",
                    len(positions),
                )
            ],
            alignable_interventions_type=intervention_type,
        )
        alignable = AlignableModel(
            alignable_config, self.gpt2)
        intervention = list(alignable.interventions.values())[0][0]

        base_activations = {}
        source_activations = {}
        _ = GPT2_RUN(self.gpt2, base["input_ids"], base_activations, {})
        _ = GPT2_RUN(self.gpt2, source["input_ids"], source_activations, {})
        _key = f"{intervention_layer}.{intervention_stream}"
        if isinstance(positions[0], list):
            for i, position in enumerate(positions):
                for pos in position:
                    base_activations[_key][i,pos] = intervention(
                            base_activations[_key][i,pos].unsqueeze(dim=0), 
                            source_activations[_key][i,pos].unsqueeze(dim=0)
                        ).squeeze(dim=0)
        else:
            for position in positions:
                base_activations[_key][:,position] = intervention(
                    base_activations[_key][:,position],
                    source_activations[_key][:,position]
                )
        golden_out = GPT2_RUN(self.gpt2, base["input_ids"], {}, {
            _key: base_activations[_key]
        })
        
        if isinstance(positions[0], list):
            _, out_output = alignable(
                base,
                [source],
                {"sources->base": ([positions], [positions])}
            )
        else:
            _, out_output = alignable(
                base,
                [source],
                {"sources->base": ([[positions]*b_s], [[positions]*b_s])}
            )
        
        self.assertTrue(torch.allclose(out_output[0], golden_out))
        
        
    def test_with_single_position_intervention(self):

        for stream in self.nonhead_streams:
            print(f"testing stream: {stream} with a single position")
            self._test_with_position_intervention(
                intervention_layer=0, 
                intervention_stream=stream, 
                intervention_type=VanillaIntervention, 
                positions=[0]
            )
            
    def test_with_multiple_position_intervention(self):

        for stream in self.nonhead_streams:
            print(f"testing stream: {stream} with multiple positions")
            self._test_with_position_intervention(
                intervention_layer=0, 
                intervention_stream=stream, 
                intervention_type=VanillaIntervention, 
                positions=[1,3]
            )
            
            
def suite():
    suite = unittest.TestSuite()
    suite.addTest(BasicInterventionWithTransformerTestCase('test_clean_run_positive'))
    suite.addTest(BasicInterventionWithTransformerTestCase('test_with_single_position_intervention'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())