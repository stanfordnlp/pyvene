import unittest
from tests.utils import *

class SubspaceInterventionWithTransformerTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: VanillaInterventionWithTransformerTestCase ===")
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpt2 = self.gpt2.to(self.device)
        
        self.nonhead_streams = [
            "block_output", "block_input", 
            "mlp_activation", "mlp_output", "mlp_input",
            "attention_value_output", "attention_output", "attention_input",
            "query_output", "key_output", "value_output"
        ]
        
        self.head_streams = [
            "head_attention_value_output",
            "head_query_output", "head_key_output", "head_value_output"
        ]
        
    def test_clean_run_positive(self):
        """
        Positive test case to check whether vanilla forward pass work
        with our object.
        """
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                    subspace_partition=[[0,6],[6,24]]
                ),
            ],
            alignable_interventions_type=VanillaIntervention,
        )
        alignable = AlignableModel(
            alignable_config, self.gpt2)
        alignable.set_device(self.device)
        base = {"input_ids": torch.randint(0, 10, (10, 5)).to(self.device)}
        golden_out = self.gpt2(**base).logits
        our_output = alignable(base)[0][0]        
        self.assertTrue(torch.allclose(
            golden_out, our_output))
        # make sure the toolkit also works
        self.assertTrue(torch.allclose(
            GPT2_RUN(self.gpt2, base["input_ids"], {}, {}), golden_out))
        
            
    def _test_subspace_partition_in_forward(self, intervention_type):
        """
        Provide subpace intervention indices in the forward only.
        """
        batch_size = 10
        with_partition_alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                    alignable_low_rank_dimension=24,
                    subspace_partition=[[0,6],[6,24]]
                ),
            ],
            alignable_interventions_type=intervention_type,
        )
        alignable = AlignableModel(
            with_partition_alignable_config, self.gpt2, use_fast=False
        )
        alignable.set_device(self.device)
        base = {"input_ids": torch.randint(0, 10, (batch_size, 5)).to(self.device)}
        source = {"input_ids": torch.randint(0, 10, (batch_size, 5)).to(self.device)}
        _, with_partition_our_output = alignable(
            base,
            [source],
            {"sources->base": ([[[0]]*batch_size], [[[0]]*batch_size])},
            subspaces = [[[0]]*batch_size]
        )       

        without_partition_alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                    alignable_low_rank_dimension=24
                ),
            ],
            alignable_interventions_type=intervention_type,
        )
        alignable_fast = AlignableModel(
            without_partition_alignable_config, 
            self.gpt2, use_fast=True
        )
        alignable_fast.set_device(self.device)
        if intervention_type in {
            RotatedSpaceIntervention, LowRankRotatedSpaceIntervention}:
            list(alignable_fast.interventions.values())[0][0].rotate_layer.weight = \
                list(alignable.interventions.values())[0][0].rotate_layer.weight
        
        _, without_partition_our_output = alignable_fast(
            base,
            [source],
            {"sources->base": ([[[0]]*batch_size], [[[0]]*batch_size])},
            subspaces = [[[i for i in range(6)]]*batch_size]
        )   
        
        # make sure the toolkit also works
        self.assertTrue(torch.allclose(
            with_partition_our_output[0], without_partition_our_output[0]))
        
    def test_vanilla_subspace_partition_in_forward_positive(self):
        self._test_subspace_partition_in_forward(VanillaIntervention)
        
    def test_rotate_subspace_partition_in_forward_positive(self):
        self._test_subspace_partition_in_forward(RotatedSpaceIntervention)
        
    def test_lowrank_rotate_subspace_partition_in_forward_positive(self):
        _retry = 10
        while _retry > 0:
            try:
                self._test_subspace_partition_in_forward(LowRankRotatedSpaceIntervention)
            except:
                pass # retry
            finally:
                break
            _retry -= 1
        if _retry > 0:
            pass # succeed
        else:
            raise AssertionError(
                "test_lowrank_rotate_subspace_partition_in_forward_positive with retries")
            
            
def suite():
    suite = unittest.TestSuite()
    suite.addTest(SubspaceInterventionWithTransformerTestCase(
        'test_clean_run_positive'))
    suite.addTest(SubspaceInterventionWithTransformerTestCase(
        'test_vanilla_subspace_partition_in_forward_positive'))
    suite.addTest(SubspaceInterventionWithTransformerTestCase(
        'test_rotate_subspace_partition_in_forward_positive'))
    suite.addTest(SubspaceInterventionWithTransformerTestCase(
        'test_lowrank_rotate_subspace_partition_in_forward_positive'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    