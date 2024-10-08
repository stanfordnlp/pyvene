import unittest
from ..utils import *


class InterventionWithGPT2TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: InterventionWithGPT2TestCase ===")
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
                n_positions=1024,
                vocab_size=10,
            )
        )
        self.vanilla_block_output_config = IntervenableConfig(
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpt2 = self.gpt2.to(self.device)

        self.nonhead_streams = [
            "block_output",
            "block_input",
            "mlp_activation",
            "mlp_output",
            "mlp_input",
            "attention_value_output",
            "attention_output",
            "attention_input",
            "query_output",
            "key_output",
            "value_output",
        ]

        self.head_streams = [
            "head_attention_value_output",
            "head_query_output",
            "head_key_output",
            "head_value_output",
        ]

    def _test_with_head_position_intervention(
        self,
        intervention_layer,
        intervention_stream,
        intervention_type,
        heads=[0],
        positions=[0],
    ):
        max_position = np.max(np.array(positions))
        if isinstance(positions[0], list):
            b_s = len(positions)
        else:
            b_s = 10
        base = {
            "input_ids": torch.randint(0, 10, (b_s, max_position + 1)).to(self.device)
        }
        source = {
            "input_ids": torch.randint(0, 10, (b_s, max_position + 2)).to(self.device)
        }

        config = IntervenableConfig(
            model_type=type(self.gpt2),
            representations=[
                RepresentationConfig(
                    intervention_layer,
                    intervention_stream,
                    "h.pos",
                    len(positions),
                )
            ],
            intervention_types=intervention_type,
        )
        intervenable = IntervenableModel(config, self.gpt2)
        intervention = list(intervenable.interventions.values())[0][0]

        base_activations = {}
        source_activations = {}
        _ = GPT2_RUN(self.gpt2, base["input_ids"], base_activations, {})
        _ = GPT2_RUN(self.gpt2, source["input_ids"], source_activations, {})
        _key = f"{intervention_layer}.{intervention_stream}"
        if isinstance(heads[0], list):
            for i, head in enumerate(heads):
                for h in head:
                    for position in positions:
                        base_activations[_key][:, head, position] = intervention(
                            base_activations[_key][:, head, position],
                            source_activations[_key][:, head, position],
                        )
        else:
            for head in heads:
                for position in positions:
                    base_activations[_key][:, head, position] = intervention(
                        base_activations[_key][:, head, position],
                        source_activations[_key][:, head, position],
                    )
        golden_out = GPT2_RUN(
            self.gpt2, base["input_ids"], {}, {_key: base_activations[_key]}
        )

        if isinstance(positions[0], list):
            _, out_output = intervenable(
                base,
                [source],
                {"sources->base": ([[heads, positions]], [[heads, positions]])},
            )
        else:
            _, out_output = intervenable(
                base,
                [source],
                {
                    "sources->base": (
                        [[[heads] * b_s, [positions] * b_s]],
                        [[[heads] * b_s, [positions] * b_s]],
                    )
                },
            )
        # Relax the atol to 1e-6 to accommodate for different Transformers versions.
        # The max of the absolute diff is usually between 1e-8 to 1e-7.
        self.assertTrue(torch.allclose(out_output[0], golden_out, rtol=1e-05, atol=1e-06))


    def test_with_multiple_heads_positions_vanilla_intervention_positive(self):
        """
        Multiple head and position with vanilla intervention.
        """
        for stream in self.head_streams:
            print(f"testing stream: {stream} with multiple heads positions")
            self._test_with_head_position_intervention(
                intervention_layer=random.randint(0, 3),
                intervention_stream=stream,
                intervention_type=VanillaIntervention,
                heads=[3, 7],
                positions=[2, 6],
            )

            self._test_with_head_position_intervention(
                intervention_layer=random.randint(0, 3),
                intervention_stream=stream,
                intervention_type=VanillaIntervention,
                heads=[4, 1],
                positions=[7, 2],
            )

            
def suite():
    suite = unittest.TestSuite()
    suite.addTest(InterventionWithGPT2TestCase("test_clean_run_positive"))
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_invalid_unit_negative"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_single_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_multiple_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_complex_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_single_head_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_multiple_heads_positions_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_use_fast_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_location_broadcast_vanilla_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_position_intervention_constant_source_vanilla_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_position_intervention_constant_source_addition_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_position_intervention_constant_source_subtraction_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithGPT2TestCase(
            "test_with_position_intervention_constant_source_zero_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithGPT2TestCase(
            "_test_with_long_sequence_position_intervention_constant_source_positive"
        )
    ) 
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
