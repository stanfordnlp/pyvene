import unittest
from ..utils import *


class InterventionWithLlamaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: InterventionWithLlamaTestCase ===")
        self.config, self.tokenizer, self.llama = create_llama(
            config=LlamaConfig(
                bos_token_id=1,
                eos_token_id=2,
                intermediate_size=11008,
                max_position_embeddings=1024,
                num_attention_heads=32,
                num_hidden_layers=4,
                num_key_value_heads=32,
                hidden_size=4096,
                rms_norm_eps=1e-5,
            )
        )
        self.vanilla_block_output_config = IntervenableConfig(
            model_type=type(self.llama),
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
        self.llama = self.llama.to(self.device)

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
            model_type=type(self.llama),
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
        intervenable = IntervenableModel(config, self.llama)
        intervention = list(intervenable.interventions.values())[0][0]

        base_activations = {}
        source_activations = {}
        _ = Llama_RUN(self.llama, base["input_ids"], base_activations, {})
        _ = Llama_RUN(self.llama, source["input_ids"], source_activations, {})
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
        golden_out = Llama_RUN(
            self.llama, base["input_ids"], {}, {_key: base_activations[_key]}
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

        self.assertTrue(torch.allclose(out_output[0], golden_out))


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
    suite.addTest(InterventionWithLlamaTestCase("test_clean_run_positive"))
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_invalid_unit_negative"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_single_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_multiple_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_complex_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_single_head_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_multiple_heads_positions_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_use_fast_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_location_broadcast_vanilla_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_position_intervention_constant_source_vanilla_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_position_intervention_constant_source_addition_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_position_intervention_constant_source_subtraction_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithLlamaTestCase(
            "test_with_position_intervention_constant_source_zero_intervention_positive"
        )
    ) 
    suite.addTest(
        InterventionWithLlamaTestCase(
            "_test_with_long_sequence_position_intervention_constant_source_positive"
        )
    ) 
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())