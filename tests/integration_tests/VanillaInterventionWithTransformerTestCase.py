import unittest
from ..utils import *


class VanillaInterventionWithTransformerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: VanillaInterventionWithTransformerTestCase ===")
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
        self.vanilla_block_output_intervenable_config = IntervenableConfig(
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

    def test_clean_run_positive(self):
        """
        Positive test case to check whether vanilla forward pass work
        with our object.
        """
        intervenable = IntervenableModel(
            self.vanilla_block_output_intervenable_config, self.gpt2
        )
        intervenable.set_device(self.device)
        base = {"input_ids": torch.randint(0, 10, (10, 5)).to(self.device)}
        golden_out = self.gpt2(**base).logits
        our_output = intervenable(base)[0][0]
        self.assertTrue(torch.allclose(golden_out, our_output))
        # make sure the toolkit also works
        self.assertTrue(
            torch.allclose(GPT2_RUN(self.gpt2, base["input_ids"], {}, {}), golden_out)
        )

    def test_invalid_intervenable_unit_negative(self):
        """
        Invalid intervenable unit.
        """
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos.h",
                    1,
                ),
            ],
            intervenable_interventions_type=VanillaIntervention,
        )
        try:
            intervenable = IntervenableModel(intervenable_config, self.gpt2)
        except ValueError:
            pass
        else:
            raise ValueError("ValueError for invalid intervenable unit is not thrown")

    def _test_with_position_intervention(
        self,
        intervention_layer,
        intervention_stream,
        intervention_type,
        positions=[0],
        use_fast=False,
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

        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    intervention_layer,
                    intervention_stream,
                    "pos",
                    len(positions),
                )
            ],
            intervenable_interventions_type=intervention_type,
        )
        intervenable = IntervenableModel(
            intervenable_config, self.gpt2, use_fast=use_fast
        )
        intervention = list(intervenable.interventions.values())[0][0]

        base_activations = {}
        source_activations = {}
        _ = GPT2_RUN(self.gpt2, base["input_ids"], base_activations, {})
        _ = GPT2_RUN(self.gpt2, source["input_ids"], source_activations, {})
        _key = f"{intervention_layer}.{intervention_stream}"
        if isinstance(positions[0], list):
            for i, position in enumerate(positions):
                for pos in position:
                    base_activations[_key][i, pos] = intervention(
                        base_activations[_key][i, pos].unsqueeze(dim=0),
                        source_activations[_key][i, pos].unsqueeze(dim=0),
                    ).squeeze(dim=0)
        else:
            for position in positions:
                base_activations[_key][:, position] = intervention(
                    base_activations[_key][:, position],
                    source_activations[_key][:, position],
                )
        golden_out = GPT2_RUN(
            self.gpt2, base["input_ids"], {}, {_key: base_activations[_key]}
        )

        if isinstance(positions[0], list):
            _, out_output = intervenable(
                base, [source], {"sources->base": ([positions], [positions])}
            )
        else:
            _, out_output = intervenable(
                base,
                [source],
                {"sources->base": ([[positions] * b_s], [[positions] * b_s])},
            )

        self.assertTrue(torch.allclose(out_output[0], golden_out))

    def test_with_single_position_vanilla_intervention_positive(self):
        """
        A single position with vanilla intervention.
        """
        for stream in self.nonhead_streams:
            print(f"testing stream: {stream} with a single position")
            self._test_with_position_intervention(
                intervention_layer=random.randint(0, 3),
                intervention_stream=stream,
                intervention_type=VanillaIntervention,
                positions=[0],
            )

    def test_with_multiple_position_vanilla_intervention_positive(self):
        """
        Multiple positions with vanilla intervention.
        """
        for stream in self.nonhead_streams:
            print(f"testing stream: {stream} with multiple positions")
            self._test_with_position_intervention(
                intervention_layer=random.randint(0, 3),
                intervention_stream=stream,
                intervention_type=VanillaIntervention,
                positions=[1, 3],
            )

    def test_with_complex_position_vanilla_intervention_positive(self):
        """
        Complex positions with vanilla intervention.
        """
        for stream in self.nonhead_streams:
            print(f"testing stream: {stream} with complex positions")
            self._test_with_position_intervention(
                intervention_layer=random.randint(0, 3),
                intervention_stream=stream,
                intervention_type=VanillaIntervention,
                positions=[[1, 3], [2, 5], [8, 1]],
            )

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

        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    intervention_layer,
                    intervention_stream,
                    "h.pos",
                    len(positions),
                )
            ],
            intervenable_interventions_type=intervention_type,
        )
        intervenable = IntervenableModel(intervenable_config, self.gpt2)
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

        self.assertTrue(torch.allclose(out_output[0], golden_out))

    def test_with_single_head_position_vanilla_intervention_positive(self):
        """
        Single head and position with vanilla intervention.
        """
        for stream in self.head_streams:
            print(f"testing stream: {stream} with single head position")
            self._test_with_head_position_intervention(
                intervention_layer=random.randint(0, 3),
                intervention_stream=stream,
                intervention_type=VanillaIntervention,
                heads=[random.randint(0, 8)],
                positions=[random.randint(0, 8)],
            )

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

    def test_with_use_fast_vanilla_intervention_positive(self):
        """
        Enable use_fast with vanilla intervention.
        """
        for stream in self.nonhead_streams:
            print(f"testing stream: {stream} with a single position")
            self._test_with_position_intervention(
                intervention_layer=random.randint(0, 3),
                intervention_stream=stream,
                intervention_type=VanillaIntervention,
                positions=[0],
                use_fast=True,
            )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(VanillaInterventionWithTransformerTestCase("test_clean_run_positive"))
    suite.addTest(
        VanillaInterventionWithTransformerTestCase(
            "test_invalid_intervenable_unit_negative"
        )
    )
    suite.addTest(
        VanillaInterventionWithTransformerTestCase(
            "test_with_single_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        VanillaInterventionWithTransformerTestCase(
            "test_with_multiple_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        VanillaInterventionWithTransformerTestCase(
            "test_with_complex_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        VanillaInterventionWithTransformerTestCase(
            "test_with_single_head_position_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        VanillaInterventionWithTransformerTestCase(
            "test_with_multiple_heads_positions_vanilla_intervention_positive"
        )
    )
    suite.addTest(
        VanillaInterventionWithTransformerTestCase(
            "test_with_use_fast_vanilla_intervention_positive"
        )
    )
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
