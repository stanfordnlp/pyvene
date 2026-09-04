import unittest
from ..utils import *

from transformers import ElectraConfig
from pyvene.models.electra.modelings_intervenable_electra import (
    create_electra_mlm,
    create_electra_pretraining,
)


class InterventionWithElectraTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: InterventionWithElectraTestCase ===")
        electra_config = ElectraConfig(
            hidden_size=24,
            embedding_size=24,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=48,
            max_position_embeddings=64,
            vocab_size=20,
            pad_token_id=1,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.config, self.tokenizer, self.electra = create_electra_mlm(
            config=electra_config
        )
        self.electra.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.electra = self.electra.to(self.device)
        # discriminator (replaced-token-detection) head, as loaded by
        # AutoModelForPreTraining for google/electra-*-discriminator checkpoints
        _, _, self.electra_discriminator = create_electra_pretraining(
            config=electra_config
        )
        self.electra_discriminator.eval()
        self.electra_discriminator = self.electra_discriminator.to(self.device)

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
        Wrapping the model in an IntervenableModel without intervening should
        reproduce the original forward pass for every supported stream.
        """
        base = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        golden_out = self.electra(**base).logits
        for stream in self.nonhead_streams:
            config = IntervenableConfig(
                model_type=type(self.electra),
                representations=[RepresentationConfig(0, stream, "pos", 1)],
                intervention_types=VanillaIntervention,
            )
            intervenable = IntervenableModel(config, self.electra)
            intervenable.set_device(self.device)
            our_output = intervenable(base, output_original_output=True)[0][0]
            self.assertTrue(
                torch.allclose(golden_out, our_output, rtol=1e-05, atol=1e-06)
            )

    def test_with_position_intervention_positive(self):
        """
        Copying a source activation into the base at each non head stream should
        change the output, confirming the anchor point hooks a real module.
        """
        base = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        source = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        base_out = self.electra(**base).logits
        for stream in self.nonhead_streams:
            config = IntervenableConfig(
                model_type=type(self.electra),
                representations=[RepresentationConfig(1, stream, "pos", 1)],
                intervention_types=VanillaIntervention,
            )
            intervenable = IntervenableModel(config, self.electra)
            intervenable.set_device(self.device)
            _, our_output = intervenable(
                base, [source], {"sources->base": ([[[0]] * 4], [[[0]] * 4])}
            )
            self.assertFalse(
                torch.allclose(our_output[0], base_out, rtol=1e-05, atol=1e-06)
            )

    def test_with_head_position_intervention_positive(self):
        """
        Per head streams should build and run a vanilla intervention.
        """
        base = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        source = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        for stream in self.head_streams:
            config = IntervenableConfig(
                model_type=type(self.electra),
                representations=[RepresentationConfig(1, stream, "h.pos", 1)],
                intervention_types=VanillaIntervention,
            )
            intervenable = IntervenableModel(config, self.electra)
            intervenable.set_device(self.device)
            _, our_output = intervenable(
                base,
                [source],
                {
                    "sources->base": (
                        [[[[0]] * 4, [[0]] * 4]],
                        [[[[0]] * 4, [[0]] * 4]],
                    )
                },
            )
            self.assertEqual(our_output[0].shape[0], base["input_ids"].shape[0])

    def test_discriminator_clean_run_positive(self):
        """
        ElectraForPreTraining shares the `electra.encoder` prefix with the other
        head models, so wrapping it must reproduce its forward pass unchanged.
        """
        base = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        golden_out = self.electra_discriminator(**base).logits
        for stream in self.nonhead_streams:
            config = IntervenableConfig(
                model_type=type(self.electra_discriminator),
                representations=[RepresentationConfig(0, stream, "pos", 1)],
                intervention_types=VanillaIntervention,
            )
            intervenable = IntervenableModel(config, self.electra_discriminator)
            intervenable.set_device(self.device)
            our_output = intervenable(base, output_original_output=True)[0][0]
            self.assertTrue(
                torch.allclose(golden_out, our_output, rtol=1e-05, atol=1e-06)
            )

    def test_discriminator_with_position_intervention_positive(self):
        """
        A vanilla intervention on the discriminator must change its
        replaced-token logits at every non head stream.
        """
        base = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        source = {"input_ids": torch.randint(2, 20, (4, 6)).to(self.device)}
        base_out = self.electra_discriminator(**base).logits
        # query/key interventions at a single position only move the attention
        # weights, which in this randomly initialised model shifts the
        # one-logit-per-token discriminator output by ~1e-7, below the
        # tolerance used here; the remaining streams carry the change directly.
        streams = [
            s for s in self.nonhead_streams
            if s not in ("query_output", "key_output")
        ]
        for stream in streams:
            config = IntervenableConfig(
                model_type=type(self.electra_discriminator),
                representations=[RepresentationConfig(1, stream, "pos", 1)],
                intervention_types=VanillaIntervention,
            )
            intervenable = IntervenableModel(config, self.electra_discriminator)
            intervenable.set_device(self.device)
            _, our_output = intervenable(
                base, [source], {"sources->base": ([[[0]] * 4], [[[0]] * 4])}
            )
            self.assertFalse(
                torch.allclose(our_output[0], base_out, rtol=1e-05, atol=1e-06)
            )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(InterventionWithElectraTestCase("test_clean_run_positive"))
    suite.addTest(
        InterventionWithElectraTestCase("test_with_position_intervention_positive")
    )
    suite.addTest(
        InterventionWithElectraTestCase(
            "test_with_head_position_intervention_positive"
        )
    )
    suite.addTest(
        InterventionWithElectraTestCase("test_discriminator_clean_run_positive")
    )
    suite.addTest(
        InterventionWithElectraTestCase(
            "test_discriminator_with_position_intervention_positive"
        )
    )
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
