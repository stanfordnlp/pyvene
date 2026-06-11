import unittest
from ..utils import *

from transformers import T5Config
from pyvene.models.t5.modelings_intervenable_t5 import (
    create_t5_encoder,
    create_t5_lm,
)


class InterventionWithT5TestCase(unittest.TestCase):
    """Smoke tests for the T5 anchor mapping.

    These verify that every advertised anchor in
    ``t5_type_to_module_mapping`` (a) resolves to a real submodule on a
    small randomly-initialised T5 and (b) actually changes the model's
    output when intervened on. Tests run on CPU using a tiny config so
    no checkpoint download is required.
    """

    @classmethod
    def setUpClass(self):
        print("=== Test Suite: InterventionWithT5TestCase ===")
        self.config, _, self.t5_encoder = create_t5_encoder(
            config=T5Config(
                d_model=24,
                d_ff=48,
                d_kv=6,
                num_layers=4,
                num_decoder_layers=4,
                num_heads=4,
                vocab_size=20,
                pad_token_id=0,
                eos_token_id=1,
                decoder_start_token_id=0,
                dropout_rate=0.0,
            )
        )
        self.t5_encoder.eval()
        _, _, self.t5_lm = create_t5_lm(
            config=T5Config(
                d_model=24,
                d_ff=48,
                d_kv=6,
                num_layers=4,
                num_decoder_layers=4,
                num_heads=4,
                vocab_size=20,
                pad_token_id=0,
                eos_token_id=1,
                decoder_start_token_id=0,
                dropout_rate=0.0,
            )
        )
        self.t5_lm.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_encoder = self.t5_encoder.to(self.device)
        self.t5_lm = self.t5_lm.to(self.device)

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

    def _zero_intervention_changes_output_encoder(self, stream):
        """Patching with activations from a different source input at a
        single position should perturb the encoder's last_hidden_state."""
        b_s, seq_len = 2, 6
        base_ids = torch.randint(2, 20, (b_s, seq_len)).to(self.device)
        src_ids = torch.randint(2, 20, (b_s, seq_len)).to(self.device)
        base_mask = torch.ones_like(base_ids).to(self.device)
        src_mask = torch.ones_like(src_ids).to(self.device)

        config = IntervenableConfig(
            model_type=type(self.t5_encoder),
            representations=[
                RepresentationConfig(0, stream, "pos", 1),
            ],
            intervention_types=VanillaIntervention,
        )
        intervenable = IntervenableModel(config, self.t5_encoder)

        with torch.no_grad():
            base_out = self.t5_encoder(
                input_ids=base_ids, attention_mask=base_mask
            ).last_hidden_state
            _, intervened = intervenable(
                base={"input_ids": base_ids, "attention_mask": base_mask},
                sources=[
                    {"input_ids": src_ids, "attention_mask": src_mask}
                ],
                unit_locations={"sources->base": ([[[2]] * b_s], [[[2]] * b_s])},
            )
        self.assertFalse(
            torch.allclose(base_out, intervened.last_hidden_state),
            f"Intervention on stream '{stream}' did not change encoder output.",
        )

    def test_nonhead_streams_encoder(self):
        for stream in self.nonhead_streams:
            with self.subTest(stream=stream):
                self._zero_intervention_changes_output_encoder(stream)

    def test_head_streams_encoder(self):
        for stream in self.head_streams:
            with self.subTest(stream=stream):
                config = IntervenableConfig(
                    model_type=type(self.t5_encoder),
                    representations=[
                        RepresentationConfig(0, stream, "h.pos", 1),
                    ],
                    intervention_types=VanillaIntervention,
                )
                # Constructing the IntervenableModel resolves every
                # anchor path; if the head split is wrong it raises here.
                IntervenableModel(config, self.t5_encoder)

    def test_lm_encoder_anchors_resolve(self):
        """T5ForConditionalGeneration uses the same encoder-rooted anchor
        paths as the bare T5EncoderModel; verify they still resolve."""
        for stream in self.nonhead_streams:
            with self.subTest(stream=stream):
                config = IntervenableConfig(
                    model_type=type(self.t5_lm),
                    representations=[
                        RepresentationConfig(0, stream, "pos", 1),
                    ],
                    intervention_types=VanillaIntervention,
                )
                IntervenableModel(config, self.t5_lm)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(InterventionWithT5TestCase("test_nonhead_streams_encoder"))
    suite.addTest(InterventionWithT5TestCase("test_head_streams_encoder"))
    suite.addTest(InterventionWithT5TestCase("test_lm_encoder_anchors_resolve"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
