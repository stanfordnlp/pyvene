"""
Test cases for the NDIF (nnsight) backend integration.
"""
import unittest
import torch
import os

try:
    from nnsight import LanguageModel
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False

import pyvene as pv
from pyvene.models.basic_utils import get_batch_size
from pyvene.models.interventions import (
    CollectIntervention,
    VanillaIntervention,
    AdditionIntervention,
    SubtractionIntervention,
    ZeroIntervention,
    LowRankRotatedSpaceIntervention,
    RotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
    SigmoidMaskRotatedSpaceIntervention,
    SigmoidMaskIntervention,
    PCARotatedSpaceIntervention,
)

def get_remote_clean_output(model, base_tokens):
    """Helper function to get clean output from a remote NNsight model."""
    # Convert base_tokens to raw tensors to avoid serialization issues
    if hasattr(base_tokens, 'keys') and hasattr(base_tokens, 'values'):
        base_tokens_raw = {k: v for k, v in base_tokens.items()}
    else:
        base_tokens_raw = base_tokens

    with model.session(remote=True):
        with model.trace(base_tokens_raw):
            clean_output = model.output.save()
    return clean_output


@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
class NdifBackendTestCase(unittest.TestCase):
    """Test NDIF backend with local execution."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifBackendTestCase ===")
        cls.model = LanguageModel('openai-community/gpt2', device_map='cpu')
        cls.tokenizer = cls.model.tokenizer
        cls.base_tokens = cls.tokenizer("The capital of France is", return_tensors="pt")
        cls.source_tokens = cls.tokenizer("The capital of Germany is", return_tensors="pt")
        cls.seq_len = cls.base_tokens['input_ids'].shape[1]

    def test_collect_intervention(self):
        """Test CollectIntervention collects activations."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].mlp.c_proj.output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=False)

        result = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )

        _, collected = result[0]
        self.assertIsInstance(collected, list)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].shape, (self.seq_len, 768))

    def test_vanilla_intervention(self):
        """Test VanillaIntervention swaps activations."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": VanillaIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_generate_with_intervention(self):
        """Test generation with CollectIntervention."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=False)

        _, gen_output = pv_model.generate(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))},
            max_new_tokens=3,
        )
        self.assertIsNotNone(gen_output)

    def test_clean_run(self):
        """Test clean run produces correct output."""
        pv_model = pv.build_intervenable_model([], model=self.model, remote=False)

        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                expected = self.model.output.save()

        pv_output, _ = pv_model(base=self.base_tokens)

        self.assertTrue(torch.allclose(expected.logits, pv_output.logits, atol=1e-5))

    def test_addition_intervention(self):
        """Test AdditionIntervention adds source to base."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": AdditionIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change due to addition
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_subtraction_intervention(self):
        """Test SubtractionIntervention subtracts source from base."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": SubtractionIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change due to subtraction
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_zero_intervention(self):
        """Test ZeroIntervention zeros out activations."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].mlp.c_proj.output",
            "intervention": ZeroIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Apply zero intervention (no sources needed)
        _, intervened = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )

        # Output should change due to zeroed activations
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
class NdifBackendCorrectnessTestCase(unittest.TestCase):
    """Check the ndif-local backend against trusted references on GPT-2.

    These assert exact equality (torch.allclose) vs a plain HuggingFace forward
    and native pyvene, rather than just "the output changed". Remote reuses the
    same math, so verifying it locally on a small model is enough.
    """

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifBackendCorrectnessTestCase ===")
        from transformers import GPT2LMHeadModel
        cls.model = LanguageModel('openai-community/gpt2', device_map='cpu')
        cls.tokenizer = cls.model.tokenizer
        cls.base_tokens = cls.tokenizer("The capital of France is", return_tensors="pt")
        cls.source_tokens = cls.tokenizer("The capital of Germany is", return_tensors="pt")
        cls.seq_len = cls.base_tokens['input_ids'].shape[1]
        cls.hf = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').eval()

    def _hf_forward(self):
        with torch.no_grad():
            return self.hf(**self.base_tokens, output_hidden_states=True)

    def test_clean_logits_match_huggingface(self):
        """NDIF-local clean logits are numerically identical to a plain HF forward."""
        hf_out = self._hf_forward()
        with self.model.trace(self.base_tokens):
            pv_logits = self.model.lm_head.output.save()
        pv_logits = pv_logits.logits if hasattr(pv_logits, "logits") else pv_logits
        self.assertTrue(
            torch.allclose(pv_logits, hf_out.logits, atol=1e-4),
            f"max abs diff {(pv_logits - hf_out.logits).abs().max().item()}"
        )

    def test_collected_activation_matches_huggingface(self):
        """CollectIntervention at transformer.h[0].output == HF hidden_states[1]."""
        hf_out = self._hf_forward()
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=False)
        result = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )
        act = result[0][-1][0]
        if hasattr(act, "value"):
            act = act.value
        # HF hidden_states[1] is (batch, seq, hidden); collected is (seq, hidden)
        self.assertTrue(
            torch.allclose(act, hf_out.hidden_states[1][0], atol=1e-4),
            f"max abs diff {(act - hf_out.hidden_states[1][0]).abs().max().item()}"
        )

    def test_vanilla_interchange_matches_native_pyvene(self):
        """Single-position interchange yields the same prediction as native pyvene."""
        POS = 3  # token that differs between base/source (France vs Germany)

        # NDIF-local backend: interchange at transformer.h[0].output
        pv_ndif = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": VanillaIntervention()
        }, model=self.model, remote=False)
        _, v_out = pv_ndif(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([[[POS]]], [[[POS]]])}
        )
        v_logits = v_out.logits if hasattr(v_out, "logits") else v_out
        ndif_pred = v_logits[0, -1].argmax().item()

        # Native pyvene reference on the equivalent component (block_output, layer 0)
        _, tok_nat, gpt2 = pv.create_gpt2()
        cfg = pv.IntervenableConfig(
            {"layer": 0, "component": "block_output"},
            intervention_types=pv.VanillaIntervention,
        )
        pv_nat = pv.IntervenableModel(cfg, gpt2)
        _, nat_out = pv_nat(
            tok_nat("The capital of France is", return_tensors="pt"),
            [tok_nat("The capital of Germany is", return_tensors="pt")],
            unit_locations={"sources->base": POS},
        )
        nat_logits = torch.matmul(nat_out.last_hidden_state, gpt2.wte.weight.t())
        nat_pred = nat_logits[0, -1].argmax().item()

        self.assertEqual(ndif_pred, nat_pred)

    def test_serial_single_position_matches_native_pyvene(self):
        """Serial (chained) single-position interchange must match native pyvene,
        i.e. the serial path honors unit_locations rather than patching whole
        activations."""
        POS = 3

        cfg = pv.IntervenableConfig(
            representations=[
                {"layer": 0, "component": "transformer.h[0].output"},
                {"layer": 2, "component": "transformer.h[2].output"},
            ],
            intervention_types=VanillaIntervention,
            mode="serial",
        )
        pv_ndif = pv.build_intervenable_model(cfg, model=self.model, remote=False)
        s1 = self.tokenizer("The capital of Germany is", return_tensors="pt")
        s2 = self.tokenizer("The capital of Italy is", return_tensors="pt")
        _, out = pv_ndif(
            base=self.base_tokens,
            sources=[s1, s2],
            unit_locations={
                "source_0->source_1": ([[[POS]]], [[[POS]]]),
                "source_1->base": ([[[POS]]], [[[POS]]]),
            },
        )
        logits = out.logits if hasattr(out, "logits") else out
        ndif_pred = logits[0, -1].argmax().item()

        _, tok_nat, gpt2 = pv.create_gpt2()
        ncfg = pv.IntervenableConfig(
            [{"layer": 0, "component": "block_output"},
             {"layer": 2, "component": "block_output"}],
            intervention_types=pv.VanillaIntervention,
            mode="serial",
        )
        pv_nat = pv.IntervenableModel(ncfg, gpt2)
        _, nat_out = pv_nat(
            tok_nat("The capital of France is", return_tensors="pt"),
            [tok_nat("The capital of Germany is", return_tensors="pt"),
             tok_nat("The capital of Italy is", return_tensors="pt")],
            unit_locations={"source_0->source_1": POS, "source_1->base": POS},
        )
        nat_logits = torch.matmul(nat_out.last_hidden_state, gpt2.wte.weight.t())
        nat_pred = nat_logits[0, -1].argmax().item()

        self.assertEqual(ndif_pred, nat_pred)


class BasicUtilsNdifTestCase(unittest.TestCase):
    """Test basic_utils changes for NDIF support."""

    def test_get_batch_size_string(self):
        """Test get_batch_size with string input."""
        self.assertEqual(get_batch_size("Hello world"), 1)

    def test_get_batch_size_list_of_strings(self):
        """Test get_batch_size with list of strings."""
        self.assertEqual(get_batch_size(["Hello", "World"]), 2)


REMOTE_MODEL = "meta-llama/Llama-3.1-8B"  # a model NDIF keeps pinned


def _ndif_model_available(repo_id):
    """True if repo_id is currently running and pinned on NDIF."""
    try:
        import requests
        from nnsight import CONFIG
        host = CONFIG.API.HOST
        if not host.startswith("http"):
            host = "https://" + host
        deployments = requests.get(f"{host}/status", timeout=30).json().get("deployments", {})
        return any(
            isinstance(v, dict) and v.get("repo_id") == repo_id
            and v.get("application_state") == "RUNNING" and v.get("pinned")
            for v in deployments.values()
        )
    except Exception:
        return False


@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
@unittest.skipUnless(os.environ.get('NDIF_REMOTE_TESTS') == '1',
                     "Remote tests disabled. Set NDIF_REMOTE_TESTS=1 to enable.")
class NdifBackendRemoteTestCase(unittest.TestCase):
    """Test NDIF backend with remote execution (remote=True)."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifBackendRemoteTestCase ===")
        if not _ndif_model_available(REMOTE_MODEL):
            raise unittest.SkipTest(
                f"{REMOTE_MODEL} is not currently pinned/running on NDIF")
        cls.model = LanguageModel(REMOTE_MODEL)
        cls.tokenizer = cls.model.tokenizer
        # raw strings for remote calls (BatchEncoding isn't on NDIF's allowlist)
        cls.base_tokens = "The capital of France is"
        cls.source_tokens = "The capital of Germany is"
        cls.seq_len = len(cls.tokenizer(cls.base_tokens)["input_ids"])

    def test_remote_collect_intervention(self):
        """Test CollectIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].mlp.output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=True)

        result = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )
        _, collected = result[0]
        self.assertIsInstance(collected, list)
        self.assertEqual(len(collected), 1)

    def test_remote_vanilla_intervention(self):
        """Test VanillaIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": VanillaIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_vanilla_single_position_matches_raw_nnsight(self):
        """A single-position remote interchange patches only that position and
        matches a hand-written raw nnsight interchange (guards against replacing
        the whole activation when unit_locations name one token)."""
        POS = 3  # France vs Germany differ here

        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": VanillaIntervention()
        }, model=self.model, remote=True)
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([[[POS]]], [[[POS]]])}
        )

        # Raw nnsight reference: swap only position POS at layers[0].output.
        # Bind locals so the remote closure doesn't try to serialize the
        # (non-allowlisted) TestCase instance.
        model = self.model
        base_tokens = self.base_tokens
        source_tokens = self.source_tokens
        with model.session(remote=True):
            with model.trace(source_tokens):
                src_act = model.model.layers[0].output.save()
            with model.trace(base_tokens):
                act = model.model.layers[0].output
                patched = act.clone()
                patched[:, POS, :] = src_act[:, POS, :]
                act[:] = patched
                ref_logits = model.lm_head.output.save()

        self.assertTrue(
            torch.allclose(intervened['logits'], ref_logits, atol=1e-3),
            f"max abs diff {(intervened['logits'] - ref_logits).abs().max().item()}"
        )

    def test_remote_generate(self):
        """Test generation with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=True)

        _, gen_output = pv_model.generate(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))},
            max_new_tokens=3,
        )
        self.assertIsNotNone(gen_output)

    def test_remote_generate_single_position_matches_raw_nnsight(self):
        """generate() must honor unit_locations: a single-position swap should
        match a raw nnsight generate that patches only that position at prefill."""
        POS = 3

        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": VanillaIntervention()
        }, model=self.model, remote=True)
        _, gen_output = pv_model.generate(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([[[POS]]], [[[POS]]])},
            max_new_tokens=3,
            do_sample=False,
        )
        pv_ids = gen_output[0] if isinstance(gen_output, (list, tuple)) else gen_output

        model = self.model
        base_tokens = self.base_tokens
        source_tokens = self.source_tokens
        with model.session(remote=True):
            with model.trace(source_tokens):
                src_act = model.model.layers[0].output.save()
            with model.generate(base_tokens, max_new_tokens=3, do_sample=False):
                act = model.model.layers[0].output
                patched = act.clone()
                patched[:, POS, :] = src_act[:, POS, :]
                act[:] = patched
                ref_ids = model.generator.output.save()

        self.assertTrue(torch.equal(pv_ids, ref_ids))

    def test_remote_addition_intervention(self):
        """Test AdditionIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": AdditionIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_subtraction_intervention(self):
        """Test SubtractionIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": SubtractionIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_zero_intervention(self):
        """Test ZeroIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].mlp.output",
            "intervention": ZeroIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Apply zero intervention
        _, intervened = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
class NdifTrainableInterventionTestCase(unittest.TestCase):
    """Test trainable interventions with NDIF backend."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifTrainableInterventionTestCase ===")
        cls.model = LanguageModel('openai-community/gpt2', device_map='cpu')
        cls.tokenizer = cls.model.tokenizer
        cls.base_tokens = cls.tokenizer("The capital of France is", return_tensors="pt")
        cls.source_tokens = cls.tokenizer("The capital of Germany is", return_tensors="pt")
        cls.seq_len = cls.base_tokens['input_ids'].shape[1]
        cls.embed_dim = 768  # GPT-2's hidden size

    def test_low_rank_rotated_intervention_local(self):
        """Test LowRankRotatedSpaceIntervention with local nnsight."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            # rotated-space interventions operate on a fixed-width vector,
            # so we must target a single position (not all positions).
            # Position 3 is where base/source differ (France vs Germany).
            unit_locations={"sources->base": ([[[3]]], [[[3]]])}
        )

        # Output should change due to rotation intervention
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_rotated_space_intervention_local(self):
        """Test RotatedSpaceIntervention with local nnsight."""
        intervention = RotatedSpaceIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            # single position required for fixed-width rotation;
            # position 3 is where base/source differ (France vs Germany)
            unit_locations={"sources->base": ([[[3]]], [[[3]]])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_sigmoid_mask_intervention_local(self):
        """Test SigmoidMaskIntervention with local nnsight."""
        intervention = SigmoidMaskIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_get_remote_weights_low_rank(self):
        """Test get_remote_weights returns correct structure for LowRankRotatedSpaceIntervention."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        weights = intervention.get_remote_weights()

        self.assertIn('rotate_layer_weight', weights)
        self.assertIn('embed_dim', weights)
        self.assertIn('low_rank_dimension', weights)
        self.assertIn('intervention_type', weights)
        self.assertEqual(weights['intervention_type'], 'low_rank_rotated_space')
        self.assertEqual(weights['rotate_layer_weight'].shape, (self.embed_dim, 64))

    def test_get_remote_weights_rotated_space(self):
        """Test get_remote_weights returns correct structure for RotatedSpaceIntervention."""
        intervention = RotatedSpaceIntervention(embed_dim=self.embed_dim)
        weights = intervention.get_remote_weights()

        self.assertIn('rotate_layer_weight', weights)
        self.assertIn('intervention_type', weights)
        self.assertEqual(weights['intervention_type'], 'rotated_space')
        self.assertEqual(weights['rotate_layer_weight'].shape, (self.embed_dim, self.embed_dim))

    def test_gradient_flow_through_intervention(self):
        """Verify gradients flow through trainable intervention parameters."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Ensure intervention parameters require gradients
        self.assertTrue(intervention.trainable)
        has_grad_params = any(p.requires_grad for p in intervention.parameters())
        self.assertTrue(has_grad_params, "Intervention should have gradient-enabled parameters")

        # Forward pass
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            # single position required for fixed-width rotation
            unit_locations={"sources->base": ([[[0]]], [[[0]]])}
        )

        # Compute loss and backward (using a simple sum loss)
        # Note: This tests that the output is connected to the intervention params
        if hasattr(intervened, 'logits'):
            loss = intervened.logits.sum()
        else:
            loss = intervened.sum()
        loss.backward()

        # Check gradients exist on intervention parameters
        for name, param in intervention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Gradient should exist for parameter {name}"
                )

    def test_forward_with_gradients_method(self):
        """forward_with_gradients keeps the graph connected to the live
        intervention parameters, so loss.backward() populates their grads."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        output = pv_model.forward_with_gradients(
            base=self.base_tokens,
            sources=[self.source_tokens],
            # single position for fixed-width rotation; position 3 (France vs
            # Germany) differs between base and source so the grad is non-zero
            unit_locations={"sources->base": ([[[3]]], [[[3]]])}
        )

        self.assertIsNotNone(output)

        logits = output.logits if hasattr(output, 'logits') else output
        loss = logits.sum()
        loss.backward()

        # The rotation parameters must receive gradients; if the path used
        # detached weights (get_remote_weights), grad would stay None.
        grad_params = [
            (name, p) for name, p in intervention.named_parameters()
            if p.requires_grad
        ]
        self.assertTrue(grad_params, "intervention should expose trainable params")
        for name, param in grad_params:
            self.assertIsNotNone(param.grad, f"no gradient for {name}")
            self.assertGreater(param.grad.abs().sum().item(), 0.0,
                               f"zero gradient for {name}")


@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
@unittest.skipUnless(os.environ.get('NDIF_REMOTE_TESTS') == '1',
                     "Remote tests disabled. Set NDIF_REMOTE_TESTS=1 to enable.")
class NdifTrainableInterventionRemoteTestCase(unittest.TestCase):
    """Test trainable interventions with NDIF backend and remote=True."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifTrainableInterventionRemoteTestCase ===")
        if not _ndif_model_available(REMOTE_MODEL):
            raise unittest.SkipTest(
                f"{REMOTE_MODEL} is not currently pinned/running on NDIF")
        cls.model = LanguageModel(REMOTE_MODEL)
        cls.tokenizer = cls.model.tokenizer
        # raw strings for remote calls (BatchEncoding isn't on NDIF's allowlist)
        cls.base_tokens = "The capital of France is"
        cls.source_tokens = "The capital of Germany is"
        cls.seq_len = len(cls.tokenizer(cls.base_tokens)["input_ids"])
        cls.embed_dim = 4096

    def test_remote_low_rank_rotated_intervention(self):
        """Test LowRankRotatedSpaceIntervention with remote=True."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": intervention
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_serial_single_position(self):
        """Serial (chained) interventions run remotely and honor a single
        target position without falling back to whole-activation patching."""
        cfg = pv.IntervenableConfig(
            representations=[
                {"layer": 0, "component": "model.layers[0].output"},
                {"layer": 2, "component": "model.layers[2].output"},
            ],
            intervention_types=VanillaIntervention,
            mode="serial",
        )
        # position 4 is the country token (<BOS> The capital of France is),
        # which differs across base/sources so the swap actually changes output
        POS = 4
        pv_model = pv.build_intervenable_model(cfg, model=self.model, remote=True)
        clean_output = get_remote_clean_output(self.model, self.base_tokens)
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens, "The capital of Italy is"],
            unit_locations={
                "source_0->source_1": ([[[POS]]], [[[POS]]]),
                "source_1->base": ([[[POS]]], [[[POS]]]),
            },
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_pca_rotated_intervention(self):
        """PCARotatedSpaceIntervention isn't a TrainableIntervention but exposes
        get_remote_weights(); the remote backend must still bucket and apply it
        rather than returning the base trace unchanged."""
        import numpy as np
        n_comp = 64
        rng = np.random.RandomState(0)
        pca = type("PCA", (), {
            "components_": rng.randn(n_comp, self.embed_dim).astype("float32")
        })()
        pca_mean = rng.randn(self.embed_dim).astype("float32")
        pca_std = (np.abs(rng.randn(self.embed_dim)) + 1.0).astype("float32")
        intervention = PCARotatedSpaceIntervention(
            embed_dim=self.embed_dim, pca=pca, pca_mean=pca_mean, pca_std=pca_std
        )
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": intervention
        }, model=self.model, remote=True)

        clean_output = get_remote_clean_output(self.model, self.base_tokens)
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_rotated_space_intervention(self):
        """Test RotatedSpaceIntervention with remote=True."""
        intervention = RotatedSpaceIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": intervention
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_sigmoid_mask_intervention(self):
        """Test SigmoidMaskIntervention with remote=True."""
        intervention = SigmoidMaskIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "model.layers[0].output",
            "intervention": intervention
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BasicUtilsNdifTestCase))
    if NNSIGHT_AVAILABLE:
        suite.addTest(unittest.makeSuite(NdifBackendTestCase))
        suite.addTest(unittest.makeSuite(NdifTrainableInterventionTestCase))
        if os.environ.get('NDIF_REMOTE_TESTS') == '1':
            suite.addTest(unittest.makeSuite(NdifBackendRemoteTestCase))
            suite.addTest(unittest.makeSuite(NdifTrainableInterventionRemoteTestCase))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())