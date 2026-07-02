import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import pyvene as pv
from pyvene.models.interventions import (
    DistributedRepresentationIntervention,
    SourcelessIntervention,
    TrainableIntervention,
)


class _ReftStyleIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """A trainable, sourceless intervention (ReFT-style): a learned edit added
    to the base activation. This mirrors the setup reported in issue #231."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj = torch.nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=False)

    def forward(self, base, source=None, subspaces=None):
        return base + self.proj(base.to(self.proj.weight.dtype))


class GradientCheckpointingTestCase(unittest.TestCase):
    """Regression tests for issue #231.

    Gradient checkpointing with ``use_reentrant=False`` recomputes each layer's
    forward during ``.backward()``. Intervention hooks must stay attached through
    that recomputation; if pyvene removes them right after the forward (as it used
    to), the number of tensors saved on the forward and the recompute differ and
    PyTorch raises ``CheckpointError``.
    """

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: GradientCheckpointingTestCase ===")
        cls.device = torch.device("cpu")

    def _build(self, gradient_checkpointing, seed=0):
        torch.manual_seed(seed)
        gpt2 = GPT2LMHeadModel(
            GPT2Config(
                n_embd=24,
                n_layer=4,
                n_head=4,
                n_positions=128,
                vocab_size=64,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        if gradient_checkpointing:
            gpt2.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        config = pv.IntervenableConfig(
            model_type=type(gpt2),
            representations=[pv.RepresentationConfig(2, "block_output", "pos", 1)],
            intervention_types=_ReftStyleIntervention,
        )
        torch.manual_seed(seed + 100)  # deterministic intervention init
        pv_model = pv.IntervenableModel(config, gpt2)
        pv_model.set_device(self.device)
        # Wrapping puts the base model in eval mode; a Trainer re-enables training
        # before the loop, which is when checkpointing actually kicks in.
        gpt2.train()
        return pv_model, gpt2

    def _forward_backward(self, pv_model, input_ids):
        base = {"input_ids": input_ids, "labels": input_ids.clone()}
        _, counterfactual = pv_model(
            base, unit_locations={"base": [[[0]] * input_ids.shape[0]]}
        )
        counterfactual.loss.backward()
        return counterfactual.loss

    @staticmethod
    def _count_forward_hooks(module):
        total = len(module._forward_hooks)
        for child in module.children():
            total += GradientCheckpointingTestCase._count_forward_hooks(child)
        return total

    def test_backward_with_gradient_checkpointing(self):
        """Forward + backward under ``use_reentrant=False`` must not raise."""
        pv_model, _ = self._build(gradient_checkpointing=True)
        loss = self._forward_backward(pv_model, torch.randint(0, 64, (2, 6)))
        self.assertTrue(torch.isfinite(loss))

    def test_checkpointing_matches_non_checkpointed(self):
        """Checkpointing must not change numerics: the loss and the intervention
        gradient must match a non-checkpointed run with identical init/inputs."""
        input_ids = torch.randint(0, 64, (2, 6))

        pv_ck, _ = self._build(gradient_checkpointing=True, seed=0)
        loss_ck = self._forward_backward(pv_ck, input_ids)
        grad_ck = list(pv_ck.interventions.values())[0].proj.weight.grad.clone()

        pv_ref, _ = self._build(gradient_checkpointing=False, seed=0)
        loss_ref = self._forward_backward(pv_ref, input_ids)
        grad_ref = list(pv_ref.interventions.values())[0].proj.weight.grad.clone()

        self.assertTrue(torch.allclose(loss_ck, loss_ref, atol=1e-5))
        self.assertTrue(torch.allclose(grad_ck, grad_ref, atol=1e-5))

    def test_hooks_removed_after_backward(self):
        """Deferred teardown must still fully remove hooks and clear cached state
        once backward completes, so nothing leaks across training steps."""
        pv_model, gpt2 = self._build(gradient_checkpointing=True)
        self._forward_backward(pv_model, torch.randint(0, 64, (2, 6)))
        self.assertEqual(self._count_forward_hooks(gpt2), 0)
        self.assertEqual(len(pv_model.activations), 0)

        # a second consecutive step must also work (a Trainer runs many)
        list(pv_model.interventions.values())[0].proj.weight.grad = None
        loss_2 = self._forward_backward(pv_model, torch.randint(0, 64, (2, 6)))
        self.assertTrue(torch.isfinite(loss_2))


if __name__ == "__main__":
    unittest.main()
