import unittest
import uuid
import os
import pyvene as pv
import torch
import shutil


class GenerationInterventionTestCase(unittest.TestCase):
    """Positive test cases for interventions on generation calls."""

    @classmethod
    def setUpClass(cls):
        _uuid = str(uuid.uuid4())[:6]
        cls._test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")

    @classmethod
    def tearDownClass(cls):
        print(f"Removing testing dir {cls._test_dir}")
        if os.path.exists(cls._test_dir) and os.path.isdir(cls._test_dir):
            shutil.rmtree(cls._test_dir)

    def test_recurrent_nn(self):

        _, _, gru = pv.create_gru_classifier(pv.GRUConfig(h_dim=32))

        pv_gru = pv.IntervenableModel(
            {
                "component": "cell_output",
                "unit": "t",
                "intervention_type": pv.ZeroIntervention,
            },
            model=gru,
        )

        rand_t = torch.rand(1, 10, gru.config.h_dim)

        intervened_outputs = pv_gru(
            base={"inputs_embeds": rand_t}, unit_locations={"base": 3}
        )

    def test_lm_generation(self):

        # built-in helper to get tinystore
        _, tokenizer, tinystory = pv.create_gpt_neo()
        emb_happy = tinystory.transformer.wte(torch.tensor(14628)) * 0.3

        pv_tinystory = pv.IntervenableModel(
            [
                {
                    "layer": _,
                    "component": "mlp_output",
                    "intervention_type": pv.AdditionIntervention,
                }
                for _ in range(tinystory.config.num_layers)
            ],
            model=tinystory,
        )

        prompt = tokenizer("Once upon a time there was", return_tensors="pt")
        _, intervened_story = pv_tinystory.generate(
            prompt, source_representations=emb_happy, max_length=32
        )
        print(tokenizer.decode(intervened_story[0], skip_special_tokens=True))
