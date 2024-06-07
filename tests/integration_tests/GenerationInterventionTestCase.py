import unittest
import uuid
import os
import pyvene as pv
import torch
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GenerationInterventionTestCase(unittest.TestCase):
    """Positive test cases for interventions on generation calls."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: GenerationInterventionTestCase ===")
        _uuid = str(uuid.uuid4())[:6]
        cls._test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        cls.device = DEVICE

        cls.config, cls.tokenizer, cls.tinystory = pv.create_gpt_neo()
        cls.tinystory.to(cls.device)

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

        # built-in helper to get tinystory
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
            prompt, source_representations=emb_happy, unit_locations={"sources->base": (0, [0, 1, 2])}, max_length=32
        )
        print(tokenizer.decode(intervened_story[0], skip_special_tokens=True))

    def test_generation_with_source_intervened_prompt(self):
        torch.manual_seed(0)

        pv_model = pv.IntervenableModel(
            [
                {
                    "layer": l,
                    "component": "mlp_output",
                    "intervention": lambda b, s: b + s * 0.5,
                }
                for l in range(self.config.num_layers)
            ],
            model=self.tinystory,
        )

        prompt = self.tokenizer("Once upon a time there was", return_tensors="pt").to(
            self.device
        )
        orig, intervened = pv_model.generate(
            prompt,
            max_length=32,
            sources=self.tokenizer("Happy love", return_tensors="pt").to(self.device),
            intervene_on_prompt=True,
            unit_locations={"sources->base": 0},
            output_original_output=True,
        )
        orig_text, intervened_text = (
            self.tokenizer.decode(orig[0], skip_special_tokens=True),
            self.tokenizer.decode(intervened[0], skip_special_tokens=True),
        )

        print(orig_text)
        print(intervened_text)
        assert (
            orig_text != intervened_text
        ), "Aggressive intervention did not change the output. Probably something wrong."

    def test_dynamic_static_generation_intervention_parity(self):
        torch.manual_seed(1)

        pv_model = pv.IntervenableModel(
            [
                {
                    "layer": l,
                    "component": "mlp_output",
                    "intervention": lambda b, s: torch.ones_like(b),
                }
                for l in range(self.config.num_layers)
            ],
            model=self.tinystory,
        )

        prompt = self.tokenizer("Once upon a time there was", return_tensors="pt").to(
            self.device
        )
        INTERVENTION_DELAY = 5

        orig, intervened = pv_model.generate(
            prompt,
            max_length=prompt.input_ids.shape[1] + INTERVENTION_DELAY + 2,
            timestep_selector=[lambda idx, o: idx == INTERVENTION_DELAY]
            * self.config.num_layers,
            output_original_output=True,
        )
        orig_text, intervened_text = (
            self.tokenizer.decode(orig[0], skip_special_tokens=True),
            self.tokenizer.decode(intervened[0], skip_special_tokens=True),
        )

        print(orig_text)
        print(intervened_text)
        assert (
            orig_text != intervened_text
        ), "Aggressive intervention did not change the output. Probably something wrong."

    def test_generation_noops(self):
        torch.manual_seed(0)

        # No-op intervention
        pv_model = pv.IntervenableModel(
            [
                {
                    "layer": l,
                    "component": "mlp_output",
                    "intervention": lambda b, s: b,
                }
                for l in range(self.config.num_layers)
            ],
            model=self.tinystory,
        )

        prompt = self.tokenizer("Once upon a time there was", return_tensors="pt").to(
            self.device
        )
        sources = self.tokenizer(" love", return_tensors="pt").to(self.device)

        orig, intervened = pv_model.generate(
            prompt,
            max_length=20,
            sources=sources,
            intervene_on_prompt=True,
            unit_locations={"sources->base": (0, [0, 1, 2])},
            output_original_output=True,
        )
        orig_text, intervened_text = (
            self.tokenizer.decode(orig[0], skip_special_tokens=True),
            self.tokenizer.decode(intervened[0], skip_special_tokens=True),
        )

        print(intervened_text)
        assert (
            orig_text == intervened_text
        ), "No-op intervention changed the output. Probably something wrong."

        # Aggressive intervention with intervene_on_prompt=False
        aggressive_model = pv.IntervenableModel(
            [
                {
                    "layer": l,
                    "component": "mlp_output",
                    "intervention": lambda b, s: s * 1000,
                }
                for l in range(self.config.num_layers)
            ],
            model=self.tinystory,
        )

        orig, intervened = aggressive_model.generate(
            prompt,
            max_length=20,
            sources=sources,
            intervene_on_prompt=False,
            output_original_output=True,
        )        

        orig_text, intervened_text = (
            self.tokenizer.decode(orig[0], skip_special_tokens=True),
            self.tokenizer.decode(intervened[0], skip_special_tokens=True),
        )
        print(orig_text)
        print(intervened_text)
        assert (
            orig_text == intervened_text
        ), "Aggressive intervention changed the output. Probably something wrong."

        # Aggressive intervention with no prompt intervention, disabled selectors
        orig, intervened = aggressive_model.generate(
            prompt,
            max_length=20,
            sources=sources,
            intervene_on_prompt=False,
            output_original_output=True,
            timestep_selector=[lambda idx, o: False] * self.config.num_layers,
        )
        orig_text, intervened_text = (
            self.tokenizer.decode(orig[0], skip_special_tokens=True),
            self.tokenizer.decode(intervened[0], skip_special_tokens=True),
        )
        assert (
            orig_text == intervened_text
        ), "Aggressive intervention changed the output. Probably something wrong."

if __name__ == "__main__":
    unittest.main()
