import unittest
from ..utils import *

import torch
import pyvene as pv

class InterventionWithGPT2TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        _uuid = str(uuid.uuid4())[:6]
        self._test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")

    def test_lazy_demo(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        pv_gpt2 = pv.IntervenableModel({
            "layer": 0,
            "component": "mlp_output",
            "source_representation": torch.zeros(
                gpt2.config.n_embd)
        }, model=gpt2)

        intervened_outputs = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            unit_locations={"base": 3}
        )

    def test_less_lazy_demo(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            {
                "layer": _,
                "component": "mlp_output",
                "source_representation": torch.zeros(
                    gpt2.config.n_embd)
            } for _ in range(4)],
            mode="parallel"
        )
        print(config)
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        intervened_outputs = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            unit_locations={"base": 3}
        )

    def test_less_lazy_demo(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            {
                "layer": _,
                "component": "mlp_output",
                "source_representation": torch.zeros(
                    gpt2.config.n_embd)
            } for _ in range(4)],
            mode="parallel"
        )
        print(config)
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        intervened_outputs = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            unit_locations={"base": 3}
        )

    def test_source_reprs_pass_in_unit_loc_broadcast_demo(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        pv_gpt2 = pv.IntervenableModel({
            "layer": 0,
            "component": "mlp_output",
        }, model=gpt2)

        intervened_outputs = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            source_representations = torch.zeros(gpt2.config.n_embd),
            unit_locations={"base": 3}
        )

    def test_input_corrupt_multi_token(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig({
            "layer": 0,
            "component": "mlp_input"},
            pv.AdditionIntervention
        )

        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        intervened_outputs = pv_gpt2(
            base = tokenizer(
                "The Space Needle is in downtown", 
                return_tensors="pt"
            ), 
            unit_locations={"base": [[[0, 1, 2, 3]]]},
            source_representations = torch.rand(gpt2.config.n_embd)
        )

    def test_trainable_backward(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig({
            "layer": 8,
            "component": "block_output",
            "low_rank_dimension": 1},
            pv.LowRankRotatedSpaceIntervention
        )

        pv_gpt2 = pv.IntervenableModel(
            config, model=gpt2)

        last_hidden_state = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            sources = tokenizer(
                "The capital of Italy is", 
                return_tensors="pt"
            ), 
            unit_locations={"sources->base": 3}
        )[-1].last_hidden_state

        loss = last_hidden_state.sum()
        loss.backward()

    def test_reprs_collection(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig({
            "layer": 10,
            "component": "block_output",
            "intervention_type": pv.CollectIntervention}
        )

        pv_gpt2 = pv.IntervenableModel(
            config, model=gpt2)

        collected_activations = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            unit_locations={"sources->base": 3}
        )[0][-1]

    def test_reprs_collection_after_intervention(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig({
            "layer": 8,
            "component": "block_output",
            "intervention_type": pv.VanillaIntervention}
        )

        config.add_intervention({
            "layer": 10,
            "component": "block_output",
            "intervention_type": pv.CollectIntervention})

        pv_gpt2 = pv.IntervenableModel(
            config, model=gpt2)

        collected_activations = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            sources = [tokenizer(
                "The capital of Italy is", 
                return_tensors="pt"
            ), None], 
            unit_locations={"sources->base": 3}
        )[0][-1]

    def test_reprs_collection_on_one_neuron(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig({
            "layer": 8,
            "component": "head_attention_value_output",
            "unit": "h.pos",
            "intervention_type": pv.CollectIntervention}
        )

        pv_gpt2 = pv.IntervenableModel(
            config, model=gpt2)

        collected_activations = pv_gpt2(
            base = tokenizer(
                "The capital of Spain is", 
                return_tensors="pt"
            ), 
            unit_locations={
                "base": pv.GET_LOC((3,3))
            },
            subspaces=[0]
        )[0][-1]

    def test_new_intervention_type(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        class MultiplierIntervention(
        pv.ConstantSourceIntervention):
            def __init__(self, embed_dim, **kwargs):
                super().__init__()
            def forward(
            self, base, source=None, subspaces=None):
                return base * 99.0
        # run with new intervention type
        pv_gpt2 = pv.IntervenableModel({
        "intervention_type": MultiplierIntervention}, 
        model=gpt2)
        intervened_outputs = pv_gpt2(
        base = tokenizer("The capital of Spain is", 
            return_tensors="pt"), 
        unit_locations={"base": 3})

    def test_recurrent_nn(self):

        _, _, gru = pv.create_gru_classifier(
            pv.GRUConfig(h_dim=32))

        pv_gru = pv.IntervenableModel({
            "component": "cell_output",
            "unit": "t", 
            "intervention_type": pv.ZeroIntervention},
            model=gru)

        rand_t = torch.rand(1,10, gru.config.h_dim)

        intervened_outputs = pv_gru(
        base = {"inputs_embeds": rand_t}, 
        unit_locations={"base": 3})

    def test_lm_generation(self):

        # built-in helper to get tinystore
        _, tokenizer, tinystory = pv.create_gpt_neo()
        emb_happy = tinystory.transformer.wte(
            torch.tensor(14628)) * 0.3

        pv_tinystory = pv.IntervenableModel([{
            "layer": _,
            "component": "mlp_output",
            "intervention_type": pv.AdditionIntervention
            } for _ in range(
                tinystory.config.num_layers)],
            model=tinystory)

        prompt = tokenizer(
            "Once upon a time there was", 
            return_tensors="pt")
        _, intervened_story = pv_tinystory.generate(
            prompt,
            source_representations=emb_happy,
            max_length=32
        )
        print(tokenizer.decode(
            intervened_story[0], 
            skip_special_tokens=True
        ))

    def test_save_and_load(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        class MultiplierIntervention(
        pv.SourcelessIntervention):
            def __init__(self, embed_dim, **kwargs):
                super().__init__()
                self.register_buffer(
                    'interchange_dim', 
                    torch.tensor(embed_dim))
            def forward(
            self, base, source=None, subspaces=None):
                return base * 99.0
            def __str__(self):
                return f"MultiplierIntervention()"
        # run with new intervention type
        pv_gpt2 = pv.IntervenableModel({
        "intervention_type": MultiplierIntervention}, 
        model=gpt2)

        pv_gpt2.save(self._test_dir)

        pv_gpt2_load = pv.IntervenableModel.load(
            self._test_dir,
            model=gpt2)

    @classmethod
    def tearDownClass(self):
        print(f"Removing testing dir {self._test_dir}")
        if os.path.exists(self._test_dir) and os.path.isdir(self._test_dir):
            shutil.rmtree(self._test_dir)