import unittest
from ..utils import *

import copy
import torch
import pyvene as pv

class IntervenableBasicTestCase(unittest.TestCase):
    """These are API level positive cases."""
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

        # run with new intervention type
        pv_gpt2 = pv.IntervenableModel({
        "intervention_type": pv.ZeroIntervention}, 
        model=gpt2)

        pv_gpt2.save(self._test_dir)

        pv_gpt2_load = pv.IntervenableModel.load(
            self._test_dir,
            model=gpt2)
        
    def test_intervention_grouping(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            {"layer": 0, "component": "block_output", "group_key": 0},
            {"layer": 2, "component": "block_output", "group_key": 0}],
            intervention_types=pv.VanillaIntervention,
        )

        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        sources = [tokenizer("The capital of Italy is", return_tensors="pt")]
        intervened_outputs = pv_gpt2(
            base, sources, 
            {"sources->base": ([
                [[3]], [[4]] # these two are for two interventions
            ], [             # source position 3 into base position 4
                [[3]], [[4]] 
            ])}
        )
        
    def test_intervention_skipping(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            # these are equivalent interventions
            # we create them on purpose
            {"layer": 0, "component": "block_output"},
            {"layer": 0, "component": "block_output"},
            {"layer": 0, "component": "block_output"}],
            intervention_types=pv.VanillaIntervention,
        )
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        source = tokenizer("The capital of Italy is", return_tensors="pt")
        # skipping 1, 2 and 3
        _, pv_out1 = pv_gpt2(base, [None, None, source],
            {"sources->base": ([None, None, [[4]]], [None, None, [[4]]])})
        _, pv_out2 = pv_gpt2(base, [None, source, None],
            {"sources->base": ([None, [[4]], None], [None, [[4]], None])})
        _, pv_out3 = pv_gpt2(base, [source, None, None],
            {"sources->base": ([[[4]], None, None], [[[4]], None, None])})
        # should have the same results
        self.assertTrue(torch.equal(pv_out1.last_hidden_state, pv_out2.last_hidden_state))
        self.assertTrue(torch.equal(pv_out2.last_hidden_state, pv_out3.last_hidden_state))
       
    def test_subspace_intervention(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            # they are linked to manipulate the same representation
            # but in different subspaces
            {"layer": 0, "component": "block_output",
             # subspaces can be partitioned into continuous chunks
             # [i, j] are the boundary indices
             "subspace_partition": [[0, 128], [128, 256]]}],
            intervention_types=pv.VanillaIntervention,
        )
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        source = tokenizer("The capital of Italy is", return_tensors="pt")

        # using intervention skipping for subspace
        intervened_outputs = pv_gpt2(
            base, [source],
            {"sources->base": 4},
            # intervene only only dimensions from 128 to 256
            subspaces=1,
        )
    
    def test_linked_intervention_and_weights_sharing(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            # they are linked to manipulate the same representation
            # but in different subspaces
            {"layer": 0, "component": "block_output", 
             "subspace_partition": [[0, 128], [128, 256]], "intervention_link_key": 0},
            {"layer": 0, "component": "block_output",
             "subspace_partition": [[0, 128], [128, 256]], "intervention_link_key": 0}],
            intervention_types=pv.VanillaIntervention,
        )
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        source = tokenizer("The capital of Italy is", return_tensors="pt")

        # using intervention skipping for subspace
        _, pv_out1 = pv_gpt2(
            base, [None, source],
            # 4 means token position 4
            {"sources->base": ([None, [[4]]], [None, [[4]]])},
            # 1 means the second partition in the config
            subspaces=[None, [[1]]],
        )
        _, pv_out2 = pv_gpt2(
            base,
            [source, None],
            {"sources->base": ([[[4]], None], [[[4]], None])},
            subspaces=[[[1]], None],
        )
        self.assertTrue(torch.equal(pv_out1.last_hidden_state, pv_out2.last_hidden_state))

        # subspaces provide a list of index and they can be in any order
        _, pv_out3 = pv_gpt2(
            base,
            [source, source],
            {"sources->base": ([[[4]], [[4]]], [[[4]], [[4]]])},
            subspaces=[[[0]], [[1]]],
        )
        _, pv_out4 = pv_gpt2(
            base,
            [source, source],
            {"sources->base": ([[[4]], [[4]]], [[[4]], [[4]]])},
            subspaces=[[[1]], [[0]]],
        )
        self.assertTrue(torch.equal(pv_out3.last_hidden_state, pv_out4.last_hidden_state))
    
    def test_new_model_type(self):
        try:
            import sentencepiece
        except:
            print("sentencepiece is not installed. skipping")
            return
        # get a flan-t5 from HuggingFace
        from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
        config = T5Config.from_pretrained("google/flan-t5-small")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        t5 = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-small", config=config, cache_dir=self._test_dir
        )

        # config the intervention mapping with pv global vars
        """Only define for the block output here for simplicity"""
        pv.type_to_module_mapping[type(t5)] = {
            "mlp_output": ("encoder.block[%s].layer[1]", 
                           pv.models.constants.CONST_OUTPUT_HOOK),
            "attention_input": ("encoder.block[%s].layer[0]", 
                                pv.models.constants.CONST_OUTPUT_HOOK),
        }
        pv.type_to_dimension_mapping[type(t5)] = {
            "mlp_output": ("d_model",),
            "attention_input": ("d_model",),
            "block_output": ("d_model",),
            "head_attention_value_output": ("d_model/num_heads",),
        }

        # wrap as gpt2
        pv_t5 = pv.IntervenableModel({
            "layer": 0,
            "component": "mlp_output",
            "source_representation": torch.zeros(
                t5.config.d_model)
        }, model=t5)

        # then intervene!
        base = tokenizer("The capital of Spain is", 
                         return_tensors="pt")
        decoder_input_ids = tokenizer(
            "", return_tensors="pt").input_ids
        base["decoder_input_ids"] = decoder_input_ids
        intervened_outputs = pv_t5(
            base, 
            unit_locations={"base": 3}
        )

    def test_path_patching(self):

        def path_patching_config(
            layer, last_layer, 
            component="head_attention_value_output", unit="h.pos"
        ):
            intervening_component = [
                {"layer": layer, "component": component, "unit": unit, "group_key": 0}]
            restoring_components = []
            if not component.startswith("mlp_"):
                restoring_components += [
                    {"layer": layer, "component": "mlp_output", "group_key": 1}]
            for i in range(layer+1, last_layer):
                restoring_components += [
                    {"layer": i, "component": "attention_output", "group_key": 1},
                    {"layer": i, "component": "mlp_output", "group_key": 1}
                ]
            intervenable_config = pv.IntervenableConfig(
                intervening_component + restoring_components)
            return intervenable_config

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        pv_gpt2 = pv.IntervenableModel(
            path_patching_config(4, gpt2.config.n_layer), 
            model=gpt2
        )

        pv_gpt2.save(
            save_directory="./tmp/"
        )
        
        pv_gpt2 = pv.IntervenableModel.load(
            "./tmp/",
            model=gpt2)
    
    def test_multisource_parallel(self):

        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            {"layer": 0, "component": "mlp_output"},
            {"layer": 2, "component": "mlp_output"}],
            mode="parallel"
        )
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        sources = [tokenizer("The capital of Italy is", return_tensors="pt"),
                  tokenizer("The capital of China is", return_tensors="pt")]

        intervened_outputs = pv_gpt2(
            base, sources,
            # on same position
            {"sources->base": 4},
        )
        
        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            {"layer": 0, "component": "block_output",
             "subspace_partition": 
                 [[0, 128], [128, 256]]}]*2,
            intervention_types=pv.VanillaIntervention,
            # act in parallel
            mode="parallel"
        )
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        sources = [tokenizer("The capital of Italy is", return_tensors="pt"),
                  tokenizer("The capital of China is", return_tensors="pt")]

        intervened_outputs = pv_gpt2(
            base, sources,
            # on same position
            {"sources->base": 4},
            # on different subspaces
            subspaces=[[[0]], [[1]]],
        )
    
    def test_multisource_serial(self):
        
        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            {"layer": 0, "component": "mlp_output"},
            {"layer": 2, "component": "mlp_output"}],
            mode="serial"
        )
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        sources = [tokenizer("The capital of Italy is", return_tensors="pt"),
                  tokenizer("The capital of China is", return_tensors="pt")]

        intervened_outputs = pv_gpt2(
            base, sources,
            # serialized intervention
            # order is based on sources list
            {"source_0->source_1": 3, "source_1->base": 4},
        )
        
        _, tokenizer, gpt2 = pv.create_gpt2(cache_dir=self._test_dir)

        config = pv.IntervenableConfig([
            {"layer": 0, "component": "block_output",
             "subspace_partition": [[0, 128], [128, 256]]},
            {"layer": 2, "component": "block_output",
             "subspace_partition": [[0, 128], [128, 256]]}],
            intervention_types=pv.VanillaIntervention,
            # act in parallel
            mode="serial"
        )
        pv_gpt2 = pv.IntervenableModel(config, model=gpt2)

        base = tokenizer("The capital of Spain is", return_tensors="pt")
        sources = [tokenizer("The capital of Italy is", return_tensors="pt"),
                  tokenizer("The capital of China is", return_tensors="pt")]

        intervened_outputs = pv_gpt2(
            base, sources,
            # serialized intervention
            # order is based on sources list
            {"source_0->source_1": 3, "source_1->base": 4},
            # on different subspaces
            subspaces=[[[0]], [[1]]],
        )
        
    def test_customized_intervention_function_get(self):

        _, tokenizer, gpt2 = pv.create_gpt2()

        gpt2.config.output_attentions = True
        pv_gpt2 = pv.IntervenableModel({
            "layer": 10,
            "component": "attention_weight",
            "intervention_type": pv.CollectIntervention}, model=gpt2)

        base = "When John and Mary went to the shops, Mary gave the bag to"
        collected_attn_w = pv_gpt2(
            base = tokenizer(base, return_tensors="pt"
            ), unit_locations={"base": [h for h in range(12)]}
        )[0][-1][0]

        cached_w = {}
        def pv_patcher(b, s): cached_w["attn_w"] = copy.deepcopy(b.data)

        pv_gpt2 = pv.IntervenableModel({
            "component": "h[10].attn.attn_dropout.input", 
            "intervention": pv_patcher}, model=gpt2)

        base = "When John and Mary went to the shops, Mary gave the bag to"
        _ = pv_gpt2(tokenizer(base, return_tensors="pt"))
        torch.allclose(collected_attn_w, cached_w["attn_w"].unsqueeze(dim=0))
      
    def test_customized_intervention_function_zeroout(self):
        
        _, tokenizer, gpt2 = pv.create_gpt2()

        # define the component to zero-out
        pv_gpt2 = pv.IntervenableModel({
            "layer": 0, "component": "mlp_output",
            "source_representation": torch.zeros(gpt2.config.n_embd)
        }, model=gpt2)
        # run the intervened forward pass
        intervened_outputs = pv_gpt2(
            base = tokenizer("The capital of Spain is", return_tensors="pt"), 
            # we define the intervening token dynamically
            unit_locations={"base": 3}
        )
        
        # indices are specified in the intervention
        mask = torch.ones(1, 5, 768)
        mask[:,3,:] = 0.
        # define the component to zero-out
        pv_gpt2 = pv.IntervenableModel({
            "component": "h[0].mlp.output",
            "intervention": lambda b, s: b*mask
        }, model=gpt2)
        # run the intervened forward pass
        intervened_outputs_fn = pv_gpt2(
            base = tokenizer("The capital of Spain is", return_tensors="pt")
        )
        torch.allclose(
            intervened_outputs[1].last_hidden_state, 
            intervened_outputs_fn[1].last_hidden_state
        )
    
    @classmethod
    def tearDownClass(self):
        print(f"Removing testing dir {self._test_dir}")
        if os.path.exists(self._test_dir) and os.path.isdir(self._test_dir):
            shutil.rmtree(self._test_dir)
