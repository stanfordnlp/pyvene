import unittest
from tests.utils import *


class IntervenableUnitTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: IntervenableUnitTestCase ===")
        self.config, self.tokenizer, self.gpt2 = create_gpt2_lm(
            config = GPT2Config(
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
        self.test_output_dir_prefix = "test_tmp_output"
        self.test_output_dir_pool = []

    def test_initialization_positive(self):
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                IntervenableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervenable_interventions_type=VanillaIntervention,
        )
        
        intervenable = IntervenableModel(intervenable_config, self.gpt2)
        
        assert intervenable.mode == "parallel"
        self.assertTrue(intervenable.is_model_stateless)
        assert intervenable.use_fast == False
    
        assert len(intervenable.interventions) == 2
        key_list = []
        counter = 0
        for k, _ in intervenable.interventions.items():
            assert "#" in k
            assert int(k.split("#")[-1]) <= counter
            counter += 1
            assert "block_output" in k
        
        assert len(intervenable._intervention_group) == 2
        assert len(intervenable._key_getter_call_counter) == 2
        assert len(intervenable._key_setter_call_counter) == 2
        assert len(intervenable.activations) == 0
        assert len(intervenable.hot_activations) == 0
        assert len(intervenable._batched_setter_activation_select) == 0
        
    def test_initialization_invalid_order_negative(self):
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
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
            raise ValueError(
                "ValueError for invalid intervention "
                "order is not thrown"
            )
        
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                IntervenableRepresentationConfig(
                    0,
                    "mlp_output",
                    "pos",
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
            raise ValueError(
                "ValueError for invalid intervention "
                "order is not thrown"
            )
        
        
    def test_initialization_invalid_repr_name_negative(self):
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
                IntervenableRepresentationConfig(
                    0,
                    "strange_stream_me",
                    "pos",
                    1,
                ),
            ],
            intervenable_interventions_type=VanillaIntervention,
        )
        try:
            intervenable = IntervenableModel(intervenable_config, self.gpt2)
        except KeyError:
            pass
        else:
            raise ValueError(
                "KeyError for invalid representation name "
                "is not thrown"
            )
        
    def test_local_non_trainable_save_positive(self):
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                IntervenableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervenable_interventions_type=VanillaIntervention,
        )
        
        intervenable = IntervenableModel(intervenable_config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]
        
        intervenable.save(
            save_directory=_test_dir, 
            save_to_hf_hub=False
        )

        self.assertTrue(os.path.isfile(
            os.path.join(_test_dir, "config.json")
        ))
        for file in os.listdir(_test_dir):
            if file.endswith(".bin"):
                raise ValueError(
                    "For non-trainable interventions, "
                    "there should not be any model binary file!"
                )
        
    def test_local_trainable_save_positive(self):
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                IntervenableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervenable_interventions_type=RotatedSpaceIntervention,
        )
        
        intervenable = IntervenableModel(intervenable_config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]
        
        intervenable.save(
            save_directory=_test_dir, 
            save_to_hf_hub=False
        )

        self.assertTrue(os.path.isfile(
            os.path.join(_test_dir, "config.json")
        ))
        binary_count = 0
        for file in os.listdir(_test_dir):
            if file.endswith(".bin"):
                binary_count += 1
        if binary_count != 2:
            raise ValueError(
                "For trainable interventions, "
                "there should binary file for each of them."
            )
    
    def _test_local_trainable_load_positive(self, intervenable_interventions_type):
        b_s = 10
        
        intervenable_config = IntervenableConfig(
            intervenable_model_type=type(self.gpt2),
            intervenable_representations=[
                IntervenableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                    intervenable_low_rank_dimension=4
                ),
                IntervenableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                    intervenable_low_rank_dimension=4
                ),
            ],
            intervenable_interventions_type=intervenable_interventions_type,
        )
        
        intervenable = IntervenableModel(intervenable_config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]
        
        intervenable.save(
            save_directory=_test_dir, 
            save_to_hf_hub=False
        )
        
        intervenable_loaded = IntervenableModel.load(
            load_directory=_test_dir, 
            model=self.gpt2,
        )
        
        assert intervenable != intervenable_loaded
        
        base = {"input_ids": torch.randint(0, 10, (b_s, 10))}
        source = {"input_ids": torch.randint(0, 10, (b_s, 10))}
        
        _, counterfactual_outputs_unsaved = intervenable(
            base,
            [source, source],
            {"sources->base": ([[[3]], [[4]]], [[[3]], [[4]]])}
        )
        
        _, counterfactual_outputs_loaded = intervenable_loaded(
            base,
            [source, source],
            {"sources->base": ([[[3]], [[4]]], [[[3]], [[4]]])}
        )
        
        torch.equal(
            counterfactual_outputs_unsaved[0], 
            counterfactual_outputs_loaded[0]
        )

    def test_local_load_positive(self):
        self._test_local_trainable_load_positive(VanillaIntervention)
        self._test_local_trainable_load_positive(RotatedSpaceIntervention)
        self._test_local_trainable_load_positive(LowRankRotatedSpaceIntervention)
        
    @classmethod
    def tearDownClass(self):
        for current_dir in self.test_output_dir_pool:
            print(f"Removing testing dir {current_dir}")
            if os.path.exists(current_dir) and os.path.isdir(current_dir):
                shutil.rmtree(current_dir)
    
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(IntervenableUnitTestCase(
        'test_initialization_positive'))
    suite.addTest(IntervenableUnitTestCase(
        'test_initialization_invalid_order_negative'))
    suite.addTest(IntervenableUnitTestCase(
        'test_initialization_invalid_repr_name_negative'))
    suite.addTest(IntervenableUnitTestCase(
        'test_local_non_trainable_save_positive'))
    suite.addTest(IntervenableUnitTestCase(
        'test_local_trainable_save_positive'))
    suite.addTest(IntervenableUnitTestCase(
        'test_local_load_positive'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())