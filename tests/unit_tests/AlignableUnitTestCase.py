import unittest
from tests.utils import *


class AlignableUnitTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: AlignableUnitTestCase ===")
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
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                AlignableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            alignable_interventions_type=VanillaIntervention,
        )
        
        alignable = AlignableModel(alignable_config, self.gpt2)
        
        assert alignable.mode == "parallel"
        self.assertTrue(alignable.is_model_stateless)
        assert alignable.use_fast == False
    
        assert len(alignable.interventions) == 2
        key_list = []
        counter = 0
        for k, _ in alignable.interventions.items():
            assert "#" in k
            assert int(k.split("#")[-1]) <= counter
            counter += 1
            assert "block_output" in k
        
        assert len(alignable._intervention_group) == 2
        assert len(alignable._key_getter_call_counter) == 2
        assert len(alignable._key_setter_call_counter) == 2
        assert len(alignable.activations) == 0
        assert len(alignable.hot_activations) == 0
        assert len(alignable._batched_setter_activation_select) == 0
        
    def test_initialization_invalid_order_negative(self):
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            alignable_interventions_type=VanillaIntervention,
        )
        try:
            alignable = AlignableModel(alignable_config, self.gpt2)
        except ValueError:
            pass
        else:
            raise ValueError(
                "ValueError for invalid intervention "
                "order is not thrown"
            )
        
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                AlignableRepresentationConfig(
                    0,
                    "mlp_output",
                    "pos",
                    1,
                ),
            ],
            alignable_interventions_type=VanillaIntervention,
        )
        try:
            alignable = AlignableModel(alignable_config, self.gpt2)
        except ValueError:
            pass
        else:
            raise ValueError(
                "ValueError for invalid intervention "
                "order is not thrown"
            )
        
        
    def test_initialization_invalid_repr_name_negative(self):
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
                AlignableRepresentationConfig(
                    0,
                    "strange_stream_me",
                    "pos",
                    1,
                ),
            ],
            alignable_interventions_type=VanillaIntervention,
        )
        try:
            alignable = AlignableModel(alignable_config, self.gpt2)
        except KeyError:
            pass
        else:
            raise ValueError(
                "KeyError for invalid representation name "
                "is not thrown"
            )
        
    def test_local_non_trainable_save_positive(self):
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                AlignableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            alignable_interventions_type=VanillaIntervention,
        )
        
        alignable = AlignableModel(alignable_config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]
        
        alignable.save(
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
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                AlignableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            alignable_interventions_type=RotatedSpaceIntervention,
        )
        
        alignable = AlignableModel(alignable_config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]
        
        alignable.save(
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
    
    def _test_local_trainable_load_positive(self, alignable_interventions_type):
        b_s = 10
        
        alignable_config = AlignableConfig(
            alignable_model_type=type(self.gpt2),
            alignable_representations=[
                AlignableRepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                    alignable_low_rank_dimension=4
                ),
                AlignableRepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                    alignable_low_rank_dimension=4
                ),
            ],
            alignable_interventions_type=alignable_interventions_type,
        )
        
        alignable = AlignableModel(alignable_config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]
        
        alignable.save(
            save_directory=_test_dir, 
            save_to_hf_hub=False
        )
        
        alignable_loaded = AlignableModel.load(
            load_directory=_test_dir, 
            model=self.gpt2,
        )
        
        assert alignable != alignable_loaded
        
        base = {"input_ids": torch.randint(0, 10, (b_s, 10))}
        source = {"input_ids": torch.randint(0, 10, (b_s, 10))}
        
        _, counterfactual_outputs_unsaved = alignable(
            base,
            [source, source],
            {"sources->base": ([[[3]], [[4]]], [[[3]], [[4]]])}
        )
        
        _, counterfactual_outputs_loaded = alignable_loaded(
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
    suite.addTest(AlignableUnitTestCase(
        'test_initialization_positive'))
    suite.addTest(AlignableUnitTestCase(
        'test_initialization_invalid_order_negative'))
    suite.addTest(AlignableUnitTestCase(
        'test_initialization_invalid_repr_name_negative'))
    suite.addTest(AlignableUnitTestCase(
        'test_local_non_trainable_save_positive'))
    suite.addTest(AlignableUnitTestCase(
        'test_local_trainable_save_positive'))
    suite.addTest(AlignableUnitTestCase(
        'test_local_load_positive'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())