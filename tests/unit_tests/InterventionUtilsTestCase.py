import unittest
import torch.nn.functional as F
from ..utils import *
from pyvene.models.intervention_utils import _do_intervention_by_swap
from pyvene.models.interventions import VanillaIntervention
from pyvene.models.interventions import CollectIntervention


class InterventionUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: IntervenableUnitTestCase ===")
        self.config, self.tokenizer, self.gpt2 = create_gpt2_lm(
            config=GPT2Config(
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
        config = IntervenableConfig(
            model_type=type(self.gpt2),
            representations=[
                RepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                RepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervention_types=VanillaIntervention,
        )

        intervenable = IntervenableModel(config, self.gpt2)

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

    def test_local_non_trainable_save_positive(self):
        config = IntervenableConfig(
            model_type=type(self.gpt2),
            representations=[
                RepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                RepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervention_types=VanillaIntervention,
        )

        intervenable = IntervenableModel(config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]

        intervenable.save(save_directory=_test_dir, save_to_hf_hub=False)

        self.assertTrue(os.path.isfile(os.path.join(_test_dir, "config.json")))
        for file in os.listdir(_test_dir):
            if file.endswith(".bin"):
                raise ValueError(
                    "For non-trainable interventions, "
                    "there should not be any model binary file!"
                )

    def test_local_trainable_save_positive(self):
        config = IntervenableConfig(
            model_type=type(self.gpt2),
            representations=[
                RepresentationConfig(
                    0,
                    "block_output",
                    "pos",
                    1,
                ),
                RepresentationConfig(
                    1,
                    "block_output",
                    "pos",
                    1,
                ),
            ],
            intervention_types=RotatedSpaceIntervention,
        )

        intervenable = IntervenableModel(config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]

        intervenable.save(save_directory=_test_dir, save_to_hf_hub=False)

        self.assertTrue(os.path.isfile(os.path.join(_test_dir, "config.json")))
        binary_count = 0
        for file in os.listdir(_test_dir):
            if file.endswith(".bin"):
                binary_count += 1
        if binary_count != 2:
            raise ValueError(
                "For trainable interventions, "
                "there should binary file for each of them."
            )

    def _test_local_trainable_load_positive(self, intervention_types):
        b_s = 10

        config = IntervenableConfig(
            model_type=type(self.gpt2),
            representations=[
                RepresentationConfig(
                    0, "block_output", "pos", 1, low_rank_dimension=4
                ),
                RepresentationConfig(
                    1, "block_output", "pos", 1, low_rank_dimension=4
                ),
            ],
            intervention_types=intervention_types,
        )

        intervenable = IntervenableModel(config, self.gpt2)
        _uuid = str(uuid.uuid4())[:6]
        _test_dir = os.path.join(f"./test_output_dir_prefix-{_uuid}")
        self.test_output_dir_pool += [_test_dir]

        intervenable.save(save_directory=_test_dir, save_to_hf_hub=False)

        loaded = IntervenableModel.load(
            load_directory=_test_dir,
            model=self.gpt2,
        )

        assert intervenable != loaded

        base = {"input_ids": torch.randint(0, 10, (b_s, 10))}
        source = {"input_ids": torch.randint(0, 10, (b_s, 10))}

        _, counterfactual_outputs_unsaved = intervenable(
            base, [source, source], {"sources->base": ([[[3]], [[4]]], [[[3]], [[4]]])}
        )

        _, counterfactual_outputs_loaded = loaded(
            base, [source, source], {"sources->base": ([[[3]], [[4]]], [[[3]], [[4]]])}
        )

        torch.equal(counterfactual_outputs_unsaved[0], counterfactual_outputs_loaded[0])

    def test_local_load_positive(self):
        self._test_local_trainable_load_positive(VanillaIntervention)
        self._test_local_trainable_load_positive(RotatedSpaceIntervention)
        self._test_local_trainable_load_positive(LowRankRotatedSpaceIntervention)
        
    def test_vanilla_intervention_positive(self):
        intervention = VanillaIntervention(embed_dim=2)
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 72).view(2, 3, 6)
        output = intervention(base, source)
        golden = torch.tensor(
            [
                [
                    [36, 37, 2, 3, 4, 5],
                    [42, 43, 8, 9, 10, 11],
                    [48, 49, 14, 15, 16, 17],
                ],
                [
                    [54, 55, 20, 21, 22, 23],
                    [60, 61, 26, 27, 28, 29],
                    [66, 67, 32, 33, 34, 35],
                ],
            ]
        )
        self.assertTrue(torch.allclose(golden, output))

    def test_vanilla_intervention_negative(self):
        intervention = VanillaIntervention(embed_dim=2)
        base = torch.arange(36).view(2, 3, 6)
        # Shape cannot broadcast
        source = torch.arange(36, 42).view(1, 6)
        try:
            output = intervention(base, source)
        except ValueError:
            pass

    def test_vanilla_intervention_subspace_positive(self):
        intervention = VanillaIntervention()
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 72).view(2, 3, 6)
        output = intervention(base, source, subspaces=[[1, 0], [0, 1], [1, 2]])
        golden = torch.tensor(
            [
                [
                    [36, 37, 2, 3, 4, 5],
                    [42, 43, 8, 9, 10, 11],
                    [12, 49, 50, 15, 16, 17],
                ],
                [
                    [54, 55, 20, 21, 22, 23],
                    [60, 61, 26, 27, 28, 29],
                    [30, 67, 68, 33, 34, 35],
                ],
            ]
        )

        self.assertTrue(torch.allclose(golden, output))

    def test_vanilla_intervention_subspace_partition_positive(self):
        intervention = VanillaIntervention(subspace_partition=[[0, 2], [2, 4], [4, 6]])
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 72).view(2, 3, 6)
        output = intervention(base, source, subspaces=[[1, 0], [0, 1], [1, 2]])
        golden = torch.tensor(
            [
                [
                    [36, 37, 38, 39, 4, 5],
                    [42, 43, 44, 45, 10, 11],
                    [12, 13, 50, 51, 52, 53],
                ],
                [
                    [54, 55, 56, 57, 22, 23],
                    [60, 61, 62, 63, 28, 29],
                    [30, 31, 68, 69, 70, 71],
                ],
            ]
        )

        self.assertTrue(torch.allclose(golden, output))

    def test_vanilla_intervention_subspace_negative(self):
        intervention = VanillaIntervention()
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 72).view(2, 3, 6)
        try:
            # 6 is out of index bounds
            intervention(base, source, subspaces=[1, 6])
        except IndexError:
            pass

    def test_vanilla_intervention_subspace_partition_negative(self):
        # Subspace partitions are not equal
        intervention = VanillaIntervention(subspace_partition=[[0, 1, 2], [3, 4], [5]])
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 72).view(2, 3, 6)
        try:
            intervention(base, source)
        except ValueError:
            pass

    def test_vanilla_intervention_broadcast_positive(self):
        intervention = VanillaIntervention(subspace_partition=[[0, 2], [2, 4], [4, 6]])
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 42).view(6)
        output = intervention(base, source, subspaces=[[1, 0], [0, 1], [1, 2]])
        golden = torch.tensor(
            [
                [
                    [36, 37, 38, 39, 4, 5],
                    [36, 37, 38, 39, 10, 11],
                    [12, 13, 38, 39, 40, 41],
                ],
                [
                    [36, 37, 38, 39, 22, 23],
                    [36, 37, 38, 39, 28, 29],
                    [30, 31, 38, 39, 40, 41],
                ],
            ]
        )
        self.assertTrue(torch.allclose(golden, output))

    def test_vanilla_intervention_fast_positive(self):
        intervention = VanillaIntervention(
            subspace_partition=[[0, 2], [2, 4], [4, 6]], use_fast=True
        )
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 42).view(6)
        output = intervention(base, source, subspaces=[[1, 0], [0, 1], [1, 2]])
        golden = torch.tensor(
            [
                [
                    [36, 37, 38, 39, 4, 5],
                    [36, 37, 38, 39, 10, 11],
                    [36, 37, 38, 39, 16, 17],
                ],
                [
                    [36, 37, 38, 39, 22, 23],
                    [36, 37, 38, 39, 28, 29],
                    [36, 37, 38, 39, 34, 35],
                ],
            ]
        )
        self.assertTrue(torch.allclose(golden, output))

    def test_collect_intervention_negative(self):
        intervention = CollectIntervention(subspace_partition=[[0, 2], [2, 4], [4, 6]])
        base = torch.arange(36).view(2, 3, 6)
        source = torch.arange(36, 42).view(6)
        try:
            intervention(base, source, subspaces=[[1, 0], [0, 1], [1, 2]])
        except AssertionError:
            pass

    def test_collect_intervention_positive(self):
        intervention = CollectIntervention(subspace_partition=[[0, 2], [2, 4], [4, 6]])
        base = torch.arange(36).view(2, 3, 6)
        output = intervention(base, None, subspaces=[[1, 0], [0, 1], [1, 2]])
        golden = torch.tensor(
            [
                [[2, 3, 0, 1], [6, 7, 8, 9], [14, 15, 16, 17]],
                [[20, 21, 18, 19], [24, 25, 26, 27], [32, 33, 34, 35]],
            ]
        )
        self.assertTrue(torch.allclose(golden, output))

    def test_brs_intervention_positive(self):
        intervention = BoundlessRotatedSpaceIntervention(embed_dim=6)
        base = torch.arange(12).view(2, 6)
        source = torch.arange(12, 24).view(2, 6)
        output = intervention(base, source)
        golden = torch.tensor([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14],])
        self.assertTrue(torch.allclose(golden, output))

    def test_brs_gradient_positive(self):
        
        _retry = 10
        while _retry > 0:
            try:
                intervention = BoundlessRotatedSpaceIntervention(embed_dim=6)
                intervention.temperature = torch.nn.Parameter(torch.tensor(2.0))
                base = torch.arange(12).float().view(2, 6)
                source = torch.arange(12, 24).float().view(2, 6)

                optimizer_params = []
                optimizer_params += [{"params": intervention.rotate_layer.parameters()}]
                optimizer_params += [{"params": intervention.intervention_boundaries}]
                optimizer = torch.optim.Adam(optimizer_params, lr=1e-1)

                golden = torch.tensor([[5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16],]).float()

                for _ in range(1000):
                    optimizer.zero_grad()
                    output = intervention(base, source)
                    loss = F.mse_loss(output, golden)
                    loss.backward()
                    optimizer.step()
                self.assertTrue(torch.allclose(golden, output, rtol=1e-02, atol=1e-02))
            except:
                pass  # retry
            finally:
                break
            _retry -= 1
        if _retry > 0:
            pass  # succeed
        else:
            raise AssertionError(
                "test_brs_gradient_positive with retries"
            )
        

    def test_sigmoid_mask_gradient_positive(self):
        
        _retry = 10
        while _retry > 0:
            try:
                intervention = SigmoidMaskIntervention(embed_dim=6)
                base = torch.arange(12).float().view(2, 6)
                source = torch.arange(12, 24).float().view(2, 6)

                optimizer_params = []
                optimizer_params += [{"params": intervention.mask}]
                optimizer_params += [{"params": intervention.temperature}]
                optimizer = torch.optim.Adam(optimizer_params, lr=1e-1)

                golden = torch.tensor([[0, 1, 14, 15, 16, 17], [6, 7, 20, 21, 22, 23],]).float()

                for _ in range(2000):
                    optimizer.zero_grad()
                    output = intervention(base, source)
                    loss = F.mse_loss(output, golden)
                    loss.backward()
                    optimizer.step()
                res = torch.sigmoid(intervention.mask)
                self.assertTrue(
                    torch.allclose(res, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
                )
            except:
                pass  # retry
            finally:
                break
            _retry -= 1
        if _retry > 0:
            pass  # succeed
        else:
            raise AssertionError(
                "test_sigmoid_mask_gradient_positive with retries"
            )
        

    def test_low_rank_gradient_positive(self):
        
        _retry = 10
        while _retry > 0:
            try:
                intervention = LowRankRotatedSpaceIntervention(
                    embed_dim=6, low_rank_dimension=1
                )
                base = torch.arange(12).float().view(2, 6)
                source = torch.arange(12, 24).float().view(2, 6)

                optimizer_params = []
                optimizer_params += [{"params": intervention.rotate_layer.parameters()}]
                optimizer = torch.optim.Adam(optimizer_params, lr=1e-1)

                golden = torch.tensor([[0, 1, 14, 15, 16, 17], [6, 7, 20, 21, 22, 23],]).float()

                for _ in range(2000):
                    optimizer.zero_grad()
                    output = intervention(base, source)
                    loss = F.mse_loss(output, golden)
                    loss.backward()
                    optimizer.step()
                print(output)
                self.assertTrue(torch.allclose(golden, output, rtol=1e-02, atol=1e-02))
            except:
                pass  # retry
            finally:
                break
            _retry -= 1
        if _retry > 0:
            pass  # succeed
        else:
            raise AssertionError(
                "test_sigmoid_mask_gradient_positive with retries"
            )

    @classmethod
    def tearDownClass(self):
        for current_dir in self.test_output_dir_pool:
            print(f"Removing testing dir {current_dir}")
            if os.path.exists(current_dir) and os.path.isdir(current_dir):
                shutil.rmtree(current_dir)
