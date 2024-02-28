import unittest
from ..utils import *


class InterventionWithMLPTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("=== Test Suite: InterventionWithMLPTestCase ===")
        self.config, self.tokenizer, self.mlp = create_mlp_classifier(
            MLPConfig(
                h_dim=3, n_layer=1, pdrop=0.0, num_classes=5,
                include_bias=False, squeeze_output=False
            )
        )

        self.test_subspace_intervention_link_config = IntervenableConfig(
            model_type=type(self.mlp),
            representations=[
                RepresentationConfig(
                    0,
                    "mlp_activation",
                    "pos",  # mlp layer creates a single token reprs
                    1,
                    subspace_partition=[
                        [1, 3],
                        [0, 1],
                    ],  # partition into two sets of subspaces
                    intervention_link_key=0,  # linked ones target the same subspace
                ),
                RepresentationConfig(
                    0,
                    "mlp_activation",
                    "pos",  # mlp layer creates a single token reprs
                    1,
                    subspace_partition=[
                        [1, 3],
                        [0, 1],
                    ],  # partition into two sets of subspaces
                    intervention_link_key=0,  # linked ones target the same subspace
                ),
            ],
            intervention_types=VanillaIntervention,
        )

        self.test_subspace_no_intervention_link_config = (
            IntervenableConfig(
                model_type=type(self.mlp),
                representations=[
                    RepresentationConfig(
                        0,
                        "mlp_activation",
                        "pos",  # mlp layer creates a single token reprs
                        1,
                        subspace_partition=[
                            [0, 1],
                            [1, 3],
                        ],  # partition into two sets of subspaces
                    ),
                    RepresentationConfig(
                        0,
                        "mlp_activation",
                        "pos",  # mlp layer creates a single token reprs
                        1,
                        subspace_partition=[
                            [0, 1],
                            [1, 3],
                        ],  # partition into two sets of subspaces
                    ),
                ],
                intervention_types=VanillaIntervention,
            )
        )

        self.test_subspace_no_intervention_link_trainable_config = (
            IntervenableConfig(
                model_type=type(self.mlp),
                representations=[
                    RepresentationConfig(
                        0,
                        "mlp_activation",
                        "pos",  # mlp layer creates a single token reprs
                        1,
                        low_rank_dimension=2,
                        subspace_partition=[
                            [0, 1],
                            [1, 2],
                        ],  # partition into two sets of subspaces
                    ),
                    RepresentationConfig(
                        0,
                        "mlp_activation",
                        "pos",  # mlp layer creates a single token reprs
                        1,
                        low_rank_dimension=2,
                        subspace_partition=[
                            [0, 1],
                            [1, 2],
                        ],  # partition into two sets of subspaces
                    ),
                ],
                intervention_types=LowRankRotatedSpaceIntervention,
            )
        )

    def test_clean_run_positive(self):
        """
        Positive test case to check whether vanilla forward pass work
        with our object.
        """
        intervenable = IntervenableModel(
            self.test_subspace_intervention_link_config, self.mlp
        )
        base = {"inputs_embeds": torch.rand(10, 1, 3)}
        self.assertTrue(
            torch.allclose(ONE_MLP_CLEAN_RUN(base, self.mlp), intervenable(
                base, output_original_output=True)[0][0])
        )

    def test_with_subspace_positive(self):
        """
        Positive test case to intervene only a set of subspace.
        """
        intervenable = IntervenableModel(
            self.test_subspace_intervention_link_config, self.mlp
        )
        # golden label
        b_s = 10
        base = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_1 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_2 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        base_act = ONE_MLP_FETCH_W1_ACT(base, self.mlp)
        source_1_act = ONE_MLP_FETCH_W1_ACT(source_1, self.mlp)
        intervened_act = base_act.clone()  # relentless clone
        intervened_act[..., 1:3] = source_1_act[..., 1:3]
        golden_out = ONE_MLP_WITH_W1_ACT_RUN(intervened_act, self.mlp)

        # our label
        _, our_out = intervenable(
            base,
            [source_1, None],
            {"sources->base": ([[[0]] * b_s, None], [[[0]] * b_s, None])},
            subspaces=[[[0]] * b_s, None],
        )
        assert our_out[0].shape[-1] == 5
        self.assertTrue(torch.allclose(golden_out, our_out[0]))

    def test_with_subspace_negative(self):
        """
        Negative test case to check input length.
        """
        intervenable = IntervenableModel(
            self.test_subspace_intervention_link_config, self.mlp
        )
        # golden label
        b_s = 10
        base = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_1 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_2 = {"inputs_embeds": torch.rand(b_s, 1, 3)}

        try:
            intervenable(
                base,
                [source_1],
                {"sources->base": ([[[0]] * b_s], [[[0]] * b_s])},
                subspaces=0,
            )
        except IndexError:
            pass
        else:
            raise AssertionError("IndexError was not raised")

    def test_intervention_link_positive(self):
        """
        Positive test case to intervene linked subspace.
        """
        intervenable = IntervenableModel(
            self.test_subspace_intervention_link_config, self.mlp
        )
        # golden label
        b_s = 10
        base = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_1 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_2 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        base_act = ONE_MLP_FETCH_W1_ACT(base, self.mlp)
        source_1_act = ONE_MLP_FETCH_W1_ACT(source_1, self.mlp)
        source_2_act = ONE_MLP_FETCH_W1_ACT(source_2, self.mlp)

        # overwrite version
        intervened_act = base_act.clone()  # relentless clone
        intervened_act[..., 1:3] = source_2_act[..., 1:3]
        golden_out_overwrite = ONE_MLP_WITH_W1_ACT_RUN(intervened_act, self.mlp)

        # success version
        intervened_act = base_act.clone()  # relentless clone
        intervened_act[..., 1:3] = source_1_act[..., 1:3]
        intervened_act[..., 0] = source_2_act[..., 0]
        golden_out_success = ONE_MLP_WITH_W1_ACT_RUN(intervened_act, self.mlp)

        # subcase where the second one accidentally overwrites the first one
        _, our_out_overwrite = intervenable(
            base,
            [source_1, source_2],
            {"sources->base": ([[[0]] * b_s, [[0]] * b_s], [[[0]] * b_s, [[0]] * b_s])},
            subspaces=[[[0]] * b_s, [[0]] * b_s],
        )

        # success
        _, our_out_success = intervenable(
            base,
            [source_1, source_2],
            {"sources->base": ([[[0]] * b_s, [[0]] * b_s], [[[0]] * b_s, [[0]] * b_s])},
            subspaces=[[[0]] * b_s, [[1]] * b_s],
        )

        self.assertTrue(torch.allclose(golden_out_overwrite, our_out_overwrite[0]))
        self.assertTrue(torch.allclose(golden_out_success, our_out_success[0]))

    def test_no_intervention_link_positive(self):
        """
        Positive test case to intervene not linked subspace (overwrite).
        """
        intervenable = IntervenableModel(
            self.test_subspace_no_intervention_link_config, self.mlp
        )
        # golden label
        b_s = 10
        base = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_1 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_2 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        base_act = ONE_MLP_FETCH_W1_ACT(base, self.mlp)
        source_1_act = ONE_MLP_FETCH_W1_ACT(source_1, self.mlp)
        source_2_act = ONE_MLP_FETCH_W1_ACT(source_2, self.mlp)

        # inplace overwrite version
        intervened_act = base_act.clone()  # relentless clone
        intervened_act[..., 0] = source_2_act[..., 0]
        golden_out_inplace = ONE_MLP_WITH_W1_ACT_RUN(intervened_act, self.mlp)

        # overwrite version
        intervened_act = base_act.clone()  # relentless clone
        intervened_act[..., 0] = source_1_act[..., 0]
        intervened_act[..., 1:3] = source_2_act[..., 1:3]
        golden_out_overwrite = ONE_MLP_WITH_W1_ACT_RUN(intervened_act, self.mlp)

        # subcase where the second one accidentally overwrites the first one
        _, our_out_inplace = intervenable(
            base,
            [source_1, source_2],
            {"sources->base": ([[[0]] * b_s, [[0]] * b_s], [[[0]] * b_s, [[0]] * b_s])},
            subspaces=[[[0]] * b_s, [[0]] * b_s],
        )

        # overwrite
        _, our_out_overwrite = intervenable(
            base,
            [source_1, source_2],
            {"sources->base": ([[[0]] * b_s, [[0]] * b_s], [[[0]] * b_s, [[0]] * b_s])},
            subspaces=[[[0]] * b_s, [[1]] * b_s],
            output_original_output=True,
        )

        self.assertTrue(torch.allclose(golden_out_inplace, our_out_inplace[0]))
        # the following thing work but gradient will fail check negative test cases
        self.assertTrue(torch.allclose(golden_out_overwrite, our_out_overwrite[0]))

    def test_no_intervention_link_negative(self):
        """
        Negative test case to intervene not linked subspace with trainable interventions.
        """
        intervenable = IntervenableModel(
            self.test_subspace_no_intervention_link_trainable_config,
            self.mlp,
        )
        # golden label
        b_s = 10
        base = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_1 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        source_2 = {"inputs_embeds": torch.rand(b_s, 1, 3)}
        base_act = ONE_MLP_FETCH_W1_ACT(base, self.mlp)
        source_1_act = ONE_MLP_FETCH_W1_ACT(source_1, self.mlp)
        source_2_act = ONE_MLP_FETCH_W1_ACT(source_2, self.mlp)

        # overwrite version
        intervened_act = base_act.clone()  # relentless clone
        intervened_act[..., 0] = source_1_act[..., 0]
        intervened_act[..., 1] = source_2_act[..., 1]
        golden_out_overwrite = ONE_MLP_WITH_W1_ACT_RUN(intervened_act, self.mlp)

        # overwrite
        _, our_out_overwrite = intervenable(
            base,
            [source_1, source_2],
            {"sources->base": ([[[0]] * b_s, [[0]] * b_s], [[[0]] * b_s, [[0]] * b_s])},
            subspaces=[[[0]] * b_s, [[1]] * b_s],
        )
        # it will work but the gradient is not accurate
        our_out_overwrite[0].sum().backward()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(InterventionWithMLPTestCase("test_clean_run_positive"))
    suite.addTest(InterventionWithMLPTestCase("test_with_subspace_positive"))
    suite.addTest(InterventionWithMLPTestCase("test_with_subspace_negative"))
    suite.addTest(
        InterventionWithMLPTestCase("test_intervention_link_positive")
    )
    suite.addTest(
        InterventionWithMLPTestCase("test_no_intervention_link_positive")
    )
    suite.addTest(
        InterventionWithMLPTestCase("test_no_intervention_link_negative")
    )
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
