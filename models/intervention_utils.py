import json


class InterventionState(object):
    def __init__(self, key, **kwargs):
        self.key = key
        self.reset()

    def inc_getter_version(self):
        self.state_dict["getter_version"] += 1

    def inc_setter_version(self):
        self.state_dict["setter_version"] += 1

    def getter_version(self):
        return self.state_dict["getter_version"]

    def setter_version(self):
        return self.state_dict["setter_version"]

    def get_states(self):
        return self.state_dict

    def set_state(self, state_dict):
        self.state_dict = state_dict

    def reset(self):
        self.state_dict = {
            "key": self.key,
            "getter_version": 0,
            "setter_version": 0,
        }

    def __repr__(self):
        return json.dumps(self.state_dict, indent=4)

    def __str__(self):
        return json.dumps(self.state_dict, indent=4)


def _do_intervention_by_swap(
    base,
    source,
    mode="interchange",
    interchange_dim=None,
    subspaces=None,
    subspace_partition=None,
    use_fast=False,
):
    """The basic do function that guards interventions"""
    # interchange
    if use_fast:
        if subspaces is not None:
            if subspace_partition is None:
                sel_subspace_indices = subspaces[0]
            else:
                sel_subspace_indices = []
                for subspace in subspaces[0]:
                    sel_subspace_indices.extend(
                        [
                            i
                            for i in range(
                                subspace_partition[subspace][0],
                                subspace_partition[subspace][1],
                            )
                        ]
                    )
            if mode == "interchange":
                base[..., sel_subspace_indices] = source[..., sel_subspace_indices]
            elif mode == "add":
                base[..., sel_subspace_indices] += source[..., sel_subspace_indices]
            elif mode == "subtract":
                base[..., sel_subspace_indices] -= source[..., sel_subspace_indices]
        else:
            base[..., :interchange_dim] = source[..., :interchange_dim]
    elif subspaces is not None:
        for example_i in range(len(subspaces)):
            # render subspace as column indices
            sel_subspace_indices = []
            for subspace in subspaces[example_i]:
                sel_subspace_indices.extend(
                    [
                        i
                        for i in range(
                            subspace_partition[subspace][0],
                            subspace_partition[subspace][1],
                        )
                    ]
                )
            if mode == "interchange":
                base[example_i, ..., sel_subspace_indices] = source[
                    example_i, ..., sel_subspace_indices
                ]
            elif mode == "add":
                base[example_i, ..., sel_subspace_indices] += source[
                    example_i, ..., sel_subspace_indices
                ]
            elif mode == "subtract":
                base[example_i, ..., sel_subspace_indices] -= source[
                    example_i, ..., sel_subspace_indices
                ]
    else:
        if mode == "interchange":
            base[..., :interchange_dim] = source[..., :interchange_dim]
        elif mode == "add":
            base[..., :interchange_dim] += source[..., :interchange_dim]
        elif mode == "subtract":
            base[..., :interchange_dim] -= source[..., :interchange_dim]

    return base
