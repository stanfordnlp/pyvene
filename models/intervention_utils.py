def _do_intervention_by_swap(
    base, source, mode="interchange", 
    interchange_dim=None,
    subspaces=None,
    subspace_partition=None
):
    """The basic do function that guards interventions"""
    # interchange
    if subspaces is not None:
        for example_i in range(len(subspaces)):
            # render subspace as column indices
            sel_subspace_indices = []
            for subspace in subspaces[example_i]:
                sel_subspace_indices.extend(
                    [
                        i for i in range(
                            subspace_partition[subspace][0], 
                            subspace_partition[subspace][1]
                        )
                    ])
            if mode == "interchange":
                base[example_i, ..., sel_subspace_indices] = \
                    source[example_i, ..., sel_subspace_indices]
            elif mode == "add":
                base[example_i, ..., sel_subspace_indices] += \
                    source[example_i, ..., sel_subspace_indices]
            elif mode == "subtract":
                base[example_i, ..., sel_subspace_indices] -= \
                    source[example_i, ..., sel_subspace_indices]
    else:
        base[..., :interchange_dim] = source[..., :interchange_dim]
    
    return base