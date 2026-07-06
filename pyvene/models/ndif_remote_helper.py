"""Remote-execution helpers built on plain nnsight.

NDIF only runs allowlisted modules server-side, so this file deliberately avoids
importing pyvene: keep it to torch and the standard library, and express every
intervention as tensor ops.
"""

import torch

CONST_INPUT_HOOK = "input"
CONST_OUTPUT_HOOK = "output"


def _get_module_output(module_hook, hook_type):
    """Return the proxy output for a module hook inside a trace context."""
    if hook_type == CONST_INPUT_HOOK:
        return module_hook.input
    return module_hook.output


def _get_act(output):
    """Extract the activation tensor from a possibly-tuple proxy."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _set_act(output, value):
    """In-place assign value into the activation (handles tuple outputs)."""
    if isinstance(output, tuple):
        output[0][:] = value
    else:
        output[:] = value


def _seed_sources(activations_sources):
    """Build the source-activation dict from caller-supplied activations.

    Values may arrive as a bare tensor or wrapped in a single-element list/tuple
    (the format the local backend caches), so unwrap the wrapper here."""
    if activations_sources is None:
        return {}
    seeded = {}
    for key, value in activations_sources.items():
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        seeded[key] = value
    return seeded


def _positions(loc):
    """Flatten a unit-location spec to a list of positions, or None for 'all'.

    Handles None, [None], [[None]], [[3]], [3], [[0, 1, 2]] etc. A per-example
    (batched) spec is assumed uniform across the batch, so we take the first.
    """
    def _all_none(x):
        if x is None:
            return True
        if isinstance(x, (list, tuple)):
            return all(_all_none(i) for i in x)
        return False

    if _all_none(loc):
        return None
    cur = loc
    while isinstance(cur, (list, tuple)) and len(cur) > 0 and isinstance(cur[0], (list, tuple)):
        cur = cur[0]
    if isinstance(cur, (list, tuple)):
        return [int(i) for i in cur if i is not None]
    return [int(cur)]


def _scatter(base_act, new_full, base_loc):
    """Return base_act with new_full written in only at base_loc positions
    (or fully replaced when base_loc is None / all-positions)."""
    positions = _positions(base_loc)
    if positions is None:
        return new_full
    patched = base_act.clone()
    patched[:, positions, :] = new_full[:, positions, :]
    return patched


def _interchange(act, src, base_loc, source_loc, op="set"):
    """Apply src to act at base_loc positions (op: set/add/sub), reading src at
    source_loc. base_loc None means all positions. src may be a full activation
    or a constant that broadcasts."""
    src = src.to(act.device, act.dtype)
    bpos = _positions(base_loc)
    if bpos is None:
        value = src if getattr(src, "shape", None) == act.shape else (act * 0 + src)
        if op == "add":
            return act + value
        if op == "sub":
            return act - value
        return value
    if getattr(src, "shape", None) == act.shape:
        spos = _positions(source_loc)
        chunk = src[:, spos if spos is not None else bpos, :]
    else:
        chunk = src  # constant broadcasts across the selected positions
    patched = act.clone()
    if op == "add":
        patched[:, bpos, :] = act[:, bpos, :] + chunk
    elif op == "sub":
        patched[:, bpos, :] = act[:, bpos, :] - chunk
    else:
        patched[:, bpos, :] = chunk
    return patched


def _align_source(base_act, source_act, base_loc, source_loc):
    """Return a base-shaped activation whose base_loc positions hold the source
    values taken from source_loc positions.

    Trainable interventions run their math position-by-position, so the base and
    source slices have to line up the way the local backend gathers them. When
    either side asks for all positions we leave the source untouched (a full
    swap), matching the previous behaviour.
    """
    bpos = _positions(base_loc)
    spos = _positions(source_loc)
    if bpos is None or spos is None:
        return source_act
    aligned = base_act.clone()
    aligned[:, bpos, :] = source_act[:, spos, :]
    return aligned


def _apply_at(output, base_act, src, base_loc, source_loc, op):
    """Apply op(base_slice, source_slice) at the requested positions only.

    op takes (base, source) tensors and returns the replacement. When both
    locations are 'all', we operate on the whole activation (old behaviour);
    otherwise we patch just the base_loc positions using the source_loc slice,
    mirroring the local/native backends' targeted interchange.
    """
    bpos = _positions(base_loc)
    spos = _positions(source_loc)
    if bpos is None and spos is None:
        _set_act(output, op(base_act, src))
        return
    b_idx = bpos if bpos is not None else spos
    s_idx = spos if spos is not None else bpos
    patched = base_act.clone()
    patched[:, b_idx, :] = op(base_act[:, b_idx, :], src[:, s_idx, :])
    _set_act(output, patched)


def _apply_trainable_weights(base_act, source_act, weights, subspaces=None):
    """Compute the intervened activation for a weighted intervention from its
    serialized weights. Covers rotation-based interventions plus the weighted
    non-trainable ones (pca/autoencoder/jumprelu). Returns a full-shaped tensor;
    callers scatter it back into the requested positions."""
    intervention_type = weights.get('intervention_type', 'rotated_space')

    if intervention_type == 'sigmoid_mask':
        mask = weights['mask'].to(base_act.device)
        temperature = weights['temperature']
        mask_sigmoid = torch.sigmoid(mask / temperature)
        return (1.0 - mask_sigmoid) * base_act + mask_sigmoid * source_act

    if intervention_type == 'pca_rotated_space':
        pca_c = weights['pca_components'].to(base_act.device)
        pca_m = weights['pca_mean'].to(base_act.device)
        pca_s = weights['pca_std'].to(base_act.device)
        base_norm = (base_act.to(pca_c.dtype) - pca_m) / pca_s
        src_norm = (source_act.to(pca_c.dtype) - pca_m) / pca_s
        rot_base = torch.matmul(base_norm, pca_c.T)
        rot_src = torch.matmul(src_norm, pca_c.T)
        interchange_d = weights.get('interchange_dim')
        if interchange_d is not None:
            rot_base[..., :interchange_d] = rot_src[..., :interchange_d]
        else:
            rot_base = rot_src
        return (torch.matmul(rot_base, pca_c) * pca_s + pca_m).to(base_act.dtype)

    if intervention_type == 'autoencoder':
        enc_w = weights['encoder_weight'].to(base_act.device)
        enc_b = weights['encoder_bias'].to(base_act.device)
        dec_w = weights['decoder_weight'].to(base_act.device)
        dec_b = weights['decoder_bias'].to(base_act.device)
        interchange_d = weights.get('interchange_dim')
        base_lat = torch.relu(base_act.to(enc_w.dtype) @ enc_w.T + enc_b)
        src_lat = torch.relu(source_act.to(enc_w.dtype) @ enc_w.T + enc_b)
        if interchange_d is not None:
            base_lat[..., :interchange_d] = src_lat[..., :interchange_d]
        else:
            base_lat = src_lat
        return (base_lat @ dec_w.T + dec_b).to(base_act.dtype)

    if intervention_type == 'jumprelu_autoencoder':
        W_enc = weights['W_enc'].to(base_act.device)
        W_dec = weights['W_dec'].to(base_act.device)
        threshold = weights['threshold'].to(base_act.device)
        b_enc = weights['b_enc'].to(base_act.device)
        b_dec = weights['b_dec'].to(base_act.device)
        interchange_d = weights.get('interchange_dim')
        pre_base = base_act @ W_enc + b_enc
        base_lat = (pre_base > threshold) * torch.relu(pre_base)
        pre_src = source_act @ W_enc + b_enc
        src_lat = (pre_src > threshold) * torch.relu(pre_src)
        if interchange_d is not None:
            base_lat[..., :interchange_d] = src_lat[..., :interchange_d]
        else:
            base_lat = src_lat
        return (base_lat @ W_dec + b_dec).to(base_act.dtype)

    # rotated_space / low_rank / boundless / sigmoid_mask_rotated variants
    rotation_matrix = weights['rotate_layer_weight'].to(base_act.device)
    rotated_base = torch.matmul(base_act.to(rotation_matrix.dtype), rotation_matrix)
    rotated_source = torch.matmul(source_act.to(rotation_matrix.dtype), rotation_matrix)
    diff = rotated_source - rotated_base

    if intervention_type == 'boundless_rotated_space':
        intervention_boundaries = weights['intervention_boundaries'].to(base_act.device)
        temperature = weights['temperature']
        intervention_population = weights['intervention_population'].to(base_act.device)
        embed_dim = weights['embed_dim']
        batch_size = base_act.shape[0]
        intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)
        positions = intervention_population.repeat(batch_size, 1)
        boundary_val = intervention_boundaries[0] * embed_dim
        boundary_mask = torch.sigmoid(temperature * (boundary_val - positions))
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
        return torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)

    if intervention_type == 'sigmoid_mask_rotated_space':
        masks = weights['masks'].to(base_act.device)
        temperature = weights['temperature']
        batch_size = base_act.shape[0]
        boundary_mask = torch.sigmoid(masks / temperature)
        boundary_mask = (
            torch.ones(batch_size, device=base_act.device).unsqueeze(-1) * boundary_mask
        ).to(rotated_base.dtype)
        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
        return torch.matmul(rotated_output, rotation_matrix.T).to(base_act.dtype)

    if subspaces is not None:
        subspace_partition = weights.get('subspace_partition')
        use_fast = weights.get('use_fast', False)
        can_use_fast = use_fast or (len(set(tuple(s) for s in subspaces)) == 1)
        if can_use_fast:
            sel = subspaces[0] if subspace_partition is None else [
                i for sub in subspaces[0] for i in subspace_partition[sub]
            ]
            batched_subspace = diff[..., sel].unsqueeze(1)
            batched_weights = rotation_matrix[..., sel].T
            return (base_act + torch.matmul(batched_subspace, batched_weights).squeeze(1)).to(base_act.dtype)
        batched_subspace, batched_weights_list = [], []
        for example_i in range(len(subspaces)):
            sel = [i for sub in subspaces[example_i] for i in subspace_partition[sub]]
            batched_subspace.append(diff[example_i, sel].unsqueeze(0))
            batched_weights_list.append(rotation_matrix[..., sel].T)
        batched_subspace = torch.stack(batched_subspace, dim=0)
        bw = torch.stack(batched_weights_list, dim=0)
        return (base_act + torch.matmul(batched_subspace, bw).squeeze(1)).to(base_act.dtype)

    return (base_act + torch.matmul(diff, rotation_matrix.T)).to(base_act.dtype)


def execute_remote_intervention(
    model,
    base,
    sources,
    intervention_specs,
    intervention_group,
    output_module=None,
    activations_sources=None,
    **kwargs
):
    """Run interventions on the NDIF backend. Handles every pyvene intervention
    type, including trainable ones via their get_remote_weights()."""
    if output_module is None:
        output_module = model.lm_head  # models without _get_output_module

    collect_specs     = [s for s in intervention_specs if s.get('is_collect')]
    vanilla_specs     = [s for s in intervention_specs if s.get('is_vanilla') and not s.get('is_trainable')]
    # weighted interventions (rotation/pca/autoencoder) run through the trainable
    # loop. PCA is weighted but not a TrainableIntervention subclass, so key off
    # the serialized weights too, not just the is_trainable flag.
    trainable_specs   = [s for s in intervention_specs
                         if (s.get('is_trainable') or s.get('intervention_weights'))
                         and not s.get('is_collect')]
    zero_specs        = [s for s in intervention_specs if s.get('is_zero')]
    addition_specs    = [s for s in intervention_specs if s.get('is_addition')]
    subtraction_specs = [s for s in intervention_specs if s.get('is_subtraction')]
    noise_specs       = [s for s in intervention_specs if s.get('is_noise')]
    lambda_specs      = [s for s in intervention_specs if s.get('is_lambda')]

    # zero/noise never need a source; neither do specs with a pre-set source rep
    sourceless_keys = set(
        s['key'] for s in zero_specs + noise_specs
        if s.get('is_source_constant')
    )
    for s in vanilla_specs + addition_specs + subtraction_specs + lambda_specs:
        if s.get('source_representation') is not None:
            sourceless_keys.add(s['key'])

    # everything else pulls its source activation from sources[group_id]
    needs_source = (
        vanilla_specs + addition_specs + subtraction_specs + trainable_specs +
        [s for s in lambda_specs if not s.get('is_source_constant')]
    )
    needs_source = [s for s in needs_source if s['key'] not in sourceless_keys]

    # collect-only is a single trace with nothing to modify
    if collect_specs and not vanilla_specs and not trainable_specs \
            and not zero_specs and not addition_specs and not subtraction_specs \
            and not noise_specs and not lambda_specs:
        collected_activations = {}
        model_out = None

        for spec in collect_specs:
            with model.trace(base, remote=True, **kwargs):
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                saved = _get_act(output).save()
                model_out = output_module.output.save()
            collected_activations[spec['key']] = saved

        return {'output': model_out, 'activations': collected_activations}

    # general case: gather source activations, then apply everything to base.
    # Seed with any caller-supplied activations (source_representations) so the
    # remote path honours precomputed sources instead of no-op'ing.
    source_activations = _seed_sources(activations_sources)
    collected_activations = {}

    specs_by_group = {}
    for spec in needs_source:
        gid = spec['group_id']
        specs_by_group.setdefault(gid, []).append(spec)

    with model.session(remote=True):
        # gather source activations
        for group_id, specs_in_group in specs_by_group.items():
            if sources is None or group_id >= len(sources) or sources[group_id] is None:
                continue
            with model.trace(sources[group_id]):
                for spec in specs_in_group:
                    output = _get_module_output(spec['module_hook'], spec['hook_type'])
                    source_activations[spec['key']] = _get_act(output).save()

        # apply to base
        with model.trace(base, **kwargs):

            # collect (no modification)
            for spec in collect_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                collected_activations[spec['key']] = act.save()

            # zero out
            for spec in zero_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                _set_act(output, torch.zeros_like(act))

            # add noise
            for spec in noise_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                interchange_d = spec.get('interchange_dim')
                noise_level = spec.get('noise_level', 0.0)
                if interchange_d is not None:
                    noisy = act.clone()
                    noisy[..., :interchange_d] = (
                        act[..., :interchange_d]
                        + torch.randn_like(act[..., :interchange_d]) * noise_level
                    )
                    _set_act(output, noisy)
                else:
                    _set_act(output, act + torch.randn_like(act) * noise_level)

            # vanilla swap (targeted at base_loc / source_loc)
            for spec in vanilla_specs:
                src = spec.get('source_representation')
                if src is None:
                    src = source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = src.to(act.device, act.dtype)
                _apply_at(output, act, src, spec.get('base_loc'), spec.get('source_loc'),
                          lambda b, s: s)

            # add source (targeted at base_loc / source_loc)
            for spec in addition_specs:
                src = spec.get('source_representation')
                if src is None:
                    src = source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = src.to(act.device, act.dtype)
                _apply_at(output, act, src, spec.get('base_loc'), spec.get('source_loc'),
                          lambda b, s: b + s)

            # subtract source (targeted at base_loc / source_loc)
            for spec in subtraction_specs:
                src = spec.get('source_representation')
                if src is None:
                    src = source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = src.to(act.device, act.dtype)
                _apply_at(output, act, src, spec.get('base_loc'), spec.get('source_loc'),
                          lambda b, s: b - s)

            # custom lambda
            for spec in lambda_specs:
                fn = spec.get('lambda_fn')
                if fn is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                result = fn(act, src)
                _set_act(output, result)

            # trainable (rotation-based)
            for spec in trainable_specs:
                weights = spec.get('intervention_weights')
                if not weights:
                    # no serialized weights: fall back to a plain swap
                    src = source_activations.get(spec['key'])
                    if src is not None:
                        output = _get_module_output(spec['module_hook'], spec['hook_type'])
                        _apply_at(output, _get_act(output), src,
                                  spec.get('base_loc'), spec.get('source_loc'),
                                  lambda b, s: s)
                    continue

                if spec['key'] not in source_activations:
                    continue

                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                base_act = _get_act(output)
                # line up the source slice with the base positions we will patch
                source_act = _align_source(
                    base_act, source_activations[spec['key']],
                    spec.get('base_loc'), spec.get('source_loc'),
                )
                intervened = _apply_trainable_weights(
                    base_act, source_act, weights, spec.get('subspaces')
                )

                # write back only the requested positions; lossy transforms
                # (autoencoder/pca) must not overwrite untargeted tokens
                _set_act(output, _scatter(base_act, intervened, spec.get('base_loc')))

            counterfactual_outputs = output_module.output.save()

    return {
        'output': counterfactual_outputs,
        'activations': collected_activations,
    }


def execute_remote_generate(
    model,
    base,
    sources,
    intervention_specs,
    output_module=None,
    activations_sources=None,
    **kwargs
):
    """Run interventions during generation on the NDIF backend. Same taxonomy as
    execute_remote_intervention, wrapped in model.generate()."""
    if output_module is None:
        output_module = model.lm_head

    collect_specs     = [s for s in intervention_specs if s.get('is_collect')]
    vanilla_specs     = [s for s in intervention_specs if s.get('is_vanilla') and not s.get('is_trainable')]
    # PCA is weighted but not a TrainableIntervention subclass, so key off the
    # serialized weights too (see execute_remote_intervention).
    trainable_specs   = [s for s in intervention_specs
                         if (s.get('is_trainable') or s.get('intervention_weights'))
                         and not s.get('is_collect')]
    zero_specs        = [s for s in intervention_specs if s.get('is_zero')]
    addition_specs    = [s for s in intervention_specs if s.get('is_addition')]
    subtraction_specs = [s for s in intervention_specs if s.get('is_subtraction')]
    noise_specs       = [s for s in intervention_specs if s.get('is_noise')]
    lambda_specs      = [s for s in intervention_specs if s.get('is_lambda')]

    sourceless_keys = set(
        s['key'] for s in zero_specs + noise_specs if s.get('is_source_constant')
    )
    for s in vanilla_specs + addition_specs + subtraction_specs + lambda_specs:
        if s.get('source_representation') is not None:
            sourceless_keys.add(s['key'])

    needs_source = (
        vanilla_specs + addition_specs + subtraction_specs + trainable_specs +
        [s for s in lambda_specs if not s.get('is_source_constant')]
    )
    needs_source = [s for s in needs_source if s['key'] not in sourceless_keys]

    specs_by_group = {}
    for spec in needs_source:
        specs_by_group.setdefault(spec['group_id'], []).append(spec)

    source_activations = _seed_sources(activations_sources)
    collected_activations = {}

    with model.session(remote=True):
        # Source collection (use trace, not generate)
        for group_id, specs_in_group in specs_by_group.items():
            if sources is None or group_id >= len(sources) or sources[group_id] is None:
                continue
            with model.trace(sources[group_id]):
                for spec in specs_in_group:
                    output = _get_module_output(spec['module_hook'], spec['hook_type'])
                    source_activations[spec['key']] = _get_act(output).save()

        # Generation with interventions applied at every forward step
        with model.generate(base, **kwargs):
            for spec in collect_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                collected_activations[spec['key']] = _get_act(output).save()

            for spec in zero_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                _set_act(output, torch.zeros_like(_get_act(output)))

            for spec in noise_specs:
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                noise_level = spec.get('noise_level', 0.0)
                interchange_d = spec.get('interchange_dim')
                if interchange_d is not None:
                    noisy = act.clone()
                    noisy[..., :interchange_d] += torch.randn_like(act[..., :interchange_d]) * noise_level
                    _set_act(output, noisy)
                else:
                    _set_act(output, act + torch.randn_like(act) * noise_level)

            for spec in vanilla_specs:
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = src.to(act.device, act.dtype)
                _apply_at(output, act, src, spec.get('base_loc'), spec.get('source_loc'),
                          lambda b, s: s)

            for spec in addition_specs:
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = src.to(act.device, act.dtype)
                _apply_at(output, act, src, spec.get('base_loc'), spec.get('source_loc'),
                          lambda b, s: b + s)

            for spec in subtraction_specs:
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                if src is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = src.to(act.device, act.dtype)
                _apply_at(output, act, src, spec.get('base_loc'), spec.get('source_loc'),
                          lambda b, s: b - s)

            for spec in lambda_specs:
                fn = spec.get('lambda_fn')
                if fn is None:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                act = _get_act(output)
                src = spec.get('source_representation') or source_activations.get(spec['key'])
                _set_act(output, fn(act, src))

            for spec in trainable_specs:
                weights = spec.get('intervention_weights')
                if not weights:
                    src = source_activations.get(spec['key'])
                    if src is not None:
                        output = _get_module_output(spec['module_hook'], spec['hook_type'])
                        _apply_at(output, _get_act(output), src,
                                  spec.get('base_loc'), spec.get('source_loc'),
                                  lambda b, s: s)
                    continue
                if spec['key'] not in source_activations:
                    continue

                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                base_act = _get_act(output)
                source_act = _align_source(
                    base_act, source_activations[spec['key']],
                    spec.get('base_loc'), spec.get('source_loc'),
                )
                intervened = _apply_trainable_weights(
                    base_act, source_act, weights, spec.get('subspaces')
                )
                _set_act(output, _scatter(base_act, intervened, spec.get('base_loc')))

            gen_output = model.generator.output.save()

    return {'output': gen_output, 'activations': collected_activations}


def execute_remote_serial_intervention(
    model,
    base,
    sources,
    intervention_specs,
    intervention_group,
    output_module=None,
    activations_sources=None,
    **kwargs
):
    """Run serial (chained) interventions on the NDIF backend. Each group's source
    is traced with the prior groups' activations already applied; the final base
    trace applies everything that was gathered."""
    if output_module is None:
        output_module = model.lm_head

    sorted_group_ids = sorted(intervention_group.keys())
    source_activations = _seed_sources(activations_sources)
    specs_by_key = {s['key']: s for s in intervention_specs}

    with model.session(remote=True):
        # For each group in order: collect its activation (applying prior interventions)
        for group_id in sorted_group_ids:
            keys = intervention_group[group_id]
            source = sources[group_id] if sources and group_id < len(sources) else None
            if source is None:
                continue

            with model.trace(source):
                # Apply interventions from all prior groups
                for prior_id in sorted_group_ids:
                    if prior_id >= group_id:
                        break
                    for spec in intervention_specs:
                        if spec['group_id'] == prior_id and spec['key'] in source_activations:
                            output = _get_module_output(spec['module_hook'], spec['hook_type'])
                            src = source_activations[spec['key']]
                            act = _get_act(output)
                            src = src.to(act.device, act.dtype)
                            if spec.get('is_vanilla'):
                                _apply_at(output, act, src, spec.get('base_loc'),
                                          spec.get('source_loc'), lambda b, s: s)
                            elif spec.get('is_addition'):
                                _apply_at(output, act, src, spec.get('base_loc'),
                                          spec.get('source_loc'), lambda b, s: b + s)

                # Collect current group's activations
                for key in keys:
                    spec = next(s for s in intervention_specs if s['key'] == key)
                    output = _get_module_output(spec['module_hook'], spec['hook_type'])
                    source_activations[key] = _get_act(output).save()

        # Final pass: apply all collected activations to base
        with model.trace(base, **kwargs):
            for spec in intervention_specs:
                if spec['key'] not in source_activations:
                    continue
                output = _get_module_output(spec['module_hook'], spec['hook_type'])
                src = source_activations[spec['key']]
                act = _get_act(output)
                base_loc = spec.get('base_loc')
                source_loc = spec.get('source_loc')

                if spec.get('is_addition'):
                    _apply_at(output, act, src.to(act.device, act.dtype),
                              base_loc, source_loc, lambda b, s: b + s)
                elif spec.get('is_subtraction'):
                    _apply_at(output, act, src.to(act.device, act.dtype),
                              base_loc, source_loc, lambda b, s: b - s)
                elif spec.get('intervention_weights'):
                    # rotation/pca/autoencoder/etc. via the shared weight math
                    source_act = _align_source(act, src, base_loc, source_loc)
                    intervened = _apply_trainable_weights(
                        act, source_act, spec['intervention_weights'], spec.get('subspaces')
                    )
                    _set_act(output, _scatter(act, intervened, base_loc))
                else:
                    # plain swap (vanilla and weightless fallbacks)
                    _apply_at(output, act, src.to(act.device, act.dtype),
                              base_loc, source_loc, lambda b, s: s)

            counterfactual_outputs = output_module.output.save()

    return {'output': counterfactual_outputs, 'activations': {}}
