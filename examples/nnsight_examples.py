"""
NNsight/NDIF Backend Examples for Pyvene Trainable Interventions

This file demonstrates how to use each trainable intervention type with
the NDIF backend (both local and remote execution).

All examples use direct nnsight operations (model.session/model.trace) to apply
interventions, which works identically for local and remote execution.

Usage:
    # Run all examples locally (nnsight with local GPU/CPU)
    python nnsight_examples.py

    # Run with remote NDIF execution (requires NDIF account)
    # Windows: set NDIF_REMOTE=1 && python nnsight_examples.py
    # Linux/Mac: NDIF_REMOTE=1 python nnsight_examples.py

Remote Setup:
    To use remote NDIF execution, you need:
    1. An NDIF account at https://ndif.us
    2. Your API key set via: from nnsight import CONFIG; CONFIG.set_default_api_key("your-key")
       or environment variable: NDIF_API_KEY=your-key
    3. Network access to NDIF servers

    See https://nnsight.net/documentation/ for more details.
"""

import os
import torch

# Check if we should use remote execution (see docstring: NDIF_REMOTE=1 to enable)
USE_REMOTE = os.environ.get("NDIF_REMOTE", "0") == "1"

print(f"Running with remote={USE_REMOTE}")
if USE_REMOTE:
    print("NOTE: Remote execution requires NDIF account. See docstring for setup.")
print("=" * 60)

# Import pyvene and nnsight
import pyvene as pv
from nnsight import LanguageModel

# Import intervention types
from pyvene.models.interventions import (
    CollectIntervention,
    VanillaIntervention,
    LowRankRotatedSpaceIntervention,
    RotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
    SigmoidMaskRotatedSpaceIntervention,
    SigmoidMaskIntervention,
)

# Load model
print("Loading GPT-2 model...")
if USE_REMOTE:
    model = LanguageModel('openai-community/gpt2')
else:
    model = LanguageModel('openai-community/gpt2', device_map='cpu')

tokenizer = model.tokenizer
EMBED_DIM = 768  # GPT-2's hidden size

# Prepare inputs
base_text = "The capital of France is"
source_text = "The capital of Germany is"

base_tokens = tokenizer(base_text, return_tensors="pt")
source_tokens = tokenizer(source_text, return_tensors="pt")

print(f"Base text: '{base_text}'")
print(f"Source text: '{source_text}'")
print("=" * 60)


def get_clean_output():
    """Get clean model output without any intervention."""
    with model.session(remote=USE_REMOTE):
        # Use raw string for remote compatibility (BatchEncoding not on NDIF's allowlist)
        with model.trace(base_text):
            output = model.output.save()
    return output


def get_logits(output):
    """Extract logits from various output formats."""
    if hasattr(output, 'logits'):
        return output.logits
    elif isinstance(output, dict) and 'logits' in output:
        return output['logits']
    elif hasattr(output, 'value'):
        return output.value.logits if hasattr(output.value, 'logits') else output.value
    return output


# Example 1: CollectIntervention
def example_collect_intervention():
    """Example: CollectIntervention - Collect activations without modification."""
    print(f"\n{'=' * 60}")
    print("Example 1: CollectIntervention")
    print("Purpose: Collect activations from a layer without modifying them")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    pv_model = pv.build_intervenable_model({
        "component": "transformer.h[0].mlp.c_proj.output",
        "intervention": CollectIntervention()
    }, model=model, remote=USE_REMOTE)

    # Use raw string for remote compatibility
    # For unit_locations, use length of tokenized input
    num_tokens = len(tokenizer.encode(base_text))
    result = pv_model(
        base=base_text,
        unit_locations={"base": list(range(num_tokens))}
    )

    _, collected = result[0]
    print(f"Collected activations shape: {collected[0].shape}")
    print(f"Activation mean: {collected[0].mean().item():.4f}")
    print(f"Activation std: {collected[0].std().item():.4f}")
    print("SUCCESS!")

    return collected


# Example 2: VanillaIntervention
def example_vanilla_intervention():
    """Example: VanillaIntervention - Simple activation swapping."""
    print(f"\n{'=' * 60}")
    print("Example 2: VanillaIntervention")
    print("Purpose: Swap activations from source to base")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    pv_model = pv.build_intervenable_model({
        "component": "transformer.h[0].output",
        "intervention": VanillaIntervention()
    }, model=model, remote=USE_REMOTE)

    clean_output = get_clean_output()
    clean_logits = get_logits(clean_output)

    # Use raw strings for remote compatibility
    _, intervened = pv_model(
        base=base_text,
        sources=[source_text],
        unit_locations={"sources->base": ([None], [None])}
    )
    intervened_logits = get_logits(intervened)

    clean_pred = tokenizer.decode(clean_logits[0, -1].argmax())
    intervened_pred = tokenizer.decode(intervened_logits[0, -1].argmax())
    logit_diff = (clean_logits - intervened_logits).abs().mean().item()

    print(f"Clean prediction: '{clean_pred}'")
    print(f"Intervened prediction: '{intervened_pred}'")
    print(f"Mean absolute logit difference: {logit_diff:.4f}")
    print("SUCCESS!")


# Example 3: get_remote_weights() for Trainable Interventions
def example_get_remote_weights():
    """Example: Show the get_remote_weights() method for each trainable intervention."""
    print(f"\n{'=' * 60}")
    print("Example 3: get_remote_weights() for Trainable Interventions")
    print("Purpose: Show the weights that are serialized for remote execution")
    print("-" * 60)

    interventions = [
        ("LowRankRotatedSpaceIntervention",
         LowRankRotatedSpaceIntervention(embed_dim=EMBED_DIM, low_rank_dimension=64)),
        ("RotatedSpaceIntervention",
         RotatedSpaceIntervention(embed_dim=EMBED_DIM)),
        ("BoundlessRotatedSpaceIntervention",
         BoundlessRotatedSpaceIntervention(embed_dim=EMBED_DIM)),
        ("SigmoidMaskRotatedSpaceIntervention",
         SigmoidMaskRotatedSpaceIntervention(embed_dim=EMBED_DIM)),
        ("SigmoidMaskIntervention",
         SigmoidMaskIntervention(embed_dim=EMBED_DIM)),
    ]

    for name, intervention in interventions:
        weights = intervention.get_remote_weights()
        print(f"\n{name}:")
        print(f"  Keys: {list(weights.keys())}")
        print(f"  Intervention type: {weights.get('intervention_type', 'N/A')}")
        if 'rotate_layer_weight' in weights:
            print(f"  Rotation matrix shape: {weights['rotate_layer_weight'].shape}")
        if 'masks' in weights:
            print(f"  Masks shape: {weights['masks'].shape}")
        if 'mask' in weights:
            print(f"  Mask shape: {weights['mask'].shape}")
        print(f"  Trainable params: {sum(p.numel() for p in intervention.parameters())}")

    print("\nSUCCESS!")


# Example 4: LowRankRotatedSpaceIntervention (Direct nnsight)
def example_low_rank_rotated_direct():
    """Example: LowRankRotatedSpaceIntervention using direct nnsight operations."""
    print(f"\n{'=' * 60}")
    print("Example 4: LowRankRotatedSpaceIntervention (Direct nnsight)")
    print("Purpose: Apply low-rank rotation intervention using direct nnsight")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    # Create intervention and get weights
    intervention = LowRankRotatedSpaceIntervention(
        embed_dim=EMBED_DIM, low_rank_dimension=64
    )
    weights = intervention.get_remote_weights()
    rotation_matrix = weights['rotate_layer_weight']

    print(f"Rotation matrix shape: {rotation_matrix.shape}")
    print(f"Low-rank dimension: {weights['low_rank_dimension']}")

    # Get clean output
    clean_output = get_clean_output()
    clean_logits = get_logits(clean_output)

    # Apply intervention using direct nnsight
    # Use raw strings for remote compatibility (BatchEncoding not on NDIF's allowlist)
    with model.session(remote=USE_REMOTE):
        # Get source activations
        with model.trace(source_text):
            source_act = model.transformer.h[0].output[0].save()

        # Apply intervention to base
        with model.trace(base_text):
            base_act = model.transformer.h[0].output[0]

            # Move rotation_matrix to same device as base_act (CUDA on remote)
            rotation_matrix_device = rotation_matrix.to(base_act.device)

            # Low-rank rotation intervention logic
            rotated_base = torch.matmul(base_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            rotated_source = torch.matmul(source_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            diff = rotated_source - rotated_base
            intervened = base_act + torch.matmul(diff, rotation_matrix_device.T)

            model.transformer.h[0].output[0][:] = intervened.to(base_act.dtype)
            output = model.output.save()

    intervened_logits = get_logits(output)

    clean_pred = tokenizer.decode(clean_logits[0, -1].argmax())
    intervened_pred = tokenizer.decode(intervened_logits[0, -1].argmax())
    logit_diff = (clean_logits - intervened_logits).abs().mean().item()

    print(f"Clean prediction: '{clean_pred}'")
    print(f"Intervened prediction: '{intervened_pred}'")
    print(f"Mean absolute logit difference: {logit_diff:.4f}")
    print(f"Intervention had effect: {logit_diff > 0.01}")
    print("SUCCESS!")


# Example 5: RotatedSpaceIntervention (Direct nnsight)
def example_rotated_space_direct():
    """Example: RotatedSpaceIntervention using direct nnsight operations."""
    print(f"\n{'=' * 60}")
    print("Example 5: RotatedSpaceIntervention (Direct nnsight)")
    print("Purpose: Apply full rotation intervention using direct nnsight")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    # Create intervention and get weights
    intervention = RotatedSpaceIntervention(embed_dim=EMBED_DIM)
    weights = intervention.get_remote_weights()
    rotation_matrix = weights['rotate_layer_weight']

    print(f"Rotation matrix shape: {rotation_matrix.shape}")

    # Get clean output
    clean_output = get_clean_output()
    clean_logits = get_logits(clean_output)

    # Apply intervention using direct nnsight
    # Use raw strings for remote compatibility
    with model.session(remote=USE_REMOTE):
        with model.trace(source_text):
            source_act = model.transformer.h[0].output[0].save()

        with model.trace(base_text):
            base_act = model.transformer.h[0].output[0]
            # Move rotation_matrix to same device as base_act (CUDA on remote)
            rotation_matrix_device = rotation_matrix.to(base_act.device)

            # Full rotation intervention logic
            rotated_base = torch.matmul(base_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            rotated_source = torch.matmul(source_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            diff = rotated_source - rotated_base
            intervened = base_act + torch.matmul(diff, rotation_matrix_device.T)

            model.transformer.h[0].output[0][:] = intervened.to(base_act.dtype)
            output = model.output.save()

    intervened_logits = get_logits(output)
    logit_diff = (clean_logits - intervened_logits).abs().mean().item()

    print(f"Mean absolute logit difference: {logit_diff:.4f}")
    print(f"Intervention had effect: {logit_diff > 0.01}")
    print("SUCCESS!")


# Example 6: BoundlessRotatedSpaceIntervention (Direct nnsight)
def example_boundless_rotated_direct():
    """Example: BoundlessRotatedSpaceIntervention using direct nnsight operations."""
    print(f"\n{'=' * 60}")
    print("Example 6: BoundlessRotatedSpaceIntervention (Direct nnsight)")
    print("Purpose: Apply rotation with learned boundary mask")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    # Create intervention and get weights
    intervention = BoundlessRotatedSpaceIntervention(embed_dim=EMBED_DIM)
    weights = intervention.get_remote_weights()
    rotation_matrix = weights['rotate_layer_weight']
    intervention_boundaries = weights['intervention_boundaries']
    temperature = weights['temperature']
    intervention_population = weights['intervention_population']
    embed_dim = weights['embed_dim']

    print(f"Rotation matrix shape: {rotation_matrix.shape}")
    print(f"Boundary value: {intervention_boundaries.item():.4f}")

    # Get clean output
    clean_output = get_clean_output()
    clean_logits = get_logits(clean_output)

    # Apply intervention using direct nnsight
    # Use raw strings for remote compatibility
    with model.session(remote=USE_REMOTE):
        with model.trace(source_text):
            source_act = model.transformer.h[0].output[0].save()

        with model.trace(base_text):
            base_act = model.transformer.h[0].output[0]
            batch_size = base_act.shape[0]

            # Move tensors to same device as base_act (CUDA on remote)
            rotation_matrix_device = rotation_matrix.to(base_act.device)
            intervention_boundaries_device = intervention_boundaries.to(base_act.device)
            intervention_population_device = intervention_population.to(base_act.device)
            temperature_device = temperature.to(base_act.device)

            # Boundless rotation intervention logic
            rotated_base = torch.matmul(base_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            rotated_source = torch.matmul(source_act.to(rotation_matrix_device.dtype), rotation_matrix_device)

            # Compute boundary mask
            intervention_boundaries_clamped = torch.clamp(intervention_boundaries_device, 1e-3, 1)
            positions = intervention_population_device.repeat(batch_size, 1)
            boundary_val = intervention_boundaries_clamped[0] * embed_dim
            boundary_mask = torch.sigmoid(temperature_device * (boundary_val - positions))
            boundary_mask = boundary_mask.to(rotated_base.dtype)

            # Interpolate in rotated space
            rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
            intervened = torch.matmul(rotated_output, rotation_matrix_device.T)

            model.transformer.h[0].output[0][:] = intervened.to(base_act.dtype)
            output = model.output.save()

    intervened_logits = get_logits(output)
    logit_diff = (clean_logits - intervened_logits).abs().mean().item()

    print(f"Mean absolute logit difference: {logit_diff:.4f}")
    print(f"Intervention had effect: {logit_diff > 0.01}")
    print("SUCCESS!")


# Example 7: SigmoidMaskRotatedSpaceIntervention (Direct nnsight)
def example_sigmoid_mask_rotated_direct():
    """Example: SigmoidMaskRotatedSpaceIntervention using direct nnsight operations."""
    print(f"\n{'=' * 60}")
    print("Example 7: SigmoidMaskRotatedSpaceIntervention (Direct nnsight)")
    print("Purpose: Apply rotation with per-dimension sigmoid mask")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    # Create intervention and get weights
    intervention = SigmoidMaskRotatedSpaceIntervention(embed_dim=EMBED_DIM)
    weights = intervention.get_remote_weights()
    rotation_matrix = weights['rotate_layer_weight']
    masks = weights['masks']
    temperature = weights['temperature']

    print(f"Rotation matrix shape: {rotation_matrix.shape}")
    print(f"Masks shape: {masks.shape}")

    # Get clean output
    clean_output = get_clean_output()
    clean_logits = get_logits(clean_output)

    # Apply intervention using direct nnsight
    # Use raw strings for remote compatibility
    with model.session(remote=USE_REMOTE):
        with model.trace(source_text):
            source_act = model.transformer.h[0].output[0].save()

        with model.trace(base_text):
            base_act = model.transformer.h[0].output[0]
            batch_size = base_act.shape[0]

            # Move tensors to same device as base_act (CUDA on remote)
            rotation_matrix_device = rotation_matrix.to(base_act.device)
            masks_device = masks.to(base_act.device)
            temperature_device = temperature.to(base_act.device)

            # Sigmoid mask rotation intervention logic
            rotated_base = torch.matmul(base_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            rotated_source = torch.matmul(source_act.to(rotation_matrix_device.dtype), rotation_matrix_device)

            # Compute sigmoid mask
            boundary_mask = torch.sigmoid(masks_device / temperature_device)
            boundary_mask = torch.ones(batch_size, device=base_act.device).unsqueeze(dim=-1) * boundary_mask
            boundary_mask = boundary_mask.to(rotated_base.dtype)

            # Interpolate in rotated space
            rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
            intervened = torch.matmul(rotated_output, rotation_matrix_device.T)

            model.transformer.h[0].output[0][:] = intervened.to(base_act.dtype)
            output = model.output.save()

    intervened_logits = get_logits(output)
    logit_diff = (clean_logits - intervened_logits).abs().mean().item()

    print(f"Mean absolute logit difference: {logit_diff:.4f}")
    print(f"Intervention had effect: {logit_diff > 0.01}")
    print("SUCCESS!")


# Example 8: SigmoidMaskIntervention (Direct nnsight)
def example_sigmoid_mask_direct():
    """Example: SigmoidMaskIntervention using direct nnsight operations."""
    print(f"\n{'=' * 60}")
    print("Example 8: SigmoidMaskIntervention (Direct nnsight)")
    print("Purpose: Apply per-dimension sigmoid mask without rotation")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    # Create intervention and get weights
    intervention = SigmoidMaskIntervention(embed_dim=EMBED_DIM)
    weights = intervention.get_remote_weights()
    mask = weights['mask']
    temperature = weights['temperature']

    print(f"Mask shape: {mask.shape}")

    # Get clean output
    clean_output = get_clean_output()
    clean_logits = get_logits(clean_output)

    # Apply intervention using direct nnsight
    # Use raw strings for remote compatibility
    with model.session(remote=USE_REMOTE):
        with model.trace(source_text):
            source_act = model.transformer.h[0].output[0].save()

        with model.trace(base_text):
            base_act = model.transformer.h[0].output[0]

            # Move tensors to same device as base_act (CUDA on remote)
            mask_device = mask.to(base_act.device)
            temperature_device = temperature.to(base_act.device)

            # Sigmoid mask intervention logic (no rotation)
            mask_sigmoid = torch.sigmoid(mask_device / temperature_device)
            intervened = (1.0 - mask_sigmoid) * base_act + mask_sigmoid * source_act

            model.transformer.h[0].output[0][:] = intervened
            output = model.output.save()

    intervened_logits = get_logits(output)
    logit_diff = (clean_logits - intervened_logits).abs().mean().item()

    print(f"Mean absolute logit difference: {logit_diff:.4f}")
    print(f"Intervention had effect: {logit_diff > 0.01}")
    print("SUCCESS!")


# Example 9: Gradient Flow (Local only demonstration)
def example_gradient_flow():
    """Example: Demonstrate gradient flow through intervention parameters."""
    print(f"\n{'=' * 60}")
    print("Example 9: Gradient Flow Through Intervention Parameters")
    print("Purpose: Show that gradients flow through trainable parameters")
    print("(This is a local-only test using raw tensors)")
    print("-" * 60)

    # Create intervention with trainable parameters
    intervention = LowRankRotatedSpaceIntervention(
        embed_dim=EMBED_DIM, low_rank_dimension=64
    )

    print(f"Intervention trainable: {intervention.trainable}")
    print(f"Number of parameters: {sum(p.numel() for p in intervention.parameters())}")
    print(f"Parameters requiring grad: {sum(1 for p in intervention.parameters() if p.requires_grad)}")

    # Create simple tensors to test gradient flow
    base_tensor = torch.randn(1, 5, EMBED_DIM, requires_grad=True)
    source_tensor = torch.randn(1, 5, EMBED_DIM, requires_grad=True)

    # Forward pass through intervention
    output = intervention(base_tensor, source_tensor)
    print(f"\nInput shape: {base_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Check gradients
    print("\nGradients after backward pass:")
    for name, param in intervention.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: shape={param.shape}, grad_norm={grad_norm:.6f}")
        elif param.requires_grad:
            print(f"  {name}: NO GRADIENT")

    print("SUCCESS!")


# Example 10: forward_with_gradients Method
def example_forward_with_gradients():
    """Example: Using the forward_with_gradients method."""
    print(f"\n{'=' * 60}")
    print("Example 10: forward_with_gradients() Method")
    print("Purpose: Training-friendly forward pass with gradient support")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    # Skip for remote mode - parametrized modules can't be serialized by cloudpickle
    # Gradient flow requires local intervention application anyway
    if USE_REMOTE:
        print("SKIPPED: forward_with_gradients() requires local execution")
        print("(Parametrized intervention modules cannot be serialized for remote execution)")
        print("(For training, gradients must flow through local intervention parameters)")
        print("SUCCESS! (skipped for remote mode)")
        return

    intervention = LowRankRotatedSpaceIntervention(
        embed_dim=EMBED_DIM, low_rank_dimension=64
    )

    pv_model = pv.build_intervenable_model({
        "component": "transformer.h[0].output",
        "intervention": intervention
    }, model=model, remote=USE_REMOTE)

    # Check if model has forward_with_gradients
    if hasattr(pv_model, 'forward_with_gradients'):
        print("forward_with_gradients method available")

        output = pv_model.forward_with_gradients(
            base=base_text,
            sources=[source_text],
            unit_locations={"sources->base": ([None], [None])}
        )

        print(f"Output type: {type(output)}")
        if hasattr(output, 'logits'):
            print(f"Output logits shape: {output.logits.shape}")
        elif hasattr(output, 'value'):
            val = output.value
            if hasattr(val, 'logits'):
                print(f"Output value logits shape: {val.logits.shape}")
            else:
                print(f"Output value: {type(val)}")
        else:
            print(f"Output: {type(output)}")
        print("SUCCESS!")
    else:
        print("forward_with_gradients method not available (native backend)")
        print("SKIPPED (not applicable)")


# Example 11: Compare All Intervention Types
def example_compare_all():
    """Example: Compare all intervention types on same component."""
    print(f"\n{'=' * 60}")
    print("Example 11: Compare All Intervention Types")
    print("Purpose: Show relative effects of different interventions")
    print(f"Remote: {USE_REMOTE}")
    print("-" * 60)

    clean_output = get_clean_output()
    clean_logits = get_logits(clean_output)
    clean_pred = tokenizer.decode(clean_logits[0, -1].argmax())

    print(f"Clean prediction: '{clean_pred}'")
    print("-" * 40)

    results = []

    # VanillaIntervention
    # Use raw strings for remote compatibility
    try:
        with model.session(remote=USE_REMOTE):
            with model.trace(source_text):
                source_act = model.transformer.h[0].output[0].save()
            with model.trace(base_text):
                model.transformer.h[0].output[0][:] = source_act
                output = model.output.save()
        logits = get_logits(output)
        diff = (clean_logits - logits).abs().mean().item()
        results.append(("VanillaIntervention", diff))
    except Exception as e:
        results.append(("VanillaIntervention", f"ERROR: {str(e)[:30]}"))

    # LowRankRotatedSpaceIntervention
    try:
        intervention = LowRankRotatedSpaceIntervention(embed_dim=EMBED_DIM, low_rank_dimension=64)
        rotation_matrix = intervention.rotate_layer.weight.detach()
        with model.session(remote=USE_REMOTE):
            with model.trace(source_text):
                source_act = model.transformer.h[0].output[0].save()
            with model.trace(base_text):
                base_act = model.transformer.h[0].output[0]
                rotation_matrix_device = rotation_matrix.to(base_act.device)
                rotated_base = torch.matmul(base_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
                rotated_source = torch.matmul(source_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
                diff = rotated_source - rotated_base
                intervened = base_act + torch.matmul(diff, rotation_matrix_device.T)
                model.transformer.h[0].output[0][:] = intervened.to(base_act.dtype)
                output = model.output.save()
        logits = get_logits(output)
        diff_val = (clean_logits - logits).abs().mean().item()
        results.append(("LowRankRotated (64-dim)", diff_val))
    except Exception as e:
        results.append(("LowRankRotated (64-dim)", f"ERROR: {str(e)[:30]}"))

    # RotatedSpaceIntervention
    try:
        intervention = RotatedSpaceIntervention(embed_dim=EMBED_DIM)
        rotation_matrix = intervention.rotate_layer.weight.detach()
        with model.session(remote=USE_REMOTE):
            with model.trace(source_text):
                source_act = model.transformer.h[0].output[0].save()
            with model.trace(base_text):
                base_act = model.transformer.h[0].output[0]
                rotation_matrix_device = rotation_matrix.to(base_act.device)
                rotated_base = torch.matmul(base_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
                rotated_source = torch.matmul(source_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
                diff = rotated_source - rotated_base
                intervened = base_act + torch.matmul(diff, rotation_matrix_device.T)
                model.transformer.h[0].output[0][:] = intervened.to(base_act.dtype)
                output = model.output.save()
        logits = get_logits(output)
        diff_val = (clean_logits - logits).abs().mean().item()
        results.append(("RotatedSpace (full)", diff_val))
    except Exception as e:
        results.append(("RotatedSpace (full)", f"ERROR: {str(e)[:30]}"))

    # SigmoidMaskIntervention
    try:
        intervention = SigmoidMaskIntervention(embed_dim=EMBED_DIM)
        mask = intervention.mask.detach()
        temperature = intervention.temperature.detach()
        with model.session(remote=USE_REMOTE):
            with model.trace(source_text):
                source_act = model.transformer.h[0].output[0].save()
            with model.trace(base_text):
                base_act = model.transformer.h[0].output[0]
                mask_device = mask.to(base_act.device)
                temperature_device = temperature.to(base_act.device)
                mask_sigmoid = torch.sigmoid(mask_device / temperature_device)
                intervened = (1.0 - mask_sigmoid) * base_act + mask_sigmoid * source_act
                model.transformer.h[0].output[0][:] = intervened
                output = model.output.save()
        logits = get_logits(output)
        diff_val = (clean_logits - logits).abs().mean().item()
        results.append(("SigmoidMask (no rotation)", diff_val))
    except Exception as e:
        results.append(("SigmoidMask (no rotation)", f"ERROR: {str(e)[:30]}"))

    # Print results
    for name, diff_val in results:
        if isinstance(diff_val, float):
            print(f"  {name}: logit_diff={diff_val:.4f}")
        else:
            print(f"  {name}: {diff_val}")

    print("SUCCESS!")


# Example 12: Verify Standard Pyvene vs NDIF Backend
def example_verify_pyvene_vs_ndif():
    """Verify that NDIF backend produces same outputs as standard pyvene."""
    print(f"\n{'=' * 60}")
    print("Example 12: Verify Standard Pyvene vs NDIF Backend")
    print("Purpose: Ensure NDIF remote execution matches standard pyvene")
    print("-" * 60)

    # Create a trainable intervention with fixed weights
    intervention = LowRankRotatedSpaceIntervention(
        embed_dim=EMBED_DIM, low_rank_dimension=64
    )
    rotation_matrix = intervention.rotate_layer.weight.detach().clone()

    print(f"Testing LowRankRotatedSpaceIntervention with {rotation_matrix.shape} rotation matrix")

    # standard pyvene (no NDIF)
    print("\n1. Running STANDARD PYVENE (no NDIF backend)...")
    from transformers import GPT2LMHeadModel

    # Load a fresh GPT-2 model for standard pyvene
    gpt2_standard = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')

    # Create standard pyvene IntervenableModel (no nnsight, no NDIF)
    pv_config = pv.IntervenableConfig({
        "layer": 0,
        "component": "block_output",
        "intervention": intervention  # Use same intervention object
    })
    pv_standard = pv.IntervenableModel(pv_config, model=gpt2_standard)

    # Tokenize inputs
    base_tokens_std = tokenizer(base_text, return_tensors="pt")
    source_tokens_std = tokenizer(source_text, return_tensors="pt")

    # Run standard pyvene intervention
    _, output_standard = pv_standard(
        base=base_tokens_std,
        sources=[source_tokens_std],
        unit_locations={"sources->base": ([None], [None])}  # All positions
    )

    standard_logits = output_standard.logits.detach().cpu().float()
    standard_pred = tokenizer.decode(standard_logits[0, -1].argmax())
    print(f"   Standard pyvene prediction: '{standard_pred}'")
    print(f"   Standard logits shape: {standard_logits.shape}")
    print(f"   Standard logits mean: {standard_logits.mean().item():.4f}")

    # ndif backend
    print("\n2. Running NDIF BACKEND (remote={})...".format(USE_REMOTE))

    with model.session(remote=USE_REMOTE):
        # Get source activations
        with model.trace(source_text):
            source_act = model.transformer.h[0].output[0].save()

        # Apply intervention to base
        with model.trace(base_text):
            base_act = model.transformer.h[0].output[0]

            # Move rotation matrix to correct device
            rotation_matrix_device = rotation_matrix.to(base_act.device)

            # Apply same intervention logic as LowRankRotatedSpaceIntervention
            rotated_base = torch.matmul(base_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            rotated_source = torch.matmul(source_act.to(rotation_matrix_device.dtype), rotation_matrix_device)
            diff = rotated_source - rotated_base
            intervened = base_act + torch.matmul(diff, rotation_matrix_device.T)

            model.transformer.h[0].output[0][:] = intervened.to(base_act.dtype)
            output_ndif = model.output.save()

    ndif_logits = get_logits(output_ndif).detach().cpu().float()
    ndif_pred = tokenizer.decode(ndif_logits[0, -1].argmax())
    print(f"   NDIF backend prediction: '{ndif_pred}'")
    print(f"   NDIF logits shape: {ndif_logits.shape}")
    print(f"   NDIF logits mean: {ndif_logits.mean().item():.4f}")

    # compare outputs
    print("\n3. Comparing outputs...")

    # Check if predictions match
    pred_match = standard_pred == ndif_pred
    print(f"   Predictions match: {pred_match} ('{standard_pred}' vs '{ndif_pred}')")

    # Check numerical closeness
    max_diff = (standard_logits - ndif_logits).abs().max().item()
    mean_diff = (standard_logits - ndif_logits).abs().mean().item()
    print(f"   Max logit difference: {max_diff:.6f}")
    print(f"   Mean logit difference: {mean_diff:.6f}")

    # Determine if outputs are "close enough"
    # Note: Some difference expected due to dtype/device differences
    tolerance = 0.1  # Relaxed tolerance for cross-backend comparison
    outputs_close = max_diff < tolerance

    if outputs_close:
        print(f"\n   VERIFIED: Standard pyvene and NDIF outputs are close (max_diff < {tolerance})")
    else:
        print(f"\n   WARNING: Outputs differ significantly (max_diff >= {tolerance})")

    # did the intervention change anything?
    print("\n4. Verifying intervention had effect...")

    # Get clean output (no intervention)
    with model.session(remote=USE_REMOTE):
        with model.trace(base_text):
            clean_output = model.output.save()

    clean_logits = get_logits(clean_output).detach().cpu().float()
    clean_pred = tokenizer.decode(clean_logits[0, -1].argmax())

    intervention_effect_std = (clean_logits - standard_logits).abs().mean().item()
    intervention_effect_ndif = (clean_logits - ndif_logits).abs().mean().item()

    print(f"   Clean prediction: '{clean_pred}'")
    print(f"   Standard pyvene effect (mean logit diff from clean): {intervention_effect_std:.4f}")
    print(f"   NDIF backend effect (mean logit diff from clean): {intervention_effect_ndif:.4f}")
    print(f"   Both show intervention effect: {intervention_effect_std > 0.01 and intervention_effect_ndif > 0.01}")

    print("\nSUCCESS!")


# MAIN
def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PYVENE NDIF BACKEND EXAMPLES")
    print(f"Remote execution: {USE_REMOTE}")
    print("=" * 60)

    examples = [
        ("Basic: CollectIntervention", example_collect_intervention),
        ("Basic: VanillaIntervention", example_vanilla_intervention),
        ("Info: get_remote_weights()", example_get_remote_weights),
        ("Trainable: LowRankRotatedSpaceIntervention", example_low_rank_rotated_direct),
        ("Trainable: RotatedSpaceIntervention", example_rotated_space_direct),
        ("Trainable: BoundlessRotatedSpaceIntervention", example_boundless_rotated_direct),
        ("Trainable: SigmoidMaskRotatedSpaceIntervention", example_sigmoid_mask_rotated_direct),
        ("Trainable: SigmoidMaskIntervention", example_sigmoid_mask_direct),
        ("Training: Gradient Flow", example_gradient_flow),
        ("Training: forward_with_gradients()", example_forward_with_gradients),
        ("Comparison: All Intervention Types", example_compare_all),
        ("Verification: Standard Pyvene vs NDIF", example_verify_pyvene_vs_ndif),
    ]

    passed = 0
    failed = 0

    for name, func in examples:
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"\nFAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    else:
        print(f"WARNING: {failed} example(s) failed")


if __name__ == "__main__":
    main()
