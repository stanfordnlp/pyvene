pyvene.models.mlp.modelings\_mlp.MLPModel
=========================================

.. currentmodule:: pyvene.models.mlp.modelings_mlp

.. autoclass:: MLPModel
   :members:                                      
   :show-inheritance:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~MLPModel.__init__
      ~MLPModel.active_adapter
      ~MLPModel.active_adapters
      ~MLPModel.add_adapter
      ~MLPModel.add_memory_hooks
      ~MLPModel.add_model_tags
      ~MLPModel.add_module
      ~MLPModel.apply
      ~MLPModel.bfloat16
      ~MLPModel.buffers
      ~MLPModel.can_generate
      ~MLPModel.children
      ~MLPModel.compile
      ~MLPModel.compute_transition_scores
      ~MLPModel.cpu
      ~MLPModel.create_extended_attention_mask_for_decoder
      ~MLPModel.cuda
      ~MLPModel.delete_adapter
      ~MLPModel.dequantize
      ~MLPModel.disable_adapters
      ~MLPModel.disable_input_require_grads
      ~MLPModel.double
      ~MLPModel.enable_adapters
      ~MLPModel.enable_input_require_grads
      ~MLPModel.estimate_tokens
      ~MLPModel.eval
      ~MLPModel.extra_repr
      ~MLPModel.float
      ~MLPModel.floating_point_ops
      ~MLPModel.forward
      ~MLPModel.from_pretrained
      ~MLPModel.generate
      ~MLPModel.get_adapter_state_dict
      ~MLPModel.get_buffer
      ~MLPModel.get_compiled_call
      ~MLPModel.get_extended_attention_mask
      ~MLPModel.get_extra_state
      ~MLPModel.get_head_mask
      ~MLPModel.get_input_embeddings
      ~MLPModel.get_memory_footprint
      ~MLPModel.get_output_embeddings
      ~MLPModel.get_parameter
      ~MLPModel.get_position_embeddings
      ~MLPModel.get_submodule
      ~MLPModel.gradient_checkpointing_disable
      ~MLPModel.gradient_checkpointing_enable
      ~MLPModel.half
      ~MLPModel.heal_tokens
      ~MLPModel.init_weights
      ~MLPModel.invert_attention_mask
      ~MLPModel.ipu
      ~MLPModel.load_adapter
      ~MLPModel.load_state_dict
      ~MLPModel.modules
      ~MLPModel.mtia
      ~MLPModel.named_buffers
      ~MLPModel.named_children
      ~MLPModel.named_modules
      ~MLPModel.named_parameters
      ~MLPModel.num_parameters
      ~MLPModel.parameters
      ~MLPModel.post_init
      ~MLPModel.prepare_inputs_for_generation
      ~MLPModel.prune_heads
      ~MLPModel.push_to_hub
      ~MLPModel.register_backward_hook
      ~MLPModel.register_buffer
      ~MLPModel.register_for_auto_class
      ~MLPModel.register_forward_hook
      ~MLPModel.register_forward_pre_hook
      ~MLPModel.register_full_backward_hook
      ~MLPModel.register_full_backward_pre_hook
      ~MLPModel.register_load_state_dict_post_hook
      ~MLPModel.register_load_state_dict_pre_hook
      ~MLPModel.register_module
      ~MLPModel.register_parameter
      ~MLPModel.register_state_dict_post_hook
      ~MLPModel.register_state_dict_pre_hook
      ~MLPModel.requires_grad_
      ~MLPModel.reset_memory_hooks_state
      ~MLPModel.resize_position_embeddings
      ~MLPModel.resize_token_embeddings
      ~MLPModel.retrieve_modules_from_names
      ~MLPModel.reverse_bettertransformer
      ~MLPModel.save_pretrained
      ~MLPModel.set_adapter
      ~MLPModel.set_extra_state
      ~MLPModel.set_input_embeddings
      ~MLPModel.set_submodule
      ~MLPModel.share_memory
      ~MLPModel.state_dict
      ~MLPModel.tensor_parallel
      ~MLPModel.tie_weights
      ~MLPModel.to
      ~MLPModel.to_bettertransformer
      ~MLPModel.to_empty
      ~MLPModel.train
      ~MLPModel.type
      ~MLPModel.warn_if_padding_and_no_attention_mask
      ~MLPModel.xpu
      ~MLPModel.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~MLPModel.T_destination
      ~MLPModel.base_model
      ~MLPModel.base_model_prefix
      ~MLPModel.call_super_init
      ~MLPModel.config_class
      ~MLPModel.device
      ~MLPModel.dtype
      ~MLPModel.dummy_inputs
      ~MLPModel.dump_patches
      ~MLPModel.framework
      ~MLPModel.is_gradient_checkpointing
      ~MLPModel.is_parallelizable
      ~MLPModel.loss_function
      ~MLPModel.main_input_name
      ~MLPModel.model_tags
      ~MLPModel.supports_gradient_checkpointing
      ~MLPModel.supports_tp_plan
      ~MLPModel.training
   
   