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
      ~MLPModel.active_adapters
      ~MLPModel.add_adapter
      ~MLPModel.add_model_tags
      ~MLPModel.add_module
      ~MLPModel.apply
      ~MLPModel.bfloat16
      ~MLPModel.buffers
      ~MLPModel.can_generate
      ~MLPModel.children
      ~MLPModel.compile
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
      ~MLPModel.enable_peft_hotswap
      ~MLPModel.eval
      ~MLPModel.extra_repr
      ~MLPModel.float
      ~MLPModel.forward
      ~MLPModel.from_pretrained
      ~MLPModel.get_adapter_state_dict
      ~MLPModel.get_buffer
      ~MLPModel.get_compiled_call
      ~MLPModel.get_correct_attn_implementation
      ~MLPModel.get_correct_experts_implementation
      ~MLPModel.get_decoder
      ~MLPModel.get_encoder
      ~MLPModel.get_expanded_tied_weights_keys
      ~MLPModel.get_extended_attention_mask
      ~MLPModel.get_extra_state
      ~MLPModel.get_init_context
      ~MLPModel.get_input_embeddings
      ~MLPModel.get_memory_footprint
      ~MLPModel.get_output_embeddings
      ~MLPModel.get_parameter
      ~MLPModel.get_parameter_or_buffer
      ~MLPModel.get_position_embeddings
      ~MLPModel.get_submodule
      ~MLPModel.gradient_checkpointing_disable
      ~MLPModel.gradient_checkpointing_enable
      ~MLPModel.half
      ~MLPModel.init_weights
      ~MLPModel.initialize_weights
      ~MLPModel.invert_attention_mask
      ~MLPModel.ipu
      ~MLPModel.is_backend_compatible
      ~MLPModel.is_remote_code
      ~MLPModel.kernelize
      ~MLPModel.load_adapter
      ~MLPModel.load_state_dict
      ~MLPModel.mark_tied_weights_as_initialized
      ~MLPModel.modules
      ~MLPModel.mtia
      ~MLPModel.named_buffers
      ~MLPModel.named_children
      ~MLPModel.named_modules
      ~MLPModel.named_non_persistent_buffers
      ~MLPModel.named_parameters
      ~MLPModel.num_parameters
      ~MLPModel.parameters
      ~MLPModel.post_init
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
      ~MLPModel.resize_position_embeddings
      ~MLPModel.resize_token_embeddings
      ~MLPModel.retrieve_modules_from_names
      ~MLPModel.save_pretrained
      ~MLPModel.set_adapter
      ~MLPModel.set_attn_implementation
      ~MLPModel.set_decoder
      ~MLPModel.set_encoder
      ~MLPModel.set_experts_implementation
      ~MLPModel.set_extra_state
      ~MLPModel.set_input_embeddings
      ~MLPModel.set_output_embeddings
      ~MLPModel.set_submodule
      ~MLPModel.set_use_kernels
      ~MLPModel.share_memory
      ~MLPModel.state_dict
      ~MLPModel.tie_weights
      ~MLPModel.to
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
      ~MLPModel.can_record_outputs
      ~MLPModel.config_class
      ~MLPModel.device
      ~MLPModel.dtype
      ~MLPModel.dummy_inputs
      ~MLPModel.dump_patches
      ~MLPModel.input_modalities
      ~MLPModel.is_gradient_checkpointing
      ~MLPModel.loss_function
      ~MLPModel.main_input_name
      ~MLPModel.model_tags
      ~MLPModel.pp_plan
      ~MLPModel.supports_gradient_checkpointing
      ~MLPModel.supports_pp_plan
      ~MLPModel.supports_tp_plan
      ~MLPModel.tp_plan
      ~MLPModel.tp_size
      ~MLPModel.use_kernels
      ~MLPModel.training
   
   