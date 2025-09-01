pyvene.models.backpack\_gpt2.modelings\_backpack\_gpt2.BackpackGPT2Model
========================================================================

.. currentmodule:: pyvene.models.backpack_gpt2.modelings_backpack_gpt2

.. autoclass:: BackpackGPT2Model
   :members:                                      
   :show-inheritance:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~BackpackGPT2Model.__init__
      ~BackpackGPT2Model.active_adapter
      ~BackpackGPT2Model.active_adapters
      ~BackpackGPT2Model.add_adapter
      ~BackpackGPT2Model.add_memory_hooks
      ~BackpackGPT2Model.add_model_tags
      ~BackpackGPT2Model.add_module
      ~BackpackGPT2Model.apply
      ~BackpackGPT2Model.bfloat16
      ~BackpackGPT2Model.buffers
      ~BackpackGPT2Model.can_generate
      ~BackpackGPT2Model.children
      ~BackpackGPT2Model.compile
      ~BackpackGPT2Model.cpu
      ~BackpackGPT2Model.create_extended_attention_mask_for_decoder
      ~BackpackGPT2Model.cuda
      ~BackpackGPT2Model.delete_adapter
      ~BackpackGPT2Model.dequantize
      ~BackpackGPT2Model.disable_adapters
      ~BackpackGPT2Model.disable_input_require_grads
      ~BackpackGPT2Model.double
      ~BackpackGPT2Model.enable_adapters
      ~BackpackGPT2Model.enable_input_require_grads
      ~BackpackGPT2Model.estimate_tokens
      ~BackpackGPT2Model.eval
      ~BackpackGPT2Model.extra_repr
      ~BackpackGPT2Model.float
      ~BackpackGPT2Model.floating_point_ops
      ~BackpackGPT2Model.forward
      ~BackpackGPT2Model.from_pretrained
      ~BackpackGPT2Model.get_adapter_state_dict
      ~BackpackGPT2Model.get_buffer
      ~BackpackGPT2Model.get_compiled_call
      ~BackpackGPT2Model.get_correct_attn_implementation
      ~BackpackGPT2Model.get_decoder
      ~BackpackGPT2Model.get_extended_attention_mask
      ~BackpackGPT2Model.get_extra_state
      ~BackpackGPT2Model.get_head_mask
      ~BackpackGPT2Model.get_init_context
      ~BackpackGPT2Model.get_input_embeddings
      ~BackpackGPT2Model.get_memory_footprint
      ~BackpackGPT2Model.get_num_senses
      ~BackpackGPT2Model.get_output_embeddings
      ~BackpackGPT2Model.get_parameter
      ~BackpackGPT2Model.get_parameter_or_buffer
      ~BackpackGPT2Model.get_position_embeddings
      ~BackpackGPT2Model.get_sense_network
      ~BackpackGPT2Model.get_submodule
      ~BackpackGPT2Model.get_word_embeddings
      ~BackpackGPT2Model.gradient_checkpointing_disable
      ~BackpackGPT2Model.gradient_checkpointing_enable
      ~BackpackGPT2Model.half
      ~BackpackGPT2Model.init_weights
      ~BackpackGPT2Model.initialize_weights
      ~BackpackGPT2Model.invert_attention_mask
      ~BackpackGPT2Model.ipu
      ~BackpackGPT2Model.is_backend_compatible
      ~BackpackGPT2Model.load_adapter
      ~BackpackGPT2Model.load_state_dict
      ~BackpackGPT2Model.load_tf_weights
      ~BackpackGPT2Model.modules
      ~BackpackGPT2Model.mtia
      ~BackpackGPT2Model.named_buffers
      ~BackpackGPT2Model.named_children
      ~BackpackGPT2Model.named_modules
      ~BackpackGPT2Model.named_parameters
      ~BackpackGPT2Model.num_parameters
      ~BackpackGPT2Model.parameters
      ~BackpackGPT2Model.post_init
      ~BackpackGPT2Model.prune_heads
      ~BackpackGPT2Model.push_to_hub
      ~BackpackGPT2Model.register_backward_hook
      ~BackpackGPT2Model.register_buffer
      ~BackpackGPT2Model.register_for_auto_class
      ~BackpackGPT2Model.register_forward_hook
      ~BackpackGPT2Model.register_forward_pre_hook
      ~BackpackGPT2Model.register_full_backward_hook
      ~BackpackGPT2Model.register_full_backward_pre_hook
      ~BackpackGPT2Model.register_load_state_dict_post_hook
      ~BackpackGPT2Model.register_load_state_dict_pre_hook
      ~BackpackGPT2Model.register_module
      ~BackpackGPT2Model.register_parameter
      ~BackpackGPT2Model.register_state_dict_post_hook
      ~BackpackGPT2Model.register_state_dict_pre_hook
      ~BackpackGPT2Model.requires_grad_
      ~BackpackGPT2Model.reset_memory_hooks_state
      ~BackpackGPT2Model.resize_position_embeddings
      ~BackpackGPT2Model.resize_token_embeddings
      ~BackpackGPT2Model.retrieve_modules_from_names
      ~BackpackGPT2Model.reverse_bettertransformer
      ~BackpackGPT2Model.run_with_custom_contextualization
      ~BackpackGPT2Model.save_pretrained
      ~BackpackGPT2Model.set_adapter
      ~BackpackGPT2Model.set_attn_implementation
      ~BackpackGPT2Model.set_decoder
      ~BackpackGPT2Model.set_extra_state
      ~BackpackGPT2Model.set_input_embeddings
      ~BackpackGPT2Model.set_output_embeddings
      ~BackpackGPT2Model.set_submodule
      ~BackpackGPT2Model.share_memory
      ~BackpackGPT2Model.state_dict
      ~BackpackGPT2Model.tie_embeddings_and_encoder_decoder
      ~BackpackGPT2Model.tie_weights
      ~BackpackGPT2Model.to
      ~BackpackGPT2Model.to_bettertransformer
      ~BackpackGPT2Model.to_empty
      ~BackpackGPT2Model.train
      ~BackpackGPT2Model.type
      ~BackpackGPT2Model.warn_if_padding_and_no_attention_mask
      ~BackpackGPT2Model.xpu
      ~BackpackGPT2Model.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~BackpackGPT2Model.T_destination
      ~BackpackGPT2Model.base_model
      ~BackpackGPT2Model.base_model_prefix
      ~BackpackGPT2Model.call_super_init
      ~BackpackGPT2Model.can_record_outputs
      ~BackpackGPT2Model.device
      ~BackpackGPT2Model.dtype
      ~BackpackGPT2Model.dummy_inputs
      ~BackpackGPT2Model.dump_patches
      ~BackpackGPT2Model.framework
      ~BackpackGPT2Model.is_gradient_checkpointing
      ~BackpackGPT2Model.is_parallelizable
      ~BackpackGPT2Model.loss_function
      ~BackpackGPT2Model.main_input_name
      ~BackpackGPT2Model.model_tags
      ~BackpackGPT2Model.pp_plan
      ~BackpackGPT2Model.supports_gradient_checkpointing
      ~BackpackGPT2Model.supports_pp_plan
      ~BackpackGPT2Model.supports_tp_plan
      ~BackpackGPT2Model.tp_plan
      ~BackpackGPT2Model.tp_size
      ~BackpackGPT2Model.config
      ~BackpackGPT2Model.training
   
   