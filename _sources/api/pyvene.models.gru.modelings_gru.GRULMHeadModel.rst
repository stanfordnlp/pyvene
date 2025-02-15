pyvene.models.gru.modelings\_gru.GRULMHeadModel
===============================================

.. currentmodule:: pyvene.models.gru.modelings_gru

.. autoclass:: GRULMHeadModel
   :members:                                      
   :show-inheritance:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~GRULMHeadModel.__init__
      ~GRULMHeadModel.active_adapter
      ~GRULMHeadModel.active_adapters
      ~GRULMHeadModel.add_adapter
      ~GRULMHeadModel.add_memory_hooks
      ~GRULMHeadModel.add_model_tags
      ~GRULMHeadModel.add_module
      ~GRULMHeadModel.apply
      ~GRULMHeadModel.bfloat16
      ~GRULMHeadModel.buffers
      ~GRULMHeadModel.can_generate
      ~GRULMHeadModel.children
      ~GRULMHeadModel.compile
      ~GRULMHeadModel.compute_transition_scores
      ~GRULMHeadModel.cpu
      ~GRULMHeadModel.create_extended_attention_mask_for_decoder
      ~GRULMHeadModel.cuda
      ~GRULMHeadModel.delete_adapter
      ~GRULMHeadModel.dequantize
      ~GRULMHeadModel.disable_adapters
      ~GRULMHeadModel.disable_input_require_grads
      ~GRULMHeadModel.double
      ~GRULMHeadModel.enable_adapters
      ~GRULMHeadModel.enable_input_require_grads
      ~GRULMHeadModel.estimate_tokens
      ~GRULMHeadModel.eval
      ~GRULMHeadModel.extra_repr
      ~GRULMHeadModel.float
      ~GRULMHeadModel.floating_point_ops
      ~GRULMHeadModel.forward
      ~GRULMHeadModel.from_pretrained
      ~GRULMHeadModel.generate
      ~GRULMHeadModel.get_adapter_state_dict
      ~GRULMHeadModel.get_buffer
      ~GRULMHeadModel.get_compiled_call
      ~GRULMHeadModel.get_extended_attention_mask
      ~GRULMHeadModel.get_extra_state
      ~GRULMHeadModel.get_head_mask
      ~GRULMHeadModel.get_input_embeddings
      ~GRULMHeadModel.get_memory_footprint
      ~GRULMHeadModel.get_output_embeddings
      ~GRULMHeadModel.get_parameter
      ~GRULMHeadModel.get_position_embeddings
      ~GRULMHeadModel.get_submodule
      ~GRULMHeadModel.gradient_checkpointing_disable
      ~GRULMHeadModel.gradient_checkpointing_enable
      ~GRULMHeadModel.half
      ~GRULMHeadModel.heal_tokens
      ~GRULMHeadModel.init_weights
      ~GRULMHeadModel.invert_attention_mask
      ~GRULMHeadModel.ipu
      ~GRULMHeadModel.load_adapter
      ~GRULMHeadModel.load_state_dict
      ~GRULMHeadModel.modules
      ~GRULMHeadModel.mtia
      ~GRULMHeadModel.named_buffers
      ~GRULMHeadModel.named_children
      ~GRULMHeadModel.named_modules
      ~GRULMHeadModel.named_parameters
      ~GRULMHeadModel.num_parameters
      ~GRULMHeadModel.parameters
      ~GRULMHeadModel.post_init
      ~GRULMHeadModel.prepare_inputs_for_generation
      ~GRULMHeadModel.prune_heads
      ~GRULMHeadModel.push_to_hub
      ~GRULMHeadModel.register_backward_hook
      ~GRULMHeadModel.register_buffer
      ~GRULMHeadModel.register_for_auto_class
      ~GRULMHeadModel.register_forward_hook
      ~GRULMHeadModel.register_forward_pre_hook
      ~GRULMHeadModel.register_full_backward_hook
      ~GRULMHeadModel.register_full_backward_pre_hook
      ~GRULMHeadModel.register_load_state_dict_post_hook
      ~GRULMHeadModel.register_load_state_dict_pre_hook
      ~GRULMHeadModel.register_module
      ~GRULMHeadModel.register_parameter
      ~GRULMHeadModel.register_state_dict_post_hook
      ~GRULMHeadModel.register_state_dict_pre_hook
      ~GRULMHeadModel.requires_grad_
      ~GRULMHeadModel.reset_memory_hooks_state
      ~GRULMHeadModel.resize_position_embeddings
      ~GRULMHeadModel.resize_token_embeddings
      ~GRULMHeadModel.retrieve_modules_from_names
      ~GRULMHeadModel.reverse_bettertransformer
      ~GRULMHeadModel.save_pretrained
      ~GRULMHeadModel.set_adapter
      ~GRULMHeadModel.set_extra_state
      ~GRULMHeadModel.set_input_embeddings
      ~GRULMHeadModel.set_output_embeddings
      ~GRULMHeadModel.set_submodule
      ~GRULMHeadModel.share_memory
      ~GRULMHeadModel.state_dict
      ~GRULMHeadModel.tensor_parallel
      ~GRULMHeadModel.tie_weights
      ~GRULMHeadModel.to
      ~GRULMHeadModel.to_bettertransformer
      ~GRULMHeadModel.to_empty
      ~GRULMHeadModel.train
      ~GRULMHeadModel.type
      ~GRULMHeadModel.warn_if_padding_and_no_attention_mask
      ~GRULMHeadModel.xpu
      ~GRULMHeadModel.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~GRULMHeadModel.T_destination
      ~GRULMHeadModel.base_model
      ~GRULMHeadModel.base_model_prefix
      ~GRULMHeadModel.call_super_init
      ~GRULMHeadModel.config_class
      ~GRULMHeadModel.device
      ~GRULMHeadModel.dtype
      ~GRULMHeadModel.dummy_inputs
      ~GRULMHeadModel.dump_patches
      ~GRULMHeadModel.framework
      ~GRULMHeadModel.is_gradient_checkpointing
      ~GRULMHeadModel.is_parallelizable
      ~GRULMHeadModel.loss_function
      ~GRULMHeadModel.main_input_name
      ~GRULMHeadModel.model_tags
      ~GRULMHeadModel.supports_gradient_checkpointing
      ~GRULMHeadModel.supports_tp_plan
      ~GRULMHeadModel.training
   
   