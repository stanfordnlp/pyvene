pyvene.models.gru.modelings\_gru.GRUPreTrainedModel
===================================================

.. currentmodule:: pyvene.models.gru.modelings_gru

.. autoclass:: GRUPreTrainedModel
   :members:                                      
   :show-inheritance:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~GRUPreTrainedModel.__init__
      ~GRUPreTrainedModel.active_adapter
      ~GRUPreTrainedModel.active_adapters
      ~GRUPreTrainedModel.add_adapter
      ~GRUPreTrainedModel.add_memory_hooks
      ~GRUPreTrainedModel.add_model_tags
      ~GRUPreTrainedModel.add_module
      ~GRUPreTrainedModel.apply
      ~GRUPreTrainedModel.bfloat16
      ~GRUPreTrainedModel.buffers
      ~GRUPreTrainedModel.can_generate
      ~GRUPreTrainedModel.children
      ~GRUPreTrainedModel.compile
      ~GRUPreTrainedModel.compute_transition_scores
      ~GRUPreTrainedModel.cpu
      ~GRUPreTrainedModel.create_extended_attention_mask_for_decoder
      ~GRUPreTrainedModel.cuda
      ~GRUPreTrainedModel.delete_adapter
      ~GRUPreTrainedModel.dequantize
      ~GRUPreTrainedModel.disable_adapters
      ~GRUPreTrainedModel.disable_input_require_grads
      ~GRUPreTrainedModel.double
      ~GRUPreTrainedModel.enable_adapters
      ~GRUPreTrainedModel.enable_input_require_grads
      ~GRUPreTrainedModel.estimate_tokens
      ~GRUPreTrainedModel.eval
      ~GRUPreTrainedModel.extra_repr
      ~GRUPreTrainedModel.float
      ~GRUPreTrainedModel.floating_point_ops
      ~GRUPreTrainedModel.forward
      ~GRUPreTrainedModel.from_pretrained
      ~GRUPreTrainedModel.generate
      ~GRUPreTrainedModel.get_adapter_state_dict
      ~GRUPreTrainedModel.get_buffer
      ~GRUPreTrainedModel.get_compiled_call
      ~GRUPreTrainedModel.get_extended_attention_mask
      ~GRUPreTrainedModel.get_extra_state
      ~GRUPreTrainedModel.get_head_mask
      ~GRUPreTrainedModel.get_input_embeddings
      ~GRUPreTrainedModel.get_memory_footprint
      ~GRUPreTrainedModel.get_output_embeddings
      ~GRUPreTrainedModel.get_parameter
      ~GRUPreTrainedModel.get_position_embeddings
      ~GRUPreTrainedModel.get_submodule
      ~GRUPreTrainedModel.gradient_checkpointing_disable
      ~GRUPreTrainedModel.gradient_checkpointing_enable
      ~GRUPreTrainedModel.half
      ~GRUPreTrainedModel.heal_tokens
      ~GRUPreTrainedModel.init_weights
      ~GRUPreTrainedModel.invert_attention_mask
      ~GRUPreTrainedModel.ipu
      ~GRUPreTrainedModel.load_adapter
      ~GRUPreTrainedModel.load_state_dict
      ~GRUPreTrainedModel.modules
      ~GRUPreTrainedModel.mtia
      ~GRUPreTrainedModel.named_buffers
      ~GRUPreTrainedModel.named_children
      ~GRUPreTrainedModel.named_modules
      ~GRUPreTrainedModel.named_parameters
      ~GRUPreTrainedModel.num_parameters
      ~GRUPreTrainedModel.parameters
      ~GRUPreTrainedModel.post_init
      ~GRUPreTrainedModel.prepare_inputs_for_generation
      ~GRUPreTrainedModel.prune_heads
      ~GRUPreTrainedModel.push_to_hub
      ~GRUPreTrainedModel.register_backward_hook
      ~GRUPreTrainedModel.register_buffer
      ~GRUPreTrainedModel.register_for_auto_class
      ~GRUPreTrainedModel.register_forward_hook
      ~GRUPreTrainedModel.register_forward_pre_hook
      ~GRUPreTrainedModel.register_full_backward_hook
      ~GRUPreTrainedModel.register_full_backward_pre_hook
      ~GRUPreTrainedModel.register_load_state_dict_post_hook
      ~GRUPreTrainedModel.register_load_state_dict_pre_hook
      ~GRUPreTrainedModel.register_module
      ~GRUPreTrainedModel.register_parameter
      ~GRUPreTrainedModel.register_state_dict_post_hook
      ~GRUPreTrainedModel.register_state_dict_pre_hook
      ~GRUPreTrainedModel.requires_grad_
      ~GRUPreTrainedModel.reset_memory_hooks_state
      ~GRUPreTrainedModel.resize_position_embeddings
      ~GRUPreTrainedModel.resize_token_embeddings
      ~GRUPreTrainedModel.retrieve_modules_from_names
      ~GRUPreTrainedModel.reverse_bettertransformer
      ~GRUPreTrainedModel.save_pretrained
      ~GRUPreTrainedModel.set_adapter
      ~GRUPreTrainedModel.set_extra_state
      ~GRUPreTrainedModel.set_input_embeddings
      ~GRUPreTrainedModel.set_submodule
      ~GRUPreTrainedModel.share_memory
      ~GRUPreTrainedModel.state_dict
      ~GRUPreTrainedModel.tensor_parallel
      ~GRUPreTrainedModel.tie_weights
      ~GRUPreTrainedModel.to
      ~GRUPreTrainedModel.to_bettertransformer
      ~GRUPreTrainedModel.to_empty
      ~GRUPreTrainedModel.train
      ~GRUPreTrainedModel.type
      ~GRUPreTrainedModel.warn_if_padding_and_no_attention_mask
      ~GRUPreTrainedModel.xpu
      ~GRUPreTrainedModel.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~GRUPreTrainedModel.T_destination
      ~GRUPreTrainedModel.base_model
      ~GRUPreTrainedModel.base_model_prefix
      ~GRUPreTrainedModel.call_super_init
      ~GRUPreTrainedModel.config_class
      ~GRUPreTrainedModel.device
      ~GRUPreTrainedModel.dtype
      ~GRUPreTrainedModel.dummy_inputs
      ~GRUPreTrainedModel.dump_patches
      ~GRUPreTrainedModel.framework
      ~GRUPreTrainedModel.is_gradient_checkpointing
      ~GRUPreTrainedModel.is_parallelizable
      ~GRUPreTrainedModel.loss_function
      ~GRUPreTrainedModel.main_input_name
      ~GRUPreTrainedModel.model_tags
      ~GRUPreTrainedModel.supports_gradient_checkpointing
      ~GRUPreTrainedModel.supports_tp_plan
      ~GRUPreTrainedModel.training
   
   