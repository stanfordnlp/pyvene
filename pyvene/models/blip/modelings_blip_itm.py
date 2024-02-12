import torch
import torch.nn as nn
from transformers import BlipConfig, BlipForImageTextRetrieval
from transformers.utils import ModelOutput
from typing import Optional, Union, Tuple, Dict


class BlipITMWrapper(nn.Module):
    def __init__(
        self, model: BlipForImageTextRetrieval, use_itm_not_contrastive: bool = True
    ):
        super(BlipITMWrapper, self).__init__()
        self.model_vis = model.vision_model
        self.model_text_enc = model.text_encoder
        self.model_vis_proj = model.vision_proj
        self.model_text_proj = model.text_proj
        self.model_itm = model.itm_head
        # do I need to keep decoder_pad_token_id and decoder_start_token_id? might be a mistake in the HF implementation lol
        self.config = model.config
        self.eos_token_id = (model.config.text_config.sep_token_id,)
        self.pad_token_id = model.config.text_config.pad_token_id
        self.output_attentions = model.config.output_attentions
        self.use_return_dict = model.config.use_return_dict
        self.output_hidden_states = model.config.output_hidden_states

        self.use_itm_head = use_itm_not_contrastive

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.output_hidden_states
        )

        vision_outputs = self.model_vis(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0].to(self.model_text_enc.device)
        input_ids = input_ids.to(self.model_text_enc.device)
        
        if self.use_itm_head:
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            )

            caption_embeds = self.model_text_enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_hidden_states=True,
            )
            caption_embeds = (
                caption_embeds[0]
                if not return_dict
                else caption_embeds.last_hidden_state
            )

            output = self.model_itm(caption_embeds[:, 0, :])
        else:
            caption_embeds = self.model_text_enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                output_hidden_states=True,
            )
            caption_embeds = (
                caption_embeds[0]
                if not return_dict
                else caption_embeds.last_hidden_state
            )

            image_feat = nn.functional.normalize(
                self.vision_proj(image_embeds[:, 0, :]), dim=-1
            )
            text_feat = nn.functional.normalize(
                self.text_proj(caption_embeds[:, 0, :]), dim=-1
            )

            output = image_feat @ text_feat.t()

        return {
            "itm_score": output,
            "image_embeds": image_embeds,
            "encoder_last_hidden_state": caption_embeds.last_hidden_state,
            "encoder_hidden_states": caption_embeds.hidden_states,
        }
