import torch
import torch.nn as nn
from transformers import BlipForQuestionAnswering, BlipConfig
from transformers.utils import ModelOutput
from typing import Optional, Union, Tuple, Dict


class BlipWrapper(nn.Module):
    def __init__(self, model: BlipForQuestionAnswering):
        super(BlipWrapper, self).__init__()
        self.model_vis = model.vision_model
        self.model_text_enc = model.text_encoder
        self.model_text_dec = model.text_decoder
        self.decoder_pad_token_id = model.decoder_pad_token_id
        self.decoder_start_token_id = model.decoder_start_token_id
        self.config = model.config
        self.eos_token_id = (model.config.text_config.sep_token_id,)
        self.pad_token_id = model.config.text_config.pad_token_id
        self.output_attentions = model.config.output_attentions
        self.use_return_dict = model.config.use_return_dict
        self.output_hidden_states = model.config.output_hidden_states

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
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        input_ids = input_ids.to(self.model_text_enc.device)
        question_embeds = self.model_text_enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_hidden_states=True,
        )

        question_embeds_w = (
            question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        )

        bos_ids = torch.full(
            (question_embeds_w.size(0), 1),
            fill_value=self.decoder_start_token_id,
            device=self.model_text_enc.device,
        )

        answer_output = self.model_text_dec(
            input_ids=bos_ids,
            encoder_hidden_states=question_embeds_w,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
            reduction="mean",
        )

        return {
            "decoder_logits": answer_output.logits,
            "image_embeds": image_embeds,
            "encoder_last_hidden_state": question_embeds.last_hidden_state,
            "encoder_hidden_states": question_embeds.hidden_states,
            "decoder_hidden_states": answer_output.hidden_states,
        }
