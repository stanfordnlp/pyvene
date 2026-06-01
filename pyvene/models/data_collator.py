"""
Data collators for working with pyvene models.

Interchange interventions train on examples that carry two sets of inputs at
once. A base example contributes ``input_ids`` and ``attention_mask`` while the
source example contributes ``source_input_ids`` and ``source_attention_mask``.
The two sets usually have different lengths, so a single collator has to pad
both of them to their own per batch maximum.

Hugging Face ships ``DataCollatorForSeq2Seq``, which pads ``input_ids``,
``attention_mask`` and ``labels`` but leaves the source keys untouched. The
collator below mirrors that base behaviour and adds matching padding for the
source set so that a pyvene dataloader produces rectangular tensors for both
sides of an intervention.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForIntervention:
    """Collate batches that carry both base and source inputs for pyvene.

    The collator pads the base set (``input_ids`` and ``attention_mask``) and
    the source set (``source_input_ids`` and ``source_attention_mask``) to their
    own per batch maximum length, padding each independently the way
    ``DataCollatorForSeq2Seq`` pads the base set. When ``labels`` are present
    they are padded with ``label_pad_token_id`` on the same side the tokenizer
    pads. Any other key is stacked through the tokenizer as usual.

    Args:
        tokenizer: the tokenizer used to encode the data. It supplies the pad
            token id and the padding side.
        padding: passed straight to ``tokenizer.pad``. ``True`` or ``"longest"``
            pads to the per batch maximum.
        max_length: optional cap on the padded length.
        pad_to_multiple_of: pad the sequence to a multiple of this value.
        label_pad_token_id: id used to pad ``labels`` so the loss can ignore it.
        return_tensors: tensor type returned by the tokenizer, ``"pt"`` for
            PyTorch tensors.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pull the source keys out so the base set can be padded on its own,
        # then strip the "source_" prefix and pad that set the same way.
        source_features = []
        for feature in features:
            source_feature = {}
            if "source_input_ids" in feature:
                source_feature["input_ids"] = feature["source_input_ids"]
            if "source_attention_mask" in feature:
                source_feature["attention_mask"] = feature["source_attention_mask"]
            source_features.append(source_feature)

        base_features = [
            {
                key: value
                for key, value in feature.items()
                if key not in ("source_input_ids", "source_attention_mask")
            }
            for feature in features
        ]

        batch = self._pad_with_labels(base_features)

        if any(len(source_feature) > 0 for source_feature in source_features):
            source_batch = self._pad_with_labels(source_features)
            if "input_ids" in source_batch:
                batch["source_input_ids"] = source_batch["input_ids"]
            if "attention_mask" in source_batch:
                batch["source_attention_mask"] = source_batch["attention_mask"]

        return batch

    def _pad_with_labels(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad a single set of features, handling ``labels`` separately.

        ``tokenizer.pad`` only knows how to pad ``input_ids`` and
        ``attention_mask``, so ``labels`` are padded by hand with
        ``label_pad_token_id`` on the tokenizer's padding side, matching the
        behaviour of ``DataCollatorForSeq2Seq``.
        """
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0]
            else None
        )

        no_labels_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if labels is not None:
            # When the caller requests ``padding="max_length"`` with a
            # ``max_length``, ``tokenizer.pad`` pads ``input_ids`` out to that
            # length, so the labels have to match it. Otherwise pad to the
            # longest label in the batch, mirroring ``DataCollatorForSeq2Seq``.
            if (
                self.padding in ("max_length", PaddingStrategy.MAX_LENGTH)
                and self.max_length is not None
            ):
                max_label_length = self.max_length
            else:
                max_label_length = max(len(label) for label in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

            padding_side = self.tokenizer.padding_side
            padded_labels = []
            for label in labels:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(label)
                )
                label = list(label)
                if padding_side == "right":
                    padded_labels.append(label + remainder)
                else:
                    padded_labels.append(remainder + label)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.int64)

        return batch
