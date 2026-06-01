import unittest

import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from pyvene.models.data_collator import DataCollatorForIntervention


def build_tiny_tokenizer():
    """Build a hermetic word level tokenizer so the test needs no downloads."""
    vocab = {f"tok{i}": i for i in range(16)}
    vocab["[PAD]"] = 16
    tokenizer_object = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object, pad_token="[PAD]"
    )


class DataCollatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = build_tiny_tokenizer()
        self.pad_id = self.tokenizer.pad_token_id
        self.collator = DataCollatorForIntervention(tokenizer=self.tokenizer)

    def test_both_sides_pad_to_batch_max(self):
        # The base set and the source set have different per batch maxima:
        # base maxes out at length 4, source maxes out at length 5.
        features = [
            {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "source_input_ids": [5, 6],
                "source_attention_mask": [1, 1],
            },
            {
                "input_ids": [7, 8],
                "attention_mask": [1, 1],
                "source_input_ids": [9, 10, 11, 12, 13],
                "source_attention_mask": [1, 1, 1, 1, 1],
            },
        ]

        batch = self.collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(batch["source_input_ids"].shape, (2, 5))
        self.assertEqual(batch["source_attention_mask"].shape, (2, 5))

    def test_base_padding_values(self):
        features = [
            {"input_ids": [1, 2, 3, 4], "source_input_ids": [5, 6]},
            {"input_ids": [7, 8], "source_input_ids": [9, 10, 11]},
        ]

        batch = self.collator(features)

        # The shorter base row is padded on the right with the pad id, and the
        # attention mask marks the padded position with a zero.
        self.assertTrue(
            torch.equal(batch["input_ids"][1], torch.tensor([7, 8, self.pad_id, self.pad_id]))
        )
        self.assertTrue(
            torch.equal(batch["attention_mask"][1], torch.tensor([1, 1, 0, 0]))
        )

    def test_source_padding_values(self):
        features = [
            {"input_ids": [1, 2], "source_input_ids": [5, 6]},
            {"input_ids": [7, 8], "source_input_ids": [9, 10, 11]},
        ]

        batch = self.collator(features)

        # The shorter source row is padded on the right with the pad id, and the
        # generated source attention mask zeroes out the padded position.
        self.assertTrue(
            torch.equal(batch["source_input_ids"][0], torch.tensor([5, 6, self.pad_id]))
        )
        self.assertTrue(
            torch.equal(batch["source_attention_mask"][0], torch.tensor([1, 1, 0]))
        )

    def test_labels_padded_with_ignore_index(self):
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "source_input_ids": [5, 6]},
            {"input_ids": [7, 8], "labels": [7, 8], "source_input_ids": [9, 10, 11]},
        ]

        batch = self.collator(features)

        self.assertEqual(batch["labels"].shape, (2, 3))
        self.assertTrue(
            torch.equal(
                batch["labels"][1],
                torch.tensor([7, 8, self.collator.label_pad_token_id]),
            )
        )

    def test_labels_pad_to_max_length(self):
        # With padding="max_length" and a max_length, tokenizer.pad pads
        # input_ids out to max_length, so the labels have to be padded to the
        # same length rather than to the per batch maximum. Otherwise the loss
        # compares logits and labels of different lengths.
        collator = DataCollatorForIntervention(
            tokenizer=self.tokenizer, padding="max_length", max_length=6
        )
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "source_input_ids": [5, 6]},
            {"input_ids": [7, 8], "labels": [7, 8], "source_input_ids": [9, 10, 11]},
        ]

        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["labels"].shape, (2, 6))
        self.assertTrue(
            torch.equal(
                batch["labels"][0],
                torch.tensor([1, 2, 3] + [collator.label_pad_token_id] * 3),
            )
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DataCollatorTestCase("test_both_sides_pad_to_batch_max"))
    suite.addTest(DataCollatorTestCase("test_base_padding_values"))
    suite.addTest(DataCollatorTestCase("test_source_padding_values"))
    suite.addTest(DataCollatorTestCase("test_labels_padded_with_ignore_index"))
    suite.addTest(DataCollatorTestCase("test_labels_pad_to_max_length"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
