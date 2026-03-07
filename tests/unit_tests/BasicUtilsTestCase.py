import unittest
import torch
from ..utils import (
    create_gpt2_lm,
    create_llama,
    embed_to_distrib,
    GPT2Config,
    LlamaConfig,
)


class BasicUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: BasicUtilsTestCase ===")
        cls.gpt2_config, cls.gpt2_tokenizer, cls.gpt2 = create_gpt2_lm(
            config=GPT2Config(
                n_embd=24,
                attn_pdrop=0.0,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
                summary_first_dropout=0.0,
                n_layer=2,
                bos_token_id=0,
                eos_token_id=0,
                n_positions=128,
                vocab_size=10,
            )
        )
        cls.llama_config, cls.llama_tokenizer, cls.llama = create_llama(
            config=LlamaConfig(
                bos_token_id=1,
                eos_token_id=2,
                hidden_size=64,
                intermediate_size=128,
                max_position_embeddings=128,
                num_attention_heads=4,
                num_hidden_layers=2,
                num_key_value_heads=4,
                rms_norm_eps=1e-5,
                vocab_size=100,
            )
        )
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.gpt2 = cls.gpt2.to(cls.device)
        cls.llama = cls.llama.to(cls.device)

    def test_embed_to_distrib_gpt2_logits(self):
        batch, seq, _ = 2, 5, self.gpt2.config.n_embd
        embed = torch.randn(batch, seq, self.gpt2.config.n_embd).to(self.device)
        out = embed_to_distrib(self.gpt2, embed, logits=True)
        self.assertEqual(out.shape, (batch, seq, self.gpt2.config.vocab_size))

    def test_embed_to_distrib_gpt2_softmax(self):
        batch, seq, _ = 2, 5, self.gpt2.config.n_embd
        embed = torch.randn(batch, seq, self.gpt2.config.n_embd).to(self.device)
        out = embed_to_distrib(self.gpt2, embed, log=False, logits=False)
        self.assertEqual(out.shape, (batch, seq, self.gpt2.config.vocab_size))
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(batch, seq).to(self.device)))

    def test_embed_to_distrib_llama_logits(self):
        batch, seq, _ = 2, 5, self.llama.config.hidden_size
        embed = torch.randn(batch, seq, self.llama.config.hidden_size).to(self.device)
        out = embed_to_distrib(self.llama, embed, logits=True)
        self.assertEqual(out.shape, (batch, seq, self.llama.config.vocab_size))

    def test_embed_to_distrib_llama_softmax(self):
        batch, seq, _ = 2, 5, self.llama.config.hidden_size
        embed = torch.randn(batch, seq, self.llama.config.hidden_size).to(self.device)
        out = embed_to_distrib(self.llama, embed, log=False, logits=False)
        self.assertEqual(out.shape, (batch, seq, self.llama.config.vocab_size))
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(batch, seq).to(self.device)))


def suite():
    s = unittest.TestSuite()
    s.addTest(BasicUtilsTestCase("test_embed_to_distrib_gpt2_logits"))
    s.addTest(BasicUtilsTestCase("test_embed_to_distrib_gpt2_softmax"))
    s.addTest(BasicUtilsTestCase("test_embed_to_distrib_llama_logits"))
    s.addTest(BasicUtilsTestCase("test_embed_to_distrib_llama_softmax"))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
