import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BillingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.seq_len = seq_len
        self.lang_tgt = lang_tgt

        # we want to conver the special tokens into the input id
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair[self.lang_src]
        tgt_text = src_target_pair[self.lang_tgt]

        # this will give us the numbers corresponding to each word in the original sentence
        # encode -> converts text into tokens
        # ids -> converts tokens inot ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # how many pads we need to add to let the sentence reachs the fixed seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 2 for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # During the training, we add only SOS to the decoder input

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # we create two sequences decoder_input and label that are shifted by one position 
        # decoder -> label
        # [SOS] -> input[0]
        # input[0] -> input[1]
        #....................
        # input[n] -> [EOS]
        # the label represents the next word we should predict, it tell the loss function how close we are to the correct prediction
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # what we expect as output from the decoder
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )


        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # .unsqueeze(0) to add batch and heads latter (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len), # (batch_size, heads, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }



def causal_mask(size):
    # give me every value that is above the diagonal 
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    # everything that is 0 will become true and everything that is not 0 will become false
    return mask == 0