import warnings

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BillingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace 

from tqdm import tqdm

from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for the target (decoder_input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate the output of the decoder
        decoder_output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # Get the next token
        prob = model.project(decoder_output[:, -1])

        # select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx: 
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    # size of the control window (just use a default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

    # if writer:


def get_all_sentences(ds, lang):
    # goes through every row and yields the sentence for the given language
    for item in ds:
        yield item[lang]



def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        # how to initially split the text 
        tokenizer.pre_tokenizer = Whitespace()
        # min_frequency means for a word to appear in our vocabulary it must apear at least 2 times in the training data to be learned by the bpe vocabulary
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # train the tokenizer (the tokenizer reads sentences one by one and learns its vocabulary)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer



def get_ds(config):
    ds_raw = load_dataset("IbrahimAmin/arz-en-parallel-corpus")

    train_ds_raw = ds_raw["train"]
    val_ds_raw = ds_raw["test"]

    # build tokenizers from train split 
    tokenizer_src =  get_or_build_tokenizer(config=config, ds=train_ds_raw, lang=config["lang_src"])
    tokenizer_tgt =  get_or_build_tokenizer(config=config, ds=train_ds_raw, lang=config["lang_tgt"])

    def fits(item): 
        src_len = len(tokenizer_src.encode(item[config['lang_src']]).ids)
        tgt_len = len(tokenizer_tgt.encode(item[config['lang_tgt']]).ids)

        return src_len <= config['seq_len'] - 2 and tgt_len <= config['seq_len'] - 1

    before_train = len(train_ds_raw)
    train_ds_raw = train_ds_raw.filter(fits)
    print(f"Filtered train: kept {len(train_ds_raw)} of {before_train} rows ({before_train - len(train_ds_raw)} removed as too long)")

    before_val = len(val_ds_raw)
    val_ds_raw = val_ds_raw.filter(fits)
    print(f"Filtered val: kept {len(val_ds_raw)} of {before_val} rows ({before_val - len(val_ds_raw)} removed as too long)")


    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"]) 

    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max length of source sentence: {max_len_src}")
    print(f"max length of target sentence: {max_len_tgt}")

    # groups multiple samples into batches
    # shuffle=True, means randomly change the order of the dataset samples each time you iterate through the DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # here the order doesn't matter we don't train the data
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False) # process each sentence one by one

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    # define the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # label_smoothing means from highest probability take 0.1 percent of score and give it to the others
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                model_filename = get_weights_file_path(config, f'{epoch:02d}_step{global_step}')
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                }, model_filename)

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            "model_state": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
