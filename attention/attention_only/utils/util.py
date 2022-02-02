
import torch
import numpy as np
from torchtext.data.metrics import bleu_score


def get_padding_mask(query, key, pad_idx):
    batch_size, len_query = query.size()
    batch_size, len_key = key.size()
    masking = key.data.eq(pad_idx).unsqueeze(1)
    return masking.expand(batch_size, len_query, len_key)

def get_subsequent_mask(query, device):
    shape = [query.size(0), query.size(1), query.size(1)]
    subsequent_mask = np.tril(np.ones(shape), k=0) == 0
    subsequent_mask = torch.from_numpy(subsequent_mask).to(device)
    return subsequent_mask


# This method translate a sentence to target language
def translate_sentence(model, sentence, src_spacy_model, source_field, targer_field, device, max_length=60):
    
    if type(sentence) == str:
        input_tokens = [token.text.lower() for token in src_spacy_model(sentence)]
    else:
        input_tokens = [token.lower() for token in sentence]


    # Add <sos> and <eos>
    input_tokens.insert(0, source_field.init_token)
    input_tokens.append(source_field.eos_token)

    # List of indices
    source_text_to_indices = [source_field.vocab.stoi[token] for token in input_tokens]
    source_tensor = torch.LongTensor(source_text_to_indices).unsqueeze(0).to(device)
   
    outputs = [targer_field.vocab.stoi["<sos>"]]
    for i in range(max_length):
        target_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(source_tensor, target_tensor)

        best_guess = output[0].argmax(1)[-1].item()
        outputs.append(best_guess)

        if best_guess == targer_field.vocab.stoi["<eos>"]:
            break

    translated_sentence = [targer_field.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def bleu(data, model, spacy_german, german_field, english_field, device):

    targets = []
    outputs = []


    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
 
        predict = translate_sentence(model, src, spacy_german, german_field, english_field, device)
        predict = predict[:-1]

        targets.append([trg])
        outputs.append(predict)

    return bleu_score(outputs, targets)
