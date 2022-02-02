
import torch
from torchtext.data.metrics import bleu_score

# This method translate a sentence to target language
def translate_sentence(model, sentence, src_spacy_model, source_field, targer_field, device, max_length=50):
    
    if type(sentence) == str:
        tokens = [token.text.lower() for token in src_spacy_model(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <sos> and <eos>
    tokens.insert(0, source_field.init_token)
    tokens.append(source_field.eos_token)

    # List of indices
    text_to_indices = [source_field.vocab.stoi[token] for token in tokens]

    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    # sentence_tensor: (seq_length, 1)

    # Build encoder hidden, cell state
    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor)

    outputs = [targer_field.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_states, hidden, cell)
            best_guess = output.argmax(1).item()

        if output.argmax(1).item() == targer_field.vocab.stoi["<eos>"]:
            break

        outputs.append(best_guess)

    translated_sentence = [targer_field.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]

import nltk

def bleu(data, model, src_spacy_model, german_field, english_field, device):

    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, src_spacy_model, german_field, english_field, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="data/checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
