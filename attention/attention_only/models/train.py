import torch
import time
import math
from utils import translate_sentence, epoch_time


class Process():
    def __init__(self, model, src_spacy_model, source_field, target_field, optimizer, scheduler, loss_func, test_sentence, clip, device):
        self.model = model
        self.src_spacy_model = src_spacy_model
        self.source_field = source_field
        self.target_field = target_field
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.test_sentence = test_sentence
        self.clip = clip
        self.device = device
        self.step = 0

    # Train model
    def train(self, iterator):

        epoch_loss = 0

        self.model.train()

        for batch in iterator:
            source = batch.src.to(self.device)
            target = batch.trg.to(self.device)
            # source: (batch_size, seq_length)
            # target: (batch_size, seq_length)

            # Last target token is <eos>. Do no need to pass it.
            output = self.model(source, target[:,:-1])
            # output: (batch_size, seq_length, target_vocab_size)
 
            target_vocab_size = output.shape[-1]
            output = output.contiguous().view(-1, target_vocab_size)
            # output: (batch_size * seq_length, target_vocab_size)

            target = target[:,1:].contiguous().view(-1)
            # target: (batch_size, seq_length)

            self.optimizer.zero_grad()
            loss = self.loss_func(output, target)
            loss.backward()

            # This is used to prevent gradient exploding. Clipping to 1.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            # This updates the trainable parameters
            self.optimizer.step()

            self.step += 1

            epoch_loss += loss.item()
            
            
        return epoch_loss / len(iterator)

    def evaluate(self, iterator):

        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for batch in iterator:

                source = batch.src.to(self.device)
                target = batch.trg.to(self.device)

                output = self.model(source, target[:,:-1])
                
                target_vocab_size = output.shape[-1]

                output = output.contiguous().view(-1, target_vocab_size)
                target = target[:,1:].contiguous().view(-1)

                loss = self.loss_func(output, target)

                epoch_loss += loss.item()


        return epoch_loss / len(iterator)

    def run(self, num_epochs, train_iterator, valid_iterator):

        best_lost = 1e10
        no_better = 0

        for epoch in range(num_epochs):

            start_time = time.time()
            
            train_loss = self.train(train_iterator)
            valid_loss = self.evaluate(valid_iterator)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            lr = self.scheduler.get_last_lr()[0]

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Exp: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Exp: {math.exp(valid_loss):7.3f}')
            print(f'\t Learning Rate: {lr:.7f}')

            # This translate the sample sentence.
            translated_sentence = translate_sentence(
                self.model, self.test_sentence,
                self.src_spacy_model, self.source_field, self.target_field,
                self.device, max_length=50,
            )

            print(f"Translated example sentence: \n {' '.join(translated_sentence[:-1])}")
            print()
            self.scheduler.step()

            if valid_loss > best_lost:
              no_better += 1
              if no_better == 5:
                break
            else: 
              best_lost = valid_loss
              no_better=0
