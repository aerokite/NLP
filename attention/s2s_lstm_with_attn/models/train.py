import torch
import time
import math
from utils import *

class Process():
    def __init__(self, model, src_spacy_model, source_field, target_field, optimizer, loss_func, test_sentence, writer, clip, device):
        self.model = model
        self.src_spacy_model = src_spacy_model
        self.source_field = source_field
        self.target_field = target_field
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.test_sentence = test_sentence
        self.clip = clip
        self.device = device
        self.writer = writer
        self.step = 0

    # Train model
    def train(self, iterator):

        epoch_loss = 0

        self.model.train()

        for batch in iterator:

            source = batch.src.to(self.device)
            target = batch.trg.to(self.device)

            output = self.model(source, target)
            output_size = output.shape[-1]

            output = output[1:].reshape(-1, output_size)
            target = target[1:].reshape(-1)

            self.optimizer.zero_grad()
            loss = self.loss_func(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            # self.writer.add_scalar("Training loss", loss, global_step=self.step)
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

                output = self.model(source, target)
                output_size = output.shape[-1]

                output = output[1:].reshape(-1, output_size)
                target = target[1:].reshape(-1)

                loss = self.loss_func(output, target)
                epoch_loss += loss.item()


        return epoch_loss / len(iterator)

    def run(self, num_epochs, train_iterator, valid_iterator):
        best_valid_loss = 1e10
        for epoch in range(num_epochs):

            start_time = time.time()
            
            train_loss = self.train(train_iterator)
            valid_loss = self.evaluate(valid_iterator)

            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # checkpoint = {
                #     "state_dict": self.model.state_dict(),
                #     "optimizer": self.optimizer.state_dict(),
                #     }
                # save_checkpoint(checkpoint)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

            translated_sentence = translate_sentence(
                self.model, self.test_sentence,
                self.src_spacy_model, self.source_field, self.target_field,
                self.device, max_length=50,
            )

            print(f"Translated example sentence: \n {translated_sentence}")

