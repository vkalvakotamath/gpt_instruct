from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Optional

from datasets import load_dataset
import torch

class InstructGPT:
    def __init__(
        self,
        model_name: str = 'gpt2-medium',
        max_length: int = 420
    ):
        self.model_name = model_name
        self.max_length = max_length


        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id



    def prepare_dataset(self):
        # Change this to a better dataset because this one sucks
        dataset = load_dataset("Dahoas/instruct-synthetic-prompt-responses")

        # Combine prompts
        def combine_prompt_response(example):
            full_text = f"{example['prompt']} {self.tokenizer.eos_token} {example['response']}"
            return {'text': full_text}
        dataset = dataset.map(combine_prompt_response, remove_columns=dataset['train'].column_names)
        return dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,

            max_length=self.max_length,
            padding='max_length'
        )

    def prepare_training_data(self, test_size: float = 0.1):

        tokenized_datasets = self.dataset.map(
            self.tokenize_function,
            remove_columns=self.dataset['train'].column_names,

            batched=True,
        )

        # Split into train and validation
        tokenized_datasets = tokenized_datasets['train'].train_test_split(test_size=test_size)

        return tokenized_datasets

    def train(
        self,
        output_dir: str = './instruct-gpt2-model',
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        num_train_epochs: int = 1, # not really sure about this stuff yet, may want to tweak this later
    ):
        
        weight_decay=0.01

        self.dataset = self.prepare_dataset() # Call prepare_dataset to initialize self.dataset
        tokenized_datasets = self.prepare_training_data()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        # Training arguments
        training_args = TrainingArguments(
            fp16=True,
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_steps=100,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )


        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            data_collator=data_collator,
        )


        trainer.train()
        trainer.save_model()

        # Stuff from Claude because I apparently messed up a bunch of things
    def generate_text(
        self,
        prompt: str,
        max_length: int = 420,
        temperature: float = 0.5,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using trained model

        Args:
            prompt (str): Input prompt
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling threshold

        Returns:
            Generated text
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=2,
            do_sample=True
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)



def main():
    instruct_trainer = InstructGPT()
    instruct_trainer.train(
        output_dir='./instruct-gpt2-model',
        learning_rate=5e-5,
        batch_size=8,
        num_train_epochs=5
    )

    prompt = "Tell me a story about Poland"
    generated_text = instruct_trainer.generate_text(prompt)
    print("Response::", generated_text)

if __name__ == "__main__":
    main()
