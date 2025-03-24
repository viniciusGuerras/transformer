import json
import regex as re
import pandas as pd
import numpy as np
from collections import Counter

#Later i should rewrite this in C or Rust
#i could batch process the tokenization process

SPECIAL_TOKENS = {
    '<sos>' : 256,
    '<eos>' : 257   
}

#split pattern from chatgpt (got from paper of 4)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer():

    def __init__(self):
        self.merges = {}

    def split(self, text):
        return re.findall(SPLIT_PATTERN, text)

    def save_merges(self, filename="checkpoints/tokenizer/tokenizer_weights.json"):
        with open(filename, 'w') as f:
            json.dump(self.merges, f)

    def load_merges(self, filename="checkpoints/tokenizer/tokenizer_weights.json"):
        try:
            with open(filename, 'r') as f:
                loaded_merges = json.load(f)
                self.merges = {int(key): value for key, value in loaded_merges.items()}
        except FileNotFoundError:
            print(f"File {filename} not found. Please train the tokenizer first.")
            
    def merge(self, input, pair, mint):
        result = []
        for element in input:
            if len(element) < 2:
                result.append(element)
                continue
            
            merged = []
            i = 0
            
            while i < len(element) - 1:
                if (element[i], element[i+1]) == pair:
                    merged.append(mint)
                    i += 2
                else:
                    merged.append(element[i])
                    i += 1
            
            if i == len(element) - 1:
                merged.append(element[i])
            
            result.append(merged)
        
        return result
        
    def pair_frequency(self, input):
        pairs = (pair for element in input if len(element) > 1 for pair in zip(element, element[1:]))
        result = Counter(pairs)
        return max(result, key=result.get) if result else None

    def encode(self, text, total_vocab_size):
        text = [['<sos>'] + self.split(phrase) + ['<eos>'] for phrase in text]
        text = [item for sublist in text for item in sublist]

        conversion = []
        for word in text:
            decoded_line = []
            print(word)
            if word in SPECIAL_TOKENS:
                decoded_line.append(SPECIAL_TOKENS.get(word))
            else:
                for letter in word:
                    byte_letter = ord(letter)
                    if(byte_letter > total_vocab_size):
                        continue
                    decoded_line.append(byte_letter)
            conversion.append(decoded_line)
        
        merges = {}
        max_merges= total_vocab_size - 256
        #range based on ASCII size
        for i in range(max_merges):
            pair = self.pair_frequency(conversion)
            if pair == None:
                break
            new_id = 256 + len(SPECIAL_TOKENS) + i
            conversion = self.merge(conversion, pair, new_id)
            merges[new_id] = pair

        self.merges = merges
        self.save_merges()
        return [item for sublist in conversion for item in sublist]

    def encoded_breakdown(self, id):
        result = []
        stack = [int(id)]

        while stack:
            current_id = int(stack.pop())

            if (current_id <= 255) and (current_id not in self.merges):
                result.append(current_id)
            else:
                left, right = self.merges[current_id]
                
                stack.append(right)
                stack.append(left)
        return result

    def decode(self, input, load=False):
        if load:
            self.load_merges()

        decoded_text = []
        for token in input:

            if token in SPECIAL_TOKENS.values():
                decoded_text.append([key for key, value in SPECIAL_TOKENS.items() if value == token][0])
            else:
                decoded_text.extend([chr(x) for x in  self.encoded_breakdown(token) ])
        return ''.join(decoded_text)
