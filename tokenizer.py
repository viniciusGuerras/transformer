import json
import regex as re

SPECIAL_TOKENS = {
    '<sos>' : 100,
    '<eos>' : 100   
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
            
    #pop is costly make another list to store results
    def merge(self, input, pair, mint):
        result = []
        for element in input:
            if isinstance(element, int):
                element = [element]

            i = 0
            while len(element) >= 2 and i < (len(element) - 1):
                if (element[i], element[i+1]) == pair:
                    element[i] = mint
                    element.pop(i + 1)
                    
                i += 1
            result.append(element)

        return result
        
    #i probably can optimize the loop
    def pair_frequency(self, input):
        result = {}
        for element in input:
            if isinstance(element, int):
                element = [element]

            if len(element) < 2:
                continue

            for i in range(len(element) - 1):
                if (element[i], element[i+1]) in result:
                    result[(element[i], element[i+1])] += 1
                    continue
                result[(element[i], element[i+1])] = 1

        if len(result) == 0:
            return None

        return max(result, key=result.get)  
            
    def flatten(self, list):
        return [item for sublist in list for item in sublist]

    def encode(self, text, total_vocab_size):
        text = "".join([str(x) if not isinstance(x, str) else x for x in text])
        text = self.split(text)
        conversion = []

        for word in text:
            decoded_line = []
            for letter in word:
                byte_letter = ord(letter)
                if(byte_letter > total_vocab_size):
                    continue
                decoded_line.append(byte_letter)
            conversion.append(decoded_line)

        num_merges = total_vocab_size - 256

        if(num_merges <= 0):
            return self.flatten(conversion)

        merges = {}
        for i in range(num_merges):
            pair = self.pair_frequency(conversion)
            if pair != None:
                new_id = 256 + i
                conversion = self.merge(conversion, pair, new_id)
                merges[new_id] = pair

        self.merges = merges
        self.save_merges()
        return self.flatten(conversion)

    def encoded_breakdown(self, id):
        result = []
        if isinstance(id, list):
            stack = id.copy()  
        else:
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
            decoded_text.extend([chr(x) for x in  self.encoded_breakdown(token) ])
        return ''.join(decoded_text)