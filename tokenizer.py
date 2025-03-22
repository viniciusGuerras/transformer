import json
import regex as re

SPECIAL_TOKENS = {
    '<sos>' : 100,
    '<eos>' : 100   
}

#split pattern from chatgpt (got from paper of 4)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def tokenize(text):
    return re.findall(SPLIT_PATTERN, text)

class Tokenizer():
    def __init__(self):
        self.merges = {}

    def save_merges(self, filename="tokenizer_weights.json"):
        with open(filename, 'w') as f:
            json.dump(self.merges, f)

    def load_merges(self, filename="tokenizer_weights.json"):
        try:
            with open(filename, 'r') as f:
                self.merges = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found. Please train the tokenizer first.")
            
    def merge(self, input, pair, mint):
        result = []
        for element in input:
            if isinstance(element, int):
                element = [element]

            merged_token = []
            i = 0  
            while i < len(element):
                if i < len(element) - 1 and element[i] == pair[0] and element[i+1] == pair[1]:
                    merged_token.append(mint)
                    i += 2  
                else:
                    merged_token.append(element[i])
                    i += 1
            result.append(merged_token)
            
        return result
        
    def pair_frequency(self, input):
        if len(input) < 2:
            return None
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

        return max(result, key=result.get)  
            
    def flatten(self, list):
        flat = [item for sublist in list for item in sublist]
        return flat

    def encode(self, text, total_vocab_size):
        text = [str(x) if not isinstance(x, str) else x for x in text]
        text = tokenize(" ".join(text))
        conversion = []
        for word in text:
            decoded_line = []
            for letter in word:
                decoded_line.append(ord(letter))
            conversion.append(decoded_line)

        num_merges = total_vocab_size - 256
        if(num_merges <= 0):
            print("no merges needed")
            return self.flatten(conversion)
        merges = {}
        for i in range(num_merges):
            pair = self.pair_frequency(conversion)
            if pair == None:
                print("no merges possible")
                return conversion
            new_id = 256 + i
            conversion = self.merge(conversion, pair, new_id)
            merges[new_id] = pair

        self.merges = merges
        print(set(self.merges))
        self.save_merges()
        return self.flatten(conversion)

    def recursive_breakdown(self, id, result):
        if id <= 255 or id not in self.merges:
            result.append(id)
            return
        
        left, right = self.merges[id]
        self.recursive_breakdown(left, result) 
        self.recursive_breakdown(right, result)


    def decode(self, input):
        decoded_text = []
        for token in input:
            breakdown = []
            self.recursive_breakdown(token, breakdown)
            decoded_text.extend([chr(x) for x in breakdown])
        return ''.join(decoded_text)

