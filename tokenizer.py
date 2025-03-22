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
            
    def merge(self, input, pair, mint):
        result = []
        for element in input:
            #print(f"the element: {element}, is being merged with pair {pair}")
            if isinstance(element, int):
                element = [element]

            i = 0
            while len(element) >= 2 and i < (len(element) - 1):
                if (element[i], element[i+1]) == pair:
                    element[i] = mint
                    element.pop(i + 1)
                    
                i += 1
            #print(f"was merged as {element}")
            result.append(element)

        return result
        
    def pair_frequency(self, input):
        result = {}
        for element in input:
            if isinstance(element, int):
                element = [element]

            if len(element) < 2:
                continue

            #print(f"trying to find a pair in {element}")
            for i in range(len(element) - 1):
                if (element[i], element[i+1]) in result:
                    result[(element[i], element[i+1])] += 1
                    continue
                result[(element[i], element[i+1])] = 1

        if len(result) == 0:
            return None

        #print(f"found pairs are: {result}")
        #print(f"the biggest one is: {max(result, key=result.get)}")
        return max(result, key=result.get)  
            
    def flatten(self, list):
        return [item for sublist in list for item in sublist]

    def train(self, text, total_vocab_size):
        text = "".join([str(x) if not isinstance(x, str) else x for x in text])
        #print(text)
        text = self.split(text)
        #print(text)
        conversion = []
        for word in text:
            decoded_line = []
            for letter in word:
                byte_letter = ord(letter)
                if(byte_letter > total_vocab_size):
                    continue
                decoded_line.append(byte_letter)
            conversion.append(decoded_line)
        #print(conversion)

        num_merges = total_vocab_size - 256

        if(num_merges <= 0):
            #print("no merges needed")
            return self.flatten(conversion)

        merges = {}
        for i in range(num_merges):
            #print(f"the current merges: {merges} for the {i} total {num_merges} merges")
            pair = self.pair_frequency(conversion)
            #print(f"the pair chosen {pair}")
            if pair == None:
                print("no merges possible")
                self.save_merges()
                self.merges = merges
                return conversion
            new_id = 256 + i
            #print(f"total: {conversion}, pair: {pair}, new_id: {new_id}")
            conversion = self.merge(conversion, pair, new_id)
            merges[new_id] = pair

        self.merges = merges
        #print(set(self.merges))
        self.save_merges()
        return self.flatten(conversion)

    def encode(self, text, vocab_size, load=False):
        if load:
            self.load_merges()
        
        tokens = self.split(text)
        encoded = []
        
        for token in tokens:
            token_ids = [ord(char) if ord(char) < vocab_size else 0 for char in token]
            encoded.append(token_ids)
        
        for merge_id, (left, right) in self.merges.items():
            for i in range(len(encoded)):
                token = encoded[i]
                j = 0
                while j < len(token) - 1:
                    if (token[j], token[j + 1]) == (left, right):
                        token[j] = merge_id
                        token.pop(j + 1)
                    else:
                        j += 1
        
        return [item for sublist in encoded for item in sublist]

    def encoded_breakdown(self, id):
        result = []
        if isinstance(id, list):
            stack = id.copy()  
        else:
            stack = [int(id)]

        #print(f"the element {stack} is being decoded")

        while stack:
            current_id = int(stack.pop())

            if (current_id < 255) and (current_id not in self.merges):
                result.append(current_id)
            else:
                left, right = self.merges[current_id]
                
                stack.append(right)
                stack.append(left)
        #print(f"it was decoded as {result}")
        return result

    def decode(self, input, load=False):
        #print(f"the merges are: {self.merges}")
        if load:
            self.load_merges()

        print(self.merges)
        decoded_text = []
        for token in input:
            decoded_text.extend([chr(x) for x in  self.encoded_breakdown(token) ])
            #print(f"the full string is {decoded_text}")
        return ''.join(decoded_text)

def main():
    # Sample text for training and testing
    text = [
        "Hello, how are you today?",
        "I am learning how to use a tokenizer.",
        "This tokenizer is based on BPE (Byte Pair Encoding)."
    ]
    
    tokenizer = Tokenizer()
    
    # Train tokenizer with the sample text, and define a vocab size
    total_vocab_size = 500  # This is an arbitrary size for testing
    tokenizer.train(text, total_vocab_size)
    
    # Save merges after training
    tokenizer.save_merges()

    # Test encoding and decoding
    encoded_text = tokenizer.encode("Hello, how are you?", vocab_size=500, load=True)
    print(f"Encoded text: {encoded_text}")
    
    decoded_text = tokenizer.decode(encoded_text, load=True)
    print(f"Decoded text: {decoded_text}")

if __name__ == "__main__":
    main()