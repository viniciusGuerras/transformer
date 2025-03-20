import json

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
        i = 0
        while i < len(input) - 1:
            if input[i] == pair[0] and input[i+1] == pair[1]:
                result.append(mint)
                i += 2  
            else:
                result.append(input[i])
                i += 1

        if i == len(input) - 1:
            result.append(input[-1])

        return result
        
    def pair_frequency(self, input):
        if len(input) < 2:
            return None
        result = {}
        for i in range(len(input) - 1):
            if (input[i], input[i+1]) in result:
                result[(input[i], input[i+1])] += 1
                continue
            result[(input[i], input[i+1])] = 1

        return max(result, key=result.get)  
            
    def encode(self, text, total_vocab_size):
        conversion = list(text.encode('utf-8', errors="ignore"))
        num_merges = total_vocab_size - 256
        if(num_merges <= 0):
            print("no merges needed")
            return conversion
        merges = {}
        for i in range(num_merges):
            pair = self.pair_frequency(conversion)
            new_id = 256 + i
            conversion = self.merge(conversion, pair, new_id)
            merges[new_id] = pair

        self.merges = merges
        return conversion

    def recursive_breakdown(self, id, result, visited=None):
        if visited is None:
            visited = set()
        if id in visited:
            raise ValueError(f"Cycle detected at token {id}")
        visited.add(id)
        
        if id <= 255 or id not in self.merges:
            result.append(id)
            return
        
        left, right = self.merges[id]
        self.recursive_breakdown(left, result, visited.copy())
        self.recursive_breakdown(right, result, visited.copy())


    def decode(self, input):
        decoded_text = []
        for token in input:
            breakdown = []
            self.recursive_breakdown(int(token), breakdown)
            decoded_text.extend([chr(x) for x in breakdown])
        return ''.join(decoded_text)

