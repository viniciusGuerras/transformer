class Tokenizer():

    def pair_frequency(self, input):
        result = {}
        for i in range(len(input) - 1):
            if result.get((input[i], input[i+1])) == True:
                result[(input[i], input[i+1])] += 1
                continue
            result[(input[i], input[i+1])] = 1
        return #the biggest one
            
        print(result)
    def encode(self, text):
        conversion = [ord(letter) for letter in text]
        vocabulary = sorted(list(set(conversion)))
        vocabulary_size = len(vocabulary)
        print(self.pair_frequency(conversion))

        return conversion

        


    def decode(self):
        pass



def main():
    test_string = "hey, how are you doing pal? everything all right?, hey, how are your balls?"
    toke = Tokenizer()
    print(toke.encode(test_string))



main()