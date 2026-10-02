# Imports
import string
import re
import numpy as np
import json


# Tokenizer
class TokenizerBPE01:

    def __init__(self, initial_vocab:list=[]):
        # Initialize vocabulary
        self.vocab = {"<Space>":0, "<UNK>":1}
        self.vocab.update(dict(zip(initial_vocab, range(len(self.vocab), len(self.vocab) + len(initial_vocab)))))
        self.vocab.update(dict(zip(list(string.ascii_lowercase), range(len(self.vocab), len(self.vocab) + len(string.ascii_lowercase)))))
      #  self.vocab.update(dict(zip(list(string.punctuation), range(len(self.vocab), len(self.vocab) + len(string.punctuation)))))
        self.vocab.update(dict(zip(list([str(i) for i in range(10)]), range(len(self.vocab), len(self.vocab) + len([str(i) for i in range(10)])))))
        print(f"Initiate vocabulary of length: {len(self.vocab)}")
        self.max_token_length = 1

    # Tokenizer
    def tokenize(self, inputs:str):
        """ Transform string into tokens ids according to current vocabulary """
        tokenized_vector = []

        # Replace space with [mask]
        inputs = inputs.lower()
        inputs = list(inputs) # Keep [mask] as single element to be counted
        inputs = [e if e != ' ' else list(self.vocab.keys())[0] for e in inputs]

        # Map vector ids to inputs
        while len(inputs) > 0:
            tokenized = False
            for token_len in range(self.max_token_length, 0, -1):
                current_subword = ''.join(inputs[:token_len]) # rejoin subword from chars
                if current_subword in self.vocab.keys():
                    tokenized_vector.append(self.vocab[current_subword])
                    inputs = inputs[token_len:]
                    tokenized = True
                    break

            if not tokenized:
                tokenized_vector.append(1)
                inputs = inputs[1:]
        
        return np.asarray(tokenized_vector)

    def tokenToWords(self, token:np.array, subword=False, keep_mask=False):
        """
        Transform token back to string
        subword: True if keep subword separated
        keep_mask: True if keep mask as [mask], else replaced according to the [mask]
        """
        reversed_vocab = {v: k for k, v in self.vocab.items()}
        subwords = [reversed_vocab[t] for t in token]

        if not keep_mask:
            #### Use regex later, now just [Space]
            subwords = [sw if sw != list(self.vocab.keys())[0] else ' ' for sw in subwords]

        if not subword:
            subwords = ''.join(subwords)

        return subwords

    # Train
    def train(self, inputs, k, min_count=1):
        """ Train new vocabulary, add maximum k more subwords
            Only add if pair appear higher or equal to min_count
        """
        num_added = 0

        while num_added < k:
            # Get segments - currently word by word
            segments = self.__preprocess__(inputs)
            
            # Tokenize segments
            segments = [self.tokenize(se) for se in segments]
            most_freq_pair, counts = self.__get_most_freq_pair__(segments)

            # Validate pair added
            if counts >= min_count:
                most_freq_pair_to_word = self.tokenToWords(most_freq_pair, keep_mask=True)
                if len(most_freq_pair_to_word) > self.max_token_length: self.max_token_length = len(most_freq_pair_to_word)
                self.vocab[most_freq_pair_to_word] = len(self.vocab)
                num_added += 1

            else:
                print("Pair repeats not frequent enough.")
                break

        # Report
        print(f"Added: {num_added}/{k} into vocabulary.")
        print(f"Current vocabulary length: {len(self.vocab)}")

    # Update vocab mannually

    def update(self, add_vocab):
        origin_len = len(add_vocab)
        # Check exisited
        existed = set(self.vocab.keys()) & set(add_vocab)
        add_vocab = set(add_vocab) - set(self.vocab.keys())

        # Update vocabulary
        self.vocab.update(dict(zip(add_vocab, range(len(self.vocab), len(self.vocab) + len(add_vocab)))))

        # Report
        print(f"Added {len(add_vocab)}/{origin_len} into vocabulary.")
        print(f"Existed: {len(existed)}/{origin_len}.")
        print(f"Current vocabulary length: {len(self.vocab)}")

    # Save
    def saveToJSON(self, path):
        """ Save current vocabulary - Index: subwords"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

        print("Saved.")

    # Load
    def loadFromJSON(self, path):
        """ Load vocabulary """
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        print("Loaded.")

    # Utils
    def __words_segment__(self, inputs):
        """ Segment preprocessed inputs and add [mask]s 
        """
        # Simple words segmentation
        segments = set(inputs.split()) 

        # Add space before segments - #### maybe fixed later
        segments = [' ' + se for se in segments]

        #### Later using regular expression + More [mask]

        return segments


    #### phrase segment


    def __preprocess__(self, inputs:str):
        """ Combine all preprocess steps """
        inputs = inputs.lower() # Later handle capitalized letters
        segments = self.__words_segment__(inputs)
        #### Later handle puntuations, numbers...

        return segments

    def __create_pair__(self, segment:np.array):
        return np.vstack((np.roll(segment, 1)[1:], segment[1:])).transpose()

    def __get_most_freq_pair__(self, segments:np.array):
        """ Get the most frequently appear pair
         """
        paired_tokens = np.empty((0, 2))
        # Create pair in each segment
        for se in segments:
            paired_tokens = np.vstack((paired_tokens, self.__create_pair__(se)))

        # Get most frequent pair
        try:
            _, indices, counts = np.unique(paired_tokens, axis=0, return_index=True, return_counts=True)
            max_counts = np.max(counts)
            most_frequent_pair = paired_tokens[indices[np.argmax(counts)]]

            return most_frequent_pair, max_counts

        except:
            print("All words tokenized.")
            return None, -1
