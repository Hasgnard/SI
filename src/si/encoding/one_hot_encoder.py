
import numpy as np



class OneHotEncoder:
    """
    One-hot encoder for categorical features.
    """

    def __init__(self, padder: str, max_length: int = None):
        """
        Initializes the encoder.

        Parameters
        ----------
        padder: str
            The string to use for padding.
        max_length: int
            The maximum length of the categorical features.
        
        
        Attributes
        ----------
        alphabet: set
            The unique characters in the sequences
        
        char_to_index: dict
            Dictionary mapping characters in the alphabet to unique integers.
        
        index_to_char: dict
            Reverse dictionary mapping integers to characters in the alphabet.
        """
        self.padder = padder
        self.max_length = max_length

        self.alphabet = set()
        self.index_to_char = {}
        self.char_to_index = {}


    def fit(self, data:list[str]):
        """
        Fits the encoder to the data.

        Parameters
        ----------
        categorical_features: numpy.ndarray
            The categorical features to encode.
        """
        
        if self.max_length is None:
            self.max_length = max([len(seq) for seq in data])

        # get the unique characters in data
        seq = ''.join(data)
        self.alphabet = set(np.unique(list(seq)))

        for i, char in enumerate(self.alphabet):
            self.char_to_index[char] = i
            self.index_to_char[i] = char
        

        # add the padder to the alphabet if it is not already in there
        if self.padder not in self.alphabet:
            self.alphabet.add(self.padder)
            # get the next available index even if the dictionary is empty
            max_index = max(self.char_to_index.values(), default=-1)
            new_index = max_index + 1
            # update the dictionaries
            self.char_to_index[self.padder] = new_index
            self.index_to_char[new_index] = self.padder

        return self
        

    def transform(self, data:list[str]) -> np.ndarray:
        """
        Transforms the given data into one-hot encoded features.

        Parameters
        ----------
        data: numpy.ndarray
            The categorical features to encode.

        Returns
        -------
        numpy.ndarray
            The one-hot encoded features.
        """

        # Trim the sequences to the maximum length
        data = [seq[:self.max_length] for seq in data]
       
        # Pad the sequences with the padding character
        data = [seq.ljust(self.max_length, self.padder) for seq in data]

        # Encode the data to one-hot matrices
        encoded_sequences = np.zeros((len(data), self.max_length, len(self.alphabet)))
        
        for i, seq in enumerate(data):
            for j, char in enumerate(seq):
                encoded_sequences[i, j, self.char_to_index[char]] = 1

       
        return np.array(encoded_sequences)

    

    def fit_transform(self, data:list[str]) -> np.ndarray:
        """
        Fits the encoder to the data and transforms them into one-hot encoded features.

        Parameters
        ----------
        categorical_features: numpy.ndarray
            The categorical features to encode.

        Returns
        -------
        numpy.ndarray
            The one-hot encoded features.
        """

        self.fit(data)
        return self.transform(data)
    

    def inverse_transform(self, data:np.ndarray) -> np.ndarray:
        
        """
        Transforms the given one-hot encoded features back into a list of the original sequences.

        Parameters
        ----------
        data: numpy.ndarray
            The one-hot encoded features to decode.

        Returns
        -------
        numpy.ndarray
            The decoded categorical features.
        """
        # Decode the one-hot encoded for each sequence
        decoded_sequences = [
            # Join the characters corresponding to the maximum value indices in each one-hot vector
            "".join([self.index_to_char[np.argmax(one_hot_vector)] for one_hot_vector in sequence])
            # Remove trailing padding characters
            .rstrip(self.padder)
            # Iterate over the one-hot encoded sequences
            for sequence in data
        ]

        return np.array(decoded_sequences)

if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


    # example data
    data = np.array(['abc', 'def', 'ab', 'de'])

    # encoder
    custom_encoder = OneHotEncoder(padder='_')
    custom_encoded = custom_encoder.fit_transform(data)
    custom_decoded = custom_encoder.inverse_transform(custom_encoded)

    # comparing results
    print("Categories:\n", custom_encoder.alphabet)
    print("Custom Encoder Encoded:\n", custom_encoded)
    print("Custom Encoder Decoded:\n", custom_decoded)