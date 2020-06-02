import pickle
import numpy as np
from tensorflow.keras.models import load_model

with open("data/chars.txt", "rb") as fp:
    chars = pickle.load(fp)
    
n_vocab = len(chars)
int_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_int = dict((c, i) for i, c in enumerate(chars))

model = load_model('models/model.h5')

seed = 'make america great again'
seed = seed.lower()
pattern = [char_to_int[char] for char in seed]
result = ""

for i in range(140):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    # https://stackoverflow.com/questions/47125723/keras-lstm-for-text-generation-keeps-repeating-a-line-or-a-sequence
    #index = np.random.choice(len(prediction[0]), p=prediction[0])
    result += int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print(''.join(result))