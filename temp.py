dictv = {}
import pickle

with open('affected_indices.pkl', 'rb') as handle:
    b = pickle.load(handle)
print(b)
