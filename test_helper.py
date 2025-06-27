'''import pickle
from helper import query_point_creator
# from: import distance
import Levenshtein as distance


# Load saved vectorizer and SVD
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
svd = pickle.load(open("svd.pkl", "rb"))

# Sample input questions
q1 = "Where is the capital of India?"
q2 = "Which city serves as the capital of India?"

# Run query point creator
features = query_point_creator(q1, q2, vectorizer, svd)

print("✅ Feature vector shape:", features.shape)'''
