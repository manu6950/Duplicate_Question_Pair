import numpy as np
import distance
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import nltk
nltk.download('punkt')  # or other corpora

nltk.download('stopwords')


stop_words = set(stopwords.words("english"))

def preprocess(q):
    return str(q).lower().strip()

def test_common_words(q1, q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    return len(w1 & w2)

def test_total_words(q1, q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    return len(w1) + len(w2)

def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    q1_words = set([w for w in q1_tokens if w not in stop_words])
    q2_words = set([w for w in q2_tokens if w not in stop_words])
    q1_stops = set([w for w in q1_tokens if w in stop_words])
    q2_stops = set([w for w in q2_tokens if w in stop_words])
    features = [0.0]*8
    features[0] = len(q1_words & q2_words) / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    features[1] = len(q1_words & q2_words) / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    features[2] = len(q1_stops & q2_stops) / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    features[3] = len(q1_stops & q2_stops) / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    features[4] = len(set(q1_tokens) & set(q2_tokens)) / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    features[5] = len(set(q1_tokens) & set(q2_tokens)) / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    features[7] = int(q1_tokens[0] == q2_tokens[0])
    return features

def test_fetch_length_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    features = [0.0]*3
    features[0] = abs(len(q1_tokens) - len(q2_tokens))
    features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    lcs = list(distance.lcsubstrings(q1, q2))
    features[2] = len(lcs[0]) / (min(len(q1), len(q2)) + 1)
    return features

def test_fetch_fuzzy_features(q1, q2):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]
def query_point_creator(q1, q2, vectorizer, svd):
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # --- Handcrafted 19 features (same as training) ---
    features = []

    features.append(len(q1))                             # q1_len
    features.append(len(q2))                             # q2_len
    features.append(len(q1.split()))                     # q1_num_words
    features.append(len(q2.split()))                     # q2_num_words

    features.extend(test_fetch_token_features(q1, q2))   # 8 features
    features.extend(test_fetch_length_features(q1, q2))  # 3 features
    features.extend(test_fetch_fuzzy_features(q1, q2))   # 4 features

    handcrafted = np.array(features).reshape(1, -1)       # (1, 19)

    # --- BOW + SVD ---
    combined = q1 + " " + q2
    bow_vector = vectorizer.transform([combined])
    svd_vector = svd.transform(bow_vector)                # (1, 97)

    # Combine
    return np.hstack((svd_vector, handcrafted))           # Final shape: (1, 116)
