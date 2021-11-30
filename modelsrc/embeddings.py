import numpy as np
def embed_matrix(nlp, tokens):
    result = []

    for word in tokens:
        result.append(np.array(nlp(word)[0].vector))

    return np.array(result)
