import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


@memory.cache
def cosine_similarity_row_wise(a, b):
    result = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        result[i] = a[i].dot(b[i].T).toarray()[0][0]
    return result


def get_query_in_title(df):
    result = []
    for query, title in zip(df.search_term.values, df_all.product_title.values):
        query = ''.join([q for q in query if q.isalnum()])
        title = title.lower()
        title = ''.join([t for t in title.lower() if t.isalnum()])
        if query in title:
            #print(rel, query, title)
            result.append(float(len(query)) / float(len(title)))
        else:
            result.append(0)
    return np.array(result)


