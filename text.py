import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
logging.info('feature extraction text')


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


def get_distance_features(vectorizer, prefix, df, feat):
    all_text = df_all.search_term + '\t' + df_all.product_title + '\t' \
        + df_all.product_description + '\t' + df_all.attr_texts
    unigram_vectorizer.fit(all_text)

    search_terms = vectorizer.transform(df.search_term)
    feat['{}_all'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, all_text)

    title = vectorizer.transform(df.product_title)
    feat['{}_title'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, title)

    description = vectorizer.transform(df.product_description)
    feat['{}_description'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, description)

    attributes = vectorizer.transform(df.attr_texts)
    feat['{}_attributes'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, attributes)

    return feat


def get_golden_features(df, feat):
    feat['unigram_title_all_golden'] = \
        df_all.unigram_title - df_all.unigram_description
    feat['unigram_bigram_desc_desc_golden'] = \
        df_all.unigram_description - df_all.bigram_description
    return feat


@memory.cache
def get_jaccard_similarities(a, b):
    jscores = []
    for x, y in zip(a, b):
        jscores.append(jaccard_similarity_score(x, y))
    return np.array(jscores)


def get_jaccard_features(df, feat):
    vectorizer = CountVectorizer(binary=True, min_df=2)
    text = df_all.search_term + '\t' + df_all.product_title + '\t' + i\
        df_all.product_description + '\t' + df_all.attr_texts
    vectorizer.fit(text)

    search = vectorizer.transform(df_all.search_term)
    title = vectorizer.transform(df_all.product_title)
    description = vectorizer.transform(df_all.product_description)
    brand = vectorizer.transform(df_all.brand)
    attributes = vectorizer.transform(df_all.attr_texts)

    feat['jaccard_search_title'] = get_jaccard_similarities(search, title)
    feat['jaccard_search_desc'] = get_jaccard_similarities(search, description)
    feat['jaccard_search_attr'] = get_jaccard_similarities(search, attributes)
    return feat


def get_count_features(df, feat):
    feat['search_text_len_char'] = df.search_term.apply(len)
    feat['search_text_len_word'] = df.search_term.apply(lambda x: len(x.split()))
    feat['brand_text_len_char'] = df.brand.apply(len)
    feat['brand_text_len_word'] = df.brand.apply(lambda x: len(x.split()))
    feat['title_text_len_char'] = df.product_title.apply(len)
    feat['title_text_len_word'] = df.product_title.apply(lambda x: len(x.split()))
    feat['desc_text_len_char'] = df.product_description.apply(len)
    feat['desc_text_len_word'] = df.product_description.apply(lambda x: len(x.split()))
    feat['attr_text_len_char'] = df.attr_texts.apply(len)
    feat['attr_text_len_word'] = df.attr_texts.apply(lambda x: len(x.split()))

    feat['all_text_len_char'] = feat.search_text_len_char + \
        feat.title_text_len_char + feat.desc_text_len_char
    feat['all_text_len_word'] = feat.search_text_len_word + \
        feat.title_text_len_word + feat.desc_text_len_word

    feat['all_text_len_ratio'] = feat.all_text_len_word / feat.all_text_len_char
    feat['search_text_len_ratio'] = (feat.search_text_len_word / feat.search_text_len_char).replace()
    feat['brand_text_len_ratio'] = feat.brand_text_len_word / feat.brand_text_len_char
    feat['brand_text_len_ratio'] = feat.brand_text_len_ratio.fillna(value=0)
    feat['title_text_len_ratio'] = feat.title_text_len_word / feat.title_text_len_char
    feat['desc_text_len_ratio'] = feat.desc_text_len_word / df_all.desc_text_len_char
    feat['attr_text_len_ratio'] = feat.attr_text_len_word / df_all.attr_text_len_char
    feat['attr_text_len_ratio'] = feat.attr_text_len_ratio.fillna(value=0)
    return feat


def get_features(df):
    feat = pd.DataFrame(index=df.index)

    # FIXME proper binarization of word counts
    feat['two_word_query'] = (df.len_of_query == 2).astype(int)
    feat['query_in_title'] = get_query_in_title(df)
    
    all_text = df_all.search_term + '\t' + df_all.product_title + '\t' \
        + df_all.product_description + '\t'

    logging.info('get count features')
    feat = get_count_features(df, feat)

    logging.info('get unigram cosine distances')
    unigram_vectorizer = TfidfVectorizer(
        min_df=3, max_df=0.75, stop_words='english', strip_accents='unicode',
        use_idf=1, smooth_idf=1, sublinear_tf=1,
        token_pattern= r'(?u)\b\w\w+\b')
    feat = get_distance_features(unigram_vectorizer, 'unigram', df, feat)

    logging.info('get bigram cosine distances')
    bigram_vectorizer = TfidfVectorizer(
        min_df=3, max_df=0.75, stop_words='english', strip_accents='unicode',
        use_idf=1, smooth_idf=1, sublinear_tf=1,
        token_pattern= r'(?u)\b\w\w+\b', ngram_range(1, 2))
    feat = get_distance_features(bigram_vectorizer, 'bigram', df, feat)

    logging.info('get ngram cosine distances')
    ngram_vectorizer = TfidfVectorizer(
        min_df=3, max_df=0.75, strip_accents='unicode',
        use_idf=1, smooth_idf=1, sublinear_tf=1, analyzer='char_wb',
        ngram_range=(2, 5))
    feat = get_distance_features(ngram_vectorizer, 'ngram', df, feat)

    logging.info('get golden features')
    feat = get_golden_features(df, feat)

    logging.info('get jaccard similarity features')
    feat = get_jaccard_features(df, feat)


if __name__ == "__main__":
    from utils import get_data
    owl_features = get_features(get_data())
    owl_features.to_csv('/tmp/owl.csv')
