import os
import nltk
import logging
import numpy as np
import pandas as pd

from difflib import SequenceMatcher as seq_matcher

from sklearn.metrics import jaccard_similarity_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

from joblib import Memory
memory = Memory(cachedir='/tmp/joblib')

stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)
english_stemmer = nltk.stem.PorterStemmer()


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


@memory.cache
def cosine_similarity_row_wise(a, b):
    result = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        result[i] = a[i].dot(b[i].T).toarray()[0][0]
    return result


def get_query_in_title(df):
    result = []
    for query, title in zip(df.search_term.values, df.product_title.values):
        query = ''.join([q for q in query if q.isalnum()])
        title = title.lower()
        title = ''.join([t for t in title.lower() if t.isalnum()])
        if query in title:
            #print(rel, query, title)
            result.append(float(len(query)) / float(len(title)))
        else:
            result.append(0)
    return np.array(result)


def get_distance_features(vectorizer, prefix, df):
    feat = pd.DataFrame(index=df.index)
    all_text = df.search_term + '\t' + df.product_title + '\t' \
        + df.product_description + '\t' + df.attr_texts
    vectorizer.fit(all_text)

    all_text = vectorizer.transform(all_text)
    search_terms = vectorizer.transform(df.search_term)
    feat['{}_all'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, all_text)

    title = vectorizer.transform(df.product_title)
    feat['{}_title'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, title)

    description = vectorizer.transform(df.product_description)
    feat['{}_description'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, description)

    feat['{}_title_description'.format(prefix)] = cosine_similarity_row_wise(
        title, description)

    attributes = vectorizer.transform(df.attr_texts)
    feat['{}_attributes'.format(prefix)] = cosine_similarity_row_wise(
        search_terms, attributes)

    return feat


def get_golden_features(old_feat):
    logging.info('get golden features')

    feat = pd.DataFrame(index=old_feat.index)
    feat['unigram_title_all_golden'] = \
        old_feat.unigram_title - old_feat.unigram_description
    feat['unigram_bigram_desc_desc_golden'] = \
        old_feat.unigram_description - old_feat.bigram_description
    return feat


@memory.cache
def get_jaccard_similarities(a, b):
    jscores = []
    for x, y in zip(a, b):
        jscores.append(jaccard_similarity_score(x, y))
    return np.array(jscores)


def get_jaccard_features(df):
    logging.info('get jaccard similarity features')
    jaccard_path = '/tmp/jaccard.csv'
    if os.path.exists(jaccard_path):
        feat = pd.read_csv(jaccard_path, index_col=0)
    else:
        feat = pd.DataFrame(index=df.index)
        vectorizer = StemmedCountVectorizer(binary=True, min_df=2)
        text = df.search_term + '\t' + df.product_title + '\t' + \
            df.product_description + '\t' + df.attr_texts
        vectorizer.fit(text)

        search = vectorizer.transform(df.search_term)
        title = vectorizer.transform(df.product_title)
        description = vectorizer.transform(df.product_description)
        brand = vectorizer.transform(df.brand)
        attributes = vectorizer.transform(df.attr_texts)

        feat['jaccard_search_title'] = get_jaccard_similarities(search, title)
        feat['jaccard_search_desc'] = get_jaccard_similarities(
            search, description)
        feat['jaccard_search_attr'] = get_jaccard_similarities(
            search, attributes)
        feat.to_csv(jaccard_path)
    return feat


def get_count_features(df):
    feat = pd.DataFrame(index=df.index)

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
    feat['desc_text_len_ratio'] = feat.desc_text_len_word / feat.desc_text_len_char
    feat['attr_text_len_ratio'] = feat.attr_text_len_word / feat.attr_text_len_char
    feat['attr_text_len_ratio'] = feat.attr_text_len_ratio.fillna(value=0)
    return feat


def get_junk_features(df):
    logging.info('get count features')
    feat = pd.DataFrame(index=df.index)

    # FIXME proper binarization of word counts
    len_of_query = df.search_term.apply(lambda x: len(x.split()))
    feat['two_word_query'] = (len_of_query == 2).astype(int)
    feat['query_in_title'] = get_query_in_title(df)
    return feat


def get_unigram_features(df):
    logging.info('get unigram cosine distances')
    unigram_path = '/tmp/unigram_cosine.csv'
    if os.path.exists(unigram_path):
        feat = pd.read_csv(unigram_path, index_col=0)
    else:
        unigram_vectorizer = StemmedTfidfVectorizer(
            min_df=3, max_df=0.75, stop_words=stopwords,
            strip_accents='unicode', use_idf=1, smooth_idf=1,
            sublinear_tf=1, token_pattern= r'(?u)\b\w\w+\b')
        feat = get_distance_features(unigram_vectorizer, 'unigram', df)
        feat.to_csv(unigram_path)
    return feat


def get_bigram_features(df):
    logging.info('get bigram cosine distances')
    bigram_path = '/tmp/bigram_cosine.csv'
    if os.path.exists(bigram_path):
        feat = pd.read_csv(bigram_path, index_col=0)
    else:
        bigram_vectorizer = StemmedTfidfVectorizer(
            min_df=3, max_df=0.75, stop_words=stopwords,
            strip_accents='unicode', use_idf=1, smooth_idf=1, sublinear_tf=1,
            token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1, 2))
        feat = get_distance_features(bigram_vectorizer, 'bigram', df)
        feat.to_csv(bigram_path)
    return feat


def get_ngram_features(df):
    logging.info('get ngram cosine distances')
    ngram_path = '/tmp/ngram_cosine.csv'
    if os.path.exists(ngram_path):
        feat = pd.read_csv(ngram_path, index_col=0)
    else:
        ngram_vectorizer = TfidfVectorizer(
            min_df=3, max_df=0.75, strip_accents='unicode',
            use_idf=1, smooth_idf=1, sublinear_tf=1, analyzer='char_wb',
            ngram_range=(2, 5))
        feat = get_distance_features(ngram_vectorizer, 'ngram', df)
        feat.to_csv(ngram_path)
    return feat


def get_ngram_without_whitespace_features(df):
    logging.info('get ngram without whitespace cosine distances')
    ngram_ww_path = '/tmp/ngram_ww_cosine.csv'
    if os.path.exists(ngram_ww_path):
        feat = pd.read_csv(ngram_ww_path, index_col=0)
    else:
        vectorizer = TfidfVectorizer(
            min_df=3, max_df=0.75, strip_accents='unicode',
            use_idf=1, smooth_idf=1, sublinear_tf=1, analyzer='char_wb',
            ngram_range=(3, 6))

        feat = pd.DataFrame(index=df.index)
        search_text = df.search_term.apply(
            lambda x: ''.join([c for c in str(x).lower() if c.isalnum()]))
        title_text = df.product_title.apply(
            lambda x: ''.join([c for c in str(x).lower() if c.isalnum()]))
        all_text = search_text + '\t' + title_text
        vectorizer.fit(all_text)

        search_terms = vectorizer.transform(search_text)
        title_terms = vectorizer.transform(title_text)
        feat['ngram_ww_title'] = cosine_similarity_row_wise(
            search_terms, title_terms)

        desc_text = df.product_title.apply(
            lambda x: ''.join([c for c in str(x).lower() if c.isalnum()]))
        all_text = search_text + '\t' + desc_text
        vectorizer.fit(all_text)
        search_terms = vectorizer.transform(search_text)
        desc_terms = vectorizer.transform(title_text)

        feat['ngram_ww_desc'] = cosine_similarity_row_wise(
            search_terms, desc_terms)

        feat.to_csv(ngram_ww_path)
    return feat


def get_ngram_letter_features(df):
    logging.info('get ngram letter cosine distances')
    ngram_l_path = '/tmp/ngram_l_cosine.csv'
    if os.path.exists(ngram_l_path):
        feat = pd.read_csv(ngram_l_path, index_col=0)
    else:
        vectorizer = TfidfVectorizer(
            strip_accents='unicode', use_idf=1, smooth_idf=1,
            sublinear_tf=1, analyzer='char_wb',
            ngram_range=(1, 1))

        feat = pd.DataFrame(index=df.index)
        all_text = df.search_term + '\t' + df.product_title
        vectorizer.fit(all_text)

        search_terms = vectorizer.transform(df.search_term)
        title = vectorizer.transform(df.product_title)
        feat['ngram_letter_title'] = cosine_similarity_row_wise(
            search_terms, title)

        feat.to_csv(ngram_l_path)
    return feat


def get_difflib_features(df):
    logging.info('get difflib features')
    feat = pd.DataFrame(index=df.index)
    seq_distances = []
    for i, (a, b) in enumerate(zip(df.search_term, df.product_title)):
        a = ''.join([c for c in a if c.isalnum()])
        b = ''.join([c for c in b if c.isalnum()])
        seq_distances.append(seq_matcher(None, a, b).ratio())
    feat['seq_match_ratio'] = 1.0 - np.array(seq_distances)
    return feat


def get_features(df):
    logging.info('feature extraction text')
    difflib_feat = get_difflib_features(df)
    junk_feat = get_junk_features(df)
    count_feat = get_count_features(df)
    unigram_feat = get_unigram_features(df)
    bigram_feat = get_bigram_features(df)
    ngram_feat = get_ngram_features(df)
    ngram_ww_feat = get_ngram_without_whitespace_features(df)
    ngram_l_feat = get_ngram_letter_features(df)
    jaccard_feat = get_jaccard_features(df)

    feat = pd.concat([
        junk_feat, count_feat, unigram_feat, bigram_feat, ngram_feat,
        ngram_ww_feat, ngram_l_feat, jaccard_feat, difflib_feat],
        axis=1)

    golden_feat = get_golden_features(feat)

    return pd.concat([feat, golden_feat], axis=1)


if __name__ == "__main__":
    from utils import get_data
    text_features = get_features(get_data())
    text_features.to_csv('/tmp/text.csv')
