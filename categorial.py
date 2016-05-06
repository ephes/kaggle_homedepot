import logging
import numpy as np
import pandas as pd

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)


def filter_in_train_and_test(feat, attr, is_test):
    test_ids = set(feat[attr][is_test])
    train_ids = set(feat[attr][~is_test])

    relevant_ids = test_ids.intersection(train_ids)
    feat[attr][~feat[attr].isin(relevant_ids)] = 0
    return feat


def get_query_ids(df):
    feat = pd.DataFrame(index=df.index)

    qid_lookup = {}
    for i, q in enumerate(df.search_term.str.lower().unique(), start=1):
        qid_lookup[q] = i

    feat['query_id'] = (df.search_term.str.lower()
                            .apply(lambda q: qid_lookup[q]))

    return filter_in_train_and_test(feat, 'query_id', df.relevance.isnull())


def get_color_ids(df, df_attr):
    feat = pd.DataFrame(index=df.index)
    color_num = 1
    color_lookup = {}
    product_color = {}
    for idx, row in df_attr[df_attr.name == 'Color/Finish'].iterrows():
        color = row.value
        if color not in color_lookup:
            color_lookup[color] = color_num
            color_num += 1
        product_color[row.product_uid] = color_lookup[color]
    color_ids = pd.Series(0, index=df.index, dtype=np.int)
    for i, product_uid in enumerate(df.product_uid.values):
        if product_uid in product_color:
            color_ids[i] = product_color[product_uid]
    feat['color_id'] = color_ids
    return filter_in_train_and_test(feat, 'color_id', df.relevance.isnull())


def get_features(df, df_attr):
    logging.info('categorial (id) feature extraction')
    feat = pd.DataFrame(index=df.index)

    feat['product_uid'] = df.product_uid
    feat['product_uid'] = filter_in_train_and_test(
        feat, 'product_uid', df.relevance.isnull())

    feat['product_freq'] = \
        feat.groupby('product_uid')['product_uid'].transform('count')
    feat_qid = get_query_ids(df)
    feat_color = get_color_ids(df, df_attr)

    return pd.concat([feat, feat_qid, feat_color], axis=1)


if __name__ == "__main__":
    from utils import get_data
    from utils import get_attribute_data
    categorial_features = get_features(get_data(), get_attribute_data())
    categorial_features.to_csv('/tmp/categorial.csv')
