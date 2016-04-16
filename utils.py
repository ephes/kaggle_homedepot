import pandas as pd

from sklearn.metrics import mean_squared_error, make_scorer


def get_attribute_data():
    df_attr = pd.read_csv('attributes.csv')
    df_attr = df_attr.dropna()
    df_attr['product_uid'] = df_attr.product_uid.astype(int)
    return df_attr


def get_data():
    df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
    df_train = df_train[~df_train.relevance.isin([2.75, 2.5, 2.25, 1.75, 1.50, 1.25])]
    num_train = df_train.shape[0]
    df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv('product_descriptions.csv')
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df = pd.merge(df, df_pro_desc, how='left', on='product_uid')

    df_attr = pd.read_csv('attributes.csv')
    df_attr = df_attr.dropna()
    df_attr['product_uid'] = df_attr.product_uid.astype(int)

    df_attr_counts = df_attr.product_uid.value_counts().to_frame(name='attr_count')
    df_attr_counts['product_uid'] = df_attr_counts.index
    df = pd.merge(df, df_attr_counts, how='left', on='product_uid')
    df.attr_count.fillna(value=0, inplace=True)
    df['attr_count'] = df.attr_count.astype(int)

    df_attr_texts = df_attr.groupby('product_uid')['value'].apply(lambda x: '\t'.join(x)).to_frame(name='attr_texts')
    df_attr_texts['product_uid'] = df_attr_texts.index
    df = pd.merge(df, df_attr_texts, how='left', on='product_uid')
    df['attr_texts'] = df.attr_texts.fillna(value='')


    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    df = pd.merge(df, df_brand, how='left', on='product_uid')
    df['brand'] = df.brand.fillna(value='')

    return df


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)
