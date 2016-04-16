import logging
import pandas as pd

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)


def get_numeric_attribute(df, df_attr, attr_name):
    product_lookup = {}
    for idx, row in df_attr[df_attr['name'] == attr_name].iterrows():
        product_lookup[row.product_uid] = float(''.join([d for d in row.value if d.isnumeric()]))
    values = pd.Series(0.0, index=df.index)
    for i, product_uid in enumerate(df.product_uid.values):
        if product_uid in product_lookup:
            values[i] = product_lookup[product_uid]
    return values


def get_features(df, df_attr):
    logging.info('feature extraction attributes')
    feat = pd.DataFrame(index=df.index)

    # product size
    feat['product_width'] = get_numeric_attribute(
        df, df_attr, 'Product Width (in.)')
    feat['product_height'] = get_numeric_attribute(
        df, df_attr, 'Product Height (in.)')
    feat['product_depth'] = get_numeric_attribute(
        df, df_attr, 'Product Depth (in.)')
    feat['product_weight'] = get_numeric_attribute(
        df, df_attr, 'Product Weight (lb.)')

    feat['product_size'] = feat.product_width * feat.product_height * \
        feat.product_depth

    return feat


if __name__ == "__main__":
    from utils import get_data
    from utils import get_attribute_data
    attribute_features = get_features(get_data(), get_attribute_data())
    attribute_features.to_csv('/tmp/attribute.csv')
