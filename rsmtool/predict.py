"""
Functions dealing with making predictions

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""
import logging

import numpy as np
import pandas as pd

from skll import FeatureSet

from rsmtool.preprocess import trim

def predict_with_model(model, df):
    """
    Get the raw predictions of the `model` on the data
    contained in the data frame `df`.
    """

    logger = logging.getLogger(__name__)

    feature_columns = [c for c in df.columns if c not in ['spkitemid', 'sc1']]
    features = df[feature_columns].to_dict(orient='records')
    ids = df['spkitemid'].tolist()

    # if we have the labels, save them in the featureset
    labels = None
    if 'sc1' in df:
        labels = df['sc1'].tolist()

    fs = FeatureSet('data', ids=ids, labels=labels, features=features)
    predictions = model.predict(fs)

    df_predictions = pd.DataFrame()
    df_predictions['spkitemid'] = ids
    df_predictions['raw'] = predictions

    # save the labels in the dataframe if they existed in the first place
    if labels:
        df_predictions['sc1'] = labels

    return df_predictions

def generate_train_and_test_predictions(model, df_train, df_test, trim_min, trim_max):
    """
    Generate raw, scaled, and trimmed predictions of `model`
    on the given training and testing data.
    """

    logger = logging.getLogger(__name__)

    df_train_predictions = predict_with_model(model, df_train)
    df_test_predictions = predict_with_model(model, df_test)

    # get the mean and SD of the training set predictions
    train_predictions_mean = df_train_predictions['raw'].mean()
    train_predictions_sd = df_train_predictions['raw'].std()

    # get the mean and SD of the human labels
    human_labels_mean = df_train['sc1'].mean()
    human_labels_sd = df_train['sc1'].std()

    logger.info('Processing test set predictions')

    df_test_predictions = process_predictions(df_test_predictions,
                                              train_predictions_mean,
                                              train_predictions_sd,
                                              human_labels_mean,
                                              human_labels_sd,
                                              trim_min, trim_max)

    return (df_train_predictions, df_test_predictions,
            train_predictions_mean, train_predictions_sd,
            human_labels_mean, human_labels_sd)


def process_predictions(df_test_predictions,
                        train_predictions_mean,
                        train_predictions_sd,
                        human_labels_mean,
                        human_labels_sd,
                        trim_min, trim_max):

    """
    Process predictions to create scaled, trimmed
    and rounded predictions.
    """

    # rescale the test set predictions by boosting
    # them to match the human mean and SD
    scaled_test_predictions = (df_test_predictions['raw'] - train_predictions_mean) / train_predictions_sd
    scaled_test_predictions = scaled_test_predictions * human_labels_sd + human_labels_mean

    df_pred_processed = df_test_predictions.copy()
    df_pred_processed['scale'] = scaled_test_predictions

    # trim and round the predictions before running the analyses
    df_pred_processed['raw_trim'] = trim(df_pred_processed['raw'],
                                         trim_min,
                                         trim_max)

    df_pred_processed['raw_trim_round'] = np.rint(df_pred_processed['raw_trim']).astype('int64')

    df_pred_processed['scale_trim'] = trim(df_pred_processed['scale'],
                                           trim_min,
                                           trim_max)

    df_pred_processed['scale_trim_round'] = np.rint(df_pred_processed['scale_trim']).astype('int64')

    return df_pred_processed
