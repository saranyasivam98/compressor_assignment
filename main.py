# -- coding: UTF-8 --

"""Converting class objects to dataframe"""
import logging
import logging.config
import json
import pandas as pd
import pickle

from compressor.schema_classes import CompressorValuesSchema
from compressor.classes import CompressorRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from marshmallow import ValidationError

__author__ = 'saranya@gyandata.com'

LOGGER = logging.getLogger(__name__)
LOGGER_CONFIG_PATH = 'config/logging.json'


def setup_logging(default_path=LOGGER_CONFIG_PATH):
    """
    Function Description: To setup logging using the json file
    :param default_path: Path of the configuration file
    :type default_path: str
    """
    with open(default_path, 'rt') as file:
        config = json.load(file)
    logging.config.dictConfig(config)


def get_compressor_models_dataframes(df):
    models = list(df['model'].unique())
    model_list = []
    for model in models:
        model_list.append(CompressorRegressor(df[df['model'] == model], model))
    return model_list


def main():

    setup_logging()
    with open("input_data.json") as file:
        compressor_data = json.load(file)

    # Deserializing and returns CompressorValues object
    try:
        compressor_data = CompressorValuesSchema(many=True).load(compressor_data)
    except ValidationError as err:
        LOGGER.error(err.messages)

    # Converting the list of objects to a dataframe
    df = pd.DataFrame([data.to_dict() for data in compressor_data])

    # Getting the individual dataframe for each model
    models_regressor = get_compressor_models_dataframes(df)

    # Training the model and saving it as pickle file
    for model in models_regressor:
        x_train, x_test, y_train, y_test = train_test_split(model.x, model.y, test_size=0.25, random_state=0)
        model.fit_polynomial_regression(x_train, y_train)
        y_pred = model.predict(x_test)
        LOGGER.info("Accuracy for the model: %s is %s" % (model.model_name, r2_score(y_test, y_pred)))
        file_name = model.model_name + '.pkl'
        model.save_model(file_name)

    model_name = 'MT064-4'
    df_model = df[df['model'] == model_name]

    obj = CompressorRegressor(df_model, model_name)
    obj.load_model('MT064-4.pkl')
    x_train, x_test, y_train, y_test = train_test_split(obj.x, obj.y, test_size=0.25, random_state=42)
    y_pred = obj.predict(x_test)
    LOGGER.info("Accuracy for the model: %s after loading from pickle file is %s" % (obj.model_name, r2_score(y_test, y_pred)))


if __name__ == '__main__':
    main()
