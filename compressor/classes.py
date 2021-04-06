# -- coding: UTF-8 --

"""
*****************
Classes
*****************
Classes for compressor property and compressor values and the regressor model
"""
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

__author__ = 'saranya@gyandata.com'


class CompressorSpecifications:
    """
    To store the properties of compressor

    :ivar model: Model of the compressor
    :vartype model: fields.Str

    :ivar technology: Technology of the motor
    :vartype technology: fields.Str

    :ivar refrigerant: Refrigerant used in the compressor
    :vartype refrigerant: fields.Str

    :ivar capacity_control: Speed control of the compressor
    :vartype capacity_control: fields.Str
    """

    def __init__(self, model, technology, refrigerant, capacity_control):
        self.model = model
        self.technology = technology
        self.refrigerant = refrigerant
        self.capacity_control = capacity_control


class CompressorValues:
    """
    To store the compressor values

    :ivar model: Model of the compressor
    :vartype model: fields.Str

    :ivar condenser_temp: Inlet temperature to condenser
    :vartype condenser_temp: fields.Float

    :ivar evaporator_temp: Outlet temperature to condenser
    :vartype evaporator_temp: fields.Float

    :ivar power: Power required for the compressor
    :vartype power: fields.Float

    """

    def __init__(self, specs, condenser_temp, evaporator_temp, power):
        self.specs = specs
        self.condenser_temp = condenser_temp
        self.evaporator_temp = evaporator_temp
        self.power = power

    def to_dict(self):
        return_dict = {
            "model": self.specs.model,
            "technology": self.specs.technology,
            "refrigerant": self.specs.refrigerant,
            "capacity_control": self.specs.capacity_control,
            "condenser_temp": self.condenser_temp, "evaporator_temp": self.evaporator_temp, "power": self.power}
        return return_dict


class CompressorRegressor:
    """
    To perform regression on the different compressor models with condenser and evaporator temperatures as input
    parameters and power as output parameter

    :ivar df: The dataset which contains the input and output parameters for a particular model.
    :vartype df: :class:`pandas.DataFrame`

    :ivar model: Model of the compressor
    :vartype model: str

    :ivar x: Input parameters and its values
    :vartype x: :class:`pandas.DataFrame`

    :ivar y: Output parameter and its values
    :vartype y: float

    :ivar lin_reg: Object of LinearRegression
    :vartype lin_reg: class:`sklearn.linear_model.LinearRegression`

    :ivar poly_reg: Object of PolynomialFeatures of degree 2
    :vartype poly_reg: class:`sklearn.preprocessing.PolynomialFeatures`

    """
    def __init__(self, df, model):
        self.df = df
        self.model_name = model
        self.x = self.df[['condenser_temp', 'evaporator_temp']].values
        self.y = self.df[['power']].values
        self.lin_reg = LinearRegression()
        self.poly_reg = PolynomialFeatures(degree=2)

    def fit_polynomial_regression(self, x_train, y_train):
        """
        To preprocess the data with polynomial features and fit the data

        :param x_train: Input variables of training dataset
        :type x_train: :class:`numpy.ndarray`

        :param y_train: Output variable of training dataset
        :type y_train: :class:`numpy.ndarray`

        :return: The model that fits preprocessed x_train and y_train
        """
        x_poly = self.poly_reg.fit_transform(x_train)
        self.lin_reg.fit(x_poly, y_train)

    def predict(self, x_test):
        """
        To predict the values for a given x_test

        :param x_test: Input parameters of Test dataset
        :type x_test: :class:`numpy.ndarray`

        :return: The predicted values for x_test
        :rtype: :class:`numpy.ndarray`
        """
        return self.lin_reg.predict(self.poly_reg.fit_transform(x_test))

    def save_model(self, file_name):
        """
        To save the model using pickle

        :param file_name: Name of the file
        :type file_name: str

        :return: None
        """
        with open(file_name, 'wb') as file:
            pickle.dump(self.lin_reg, file)

    def load_model(self, file_name):
        """
        To open the pickle model

        :param file_name: Name of the pickle file
        :type file_name: str

        :return: None
        """
        with open(file_name, 'rb') as file:
            self.lin_reg = pickle.load(file)
