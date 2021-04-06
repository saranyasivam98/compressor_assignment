# -- coding: UTF-8 --

"""
*****************
Schema Classes
*****************
Schema classes for validating the compressor data and to create class object from the schema class """


from marshmallow import Schema, fields, post_load, validate
from compressor.classes import CompressorValues, CompressorSpecifications

__author__ = 'saranya@gyandata.com'


class CompressorSpecificationSchema(Schema):
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
    model = fields.Str()
    technology = fields.Str(validate=validate.OneOf(["Reciprocating", "Scroll"]))
    refrigerant = fields.Str(validate=validate.OneOf(["R407A", "R22", "R488A", "R410A", "R404A", "R448"]))
    capacity_control = fields.Str(validate=validate.OneOf(["Fixed Speed", "Variable Speed"]))

    @post_load()
    def make_compressor_property(self, data, **kwargs):
        return CompressorSpecifications(**data)


class CompressorValuesSchema(Schema):  # to-do: validation for temp values
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
    specs = fields.Nested(CompressorSpecificationSchema())
    condenser_temp = fields.Float(validate=validate.Range(min=-273.15))
    evaporator_temp = fields.Float()
    power = fields.Float()

    @post_load()
    def make_compressor_values(self, data, **kwargs):
        return CompressorValues(**data)
