class MarshmallowSQLAlchemyError(Exception):
    """Base exception class from which all exceptions related to
    marshmallow-sqlalchemy inherit.
    """


class ModelConversionError(MarshmallowSQLAlchemyError):
    """Raised when an error occurs in converting a SQLAlchemy construct
    to a marshmallow object.
    """


class IncorrectSchemaTypeError(ModelConversionError):
    """Raised when a ``SQLAlchemyAutoField`` is bound to ``Schema`` that
    is not an instance of ``SQLAlchemySchema``.
    """


class UnloadedAttributeError(MarshmallowSQLAlchemyError):
    """Raised during a "loaded-only" dump when a requested attribute has not
    been loaded onto the SQLAlchemy instance (for example because it is
    deferred, expired, or belongs to an unloaded relationship).

    The message always carries the full dotted path of the offending field
    relative to the object passed to ``dump``.
    """
