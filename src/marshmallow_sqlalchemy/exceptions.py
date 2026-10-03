import marshmallow as ma


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


class DuplicateTargetError(ma.ValidationError, MarshmallowSQLAlchemyError):
    """Raised during two-phase loading when more than one input record
    resolves to the same existing ORM object and the configured
    ``on_duplicate`` policy does not resolve the conflict.

    Subclasses :class:`marshmallow.ValidationError` so callers that handle
    validation failures uniformly also catch this error.
    """

    def __init__(self, message, *, model=None, identity=None):
        super().__init__(message)
        self.model = model
        self.identity = identity
