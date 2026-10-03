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
    """Raised when a schema in "loaded-only" mode is asked to serialize an
    attribute that was never loaded on the SQLAlchemy instance (a deferred
    column, an expired attribute or an unloaded relationship).

    Accessing such an attribute with ``getattr`` would trigger a lazy SQL
    query (or raise ``DetachedInstanceError`` once the object left its
    session). Raising before any access keeps serialization SQL-free.

    :param path: Full dotted field path, including nested fields.
    :param attribute: Name of the unloaded attribute on the model.
    :param model: The model class of the instance being serialized.
    :param hint: Extra detail about the ORM state of the instance.
    """

    def __init__(
        self,
        *,
        path: str,
        attribute: str,
        model: type | None = None,
        hint: str | None = None,
    ):
        self.path = path
        self.attribute = attribute
        self.model = model
        self.hint = hint
        model_name = getattr(model, "__name__", None)
        message = (
            f"Cannot serialize unloaded attribute {path!r}"
            + (f" on {model_name}" if model_name else "")
            + ": the attribute was not loaded from the database."
        )
        if hint:
            message += f" ({hint})"
        message += (
            " Load it eagerly, or set `unloaded_attribute = 'omit'` to skip it."
        )
        super().__init__(message)

    def prepend_path(self, prefix: str) -> "UnloadedAttributeError":
        """Prepend a nested field name (possibly including a list index) to
        :attr:`path`."""
        if not self.path:
            self.path = prefix
        elif self.path.startswith("["):
            self.path = f"{prefix}{self.path}"
        else:
            self.path = f"{prefix}.{self.path}"
        # Rebuild the message so it reflects the full path.
        self.__init__(
            path=self.path,
            attribute=self.attribute,
            model=self.model,
            hint=self.hint,
        )
        return self
