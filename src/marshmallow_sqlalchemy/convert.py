from __future__ import annotations

import copy
import functools
import inspect
import uuid
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

# Remove when dropping Python 3.9
try:
    from typing import TypeAlias, TypeGuard
except ImportError:
    from typing import TypeAlias, TypeGuard

import marshmallow as ma
import sqlalchemy as sa
from marshmallow import fields, validate
from marshmallow.utils import missing
from sqlalchemy.dialects import mssql, mysql, postgresql
from sqlalchemy.orm import CompositeProperty, SynonymProperty

from .exceptions import ModelConversionError
from .fields import Composite, Related, RelatedList

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.declarative import DeclarativeMeta
    from sqlalchemy.orm import MapperProperty
    from sqlalchemy.types import TypeEngine

    PropertyOrColumn: TypeAlias = MapperProperty | sa.Column

_FieldPartial: TypeAlias = Callable[[], fields.Field]
# TODO: Use more specific type for second argument
_FieldClassFactory: TypeAlias = Callable[
    ["ModelConverter", Any], type[fields.Field] | _FieldPartial
]
CompositeFieldRule: TypeAlias = (
    fields.Field | type[fields.Field] | _FieldPartial | dict[str, Any] | None
)
CompositeFieldFactory: TypeAlias = Callable[["ModelConverter", Any], CompositeFieldRule]


def _is_field(value: Any) -> TypeGuard[type[fields.Field]]:
    return isinstance(value, type) and issubclass(value, fields.Field)


def _base_column(column):
    """Unwrap proxied columns"""
    if column not in column.base_columns and len(column.base_columns) == 1:
        [base] = column.base_columns
        return base
    return column


def _has_default(column) -> bool:
    return (
        column.default is not None
        or column.server_default is not None
        or _is_auto_increment(column)
    )


def _is_auto_increment(column) -> bool:
    return column.table is not None and column is column.table._autoincrement_column


def _list_field_factory(
    converter: ModelConverter, data_type: postgresql.ARRAY
) -> Callable[[], fields.List]:
    FieldClass = converter._get_field_class_for_data_type(data_type.item_type)
    inner = FieldClass()
    if not data_type.dimensions or data_type.dimensions == 1:
        return functools.partial(fields.List, inner)

    # For multi-dimensional arrays, nest the Lists
    dimensions = data_type.dimensions
    for _ in range(dimensions - 1):
        inner = fields.List(inner)

    return functools.partial(fields.List, inner)


def _enum_field_factory(
    converter: ModelConverter, data_type: sa.Enum
) -> Callable[[], fields.Field]:
    return (
        functools.partial(fields.Enum, enum=data_type.enum_class)
        if data_type.enum_class
        else fields.Raw
    )


class ModelConverter:
    """Converts a SQLAlchemy model into a dictionary of corresponding
    marshmallow `Fields <marshmallow.fields.Field>`.
    """

    SQLA_TYPE_MAPPING: dict[
        type[TypeEngine], type[fields.Field] | _FieldClassFactory
    ] = {
        sa.Enum: _enum_field_factory,
        sa.JSON: fields.Raw,
        sa.ARRAY: _list_field_factory,
        sa.PickleType: fields.Raw,
        postgresql.BIT: fields.Integer,
        postgresql.OID: fields.Integer,
        postgresql.UUID: fields.UUID,
        postgresql.MACADDR: fields.String,
        postgresql.INET: fields.String,
        postgresql.CIDR: fields.String,
        postgresql.JSON: fields.Raw,
        postgresql.JSONB: fields.Raw,
        postgresql.HSTORE: fields.Raw,
        postgresql.ARRAY: _list_field_factory,
        postgresql.MONEY: fields.Decimal,
        postgresql.DATE: fields.Date,
        postgresql.TIME: fields.Time,
        mysql.BIT: fields.Integer,
        mysql.YEAR: fields.Integer,
        mysql.SET: fields.List,
        mysql.ENUM: fields.Field,
        mysql.INTEGER: fields.Integer,
        mysql.DATETIME: fields.DateTime,
        mssql.BIT: fields.Integer,
        mssql.UNIQUEIDENTIFIER: fields.UUID,
    }
    COMPOSITE_TYPE_MAPPING: dict[
        type, CompositeFieldRule | CompositeFieldFactory
    ] = {}
    DIRECTION_MAPPING = {"MANYTOONE": False, "MANYTOMANY": True, "ONETOMANY": True}

    def __init__(self, schema_cls: type[ma.Schema] | None = None):
        self.schema_cls = schema_cls

    @property
    def type_mapping(self) -> dict[type, type[fields.Field]]:
        if self.schema_cls:
            return self.schema_cls.TYPE_MAPPING
        return ma.Schema.TYPE_MAPPING

    def fields_for_model(
        self,
        model: type[DeclarativeMeta],
        *,
        include_fk: bool = False,
        include_relationships: bool = False,
        fields: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        base_fields: dict | None = None,
        dict_cls: type[dict] = dict,
    ) -> dict[str, fields.Field]:
        """Generate a dict of field_name: `marshmallow.fields.Field` pairs for the given model.
        Note: SynonymProperties are ignored. Use an explicit field if you want to include a synonym.

        :param model: The SQLAlchemy model
        :param bool include_fk: Whether to include foreign key fields in the output.
        :param bool include_relationships: Whether to include relationships fields in the output.
        :return: dict of field_name: Field instance pairs
        """
        result = dict_cls()
        base_fields = base_fields or {}
        composite_component_keys = self._get_active_composite_component_keys(
            model, fields=fields, exclude=exclude
        )

        for prop in sa.inspect(model).attrs:  # type: ignore[union-attr]
            key = self._get_field_name(prop)
            if self._should_exclude_field(prop, fields=fields, exclude=exclude):
                # Allow marshmallow to validate and exclude the field key.
                result[key] = None
                continue
            if isinstance(prop, CompositeProperty):
                if self._is_generated_field(base_fields.get(key)):
                    continue
            elif key in composite_component_keys and key not in base_fields:
                continue
            if isinstance(prop, SynonymProperty):
                continue
            if hasattr(prop, "columns"):
                if not include_fk:
                    # Only skip a column if there is no overriden column
                    # which does not have a Foreign Key.
                    for column in prop.columns:
                        if not column.foreign_keys:
                            break
                    else:
                        continue
            if not include_relationships and hasattr(prop, "direction"):
                continue
            field = base_fields.get(key) or self.property2field(prop)
            if field:
                result[key] = field
        return result

    def fields_for_table(
        self,
        table: sa.Table,
        *,
        include_fk: bool = False,
        fields: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        base_fields: dict | None = None,
        dict_cls: type[dict] = dict,
    ) -> dict[str, fields.Field]:
        result = dict_cls()
        base_fields = base_fields or {}
        for column in table.columns:
            key = self._get_field_name(column)
            if self._should_exclude_field(column, fields=fields, exclude=exclude):
                # Allow marshmallow to validate and exclude the field key.
                result[key] = None
                continue
            if not include_fk and column.foreign_keys:
                continue
            # Overridden fields are specified relative to key generated by
            # self._get_key_for_column(...), rather than keys in source model
            field = base_fields.get(key) or self.column2field(column)
            if field:
                result[key] = field
        return result

    @overload
    def property2field(
        self,
        prop: MapperProperty,
        *,
        instance: Literal[True] = ...,
        field_class: type[fields.Field] | None = ...,
        **kwargs,
    ) -> fields.Field: ...

    @overload
    def property2field(
        self,
        prop: MapperProperty,
        *,
        instance: Literal[False] = ...,
        field_class: type[fields.Field] | None = ...,
        **kwargs,
    ) -> type[fields.Field]: ...

    def property2field(
        self,
        prop: MapperProperty,
        *,
        instance: bool = True,
        field_class: type[fields.Field] | None = None,
        **kwargs,
    ) -> fields.Field | type[fields.Field]:
        """Convert a SQLAlchemy `Property` to a field instance or class.

        :param Property prop: SQLAlchemy Property.
        :param bool instance: If `True`, return  `Field` instance, computing relevant kwargs
            from the given property. If `False`, return the `Field` class.
        :param kwargs: Additional keyword arguments to pass to the field constructor.
        :return: A `marshmallow.fields.Field` class or instance.
        """
        # handle synonyms
        # Attribute renamed "_proxied_object" in 1.4
        for attr in ("_proxied_property", "_proxied_object"):
            proxied_obj = getattr(prop, attr, None)
            if proxied_obj is not None:
                prop = proxied_obj
        if isinstance(prop, CompositeProperty):
            return self.composite2field(prop, instance=instance, **kwargs)
        field_class = field_class or self._get_field_class_for_property(prop)
        if not instance:
            return field_class
        field_kwargs = self._get_field_kwargs_for_property(prop)
        field_kwargs.update(kwargs)
        ret = field_class(**field_kwargs)
        if (
            hasattr(prop, "direction")
            and self.DIRECTION_MAPPING[prop.direction.name]
            and prop.uselist is True
        ):
            ret = RelatedList(ret, **{**self.get_base_kwargs(), **kwargs})
        return ret

    @overload
    def column2field(
        self, column, *, instance: Literal[True] = ..., **kwargs
    ) -> fields.Field: ...

    @overload
    def column2field(
        self, column, *, instance: Literal[False] = ..., **kwargs
    ) -> type[fields.Field]: ...

    def column2field(
        self, column, *, instance: bool = True, **kwargs
    ) -> fields.Field | type[fields.Field]:
        """Convert a SQLAlchemy `Column <sqlalchemy.schema.Column>` to a field instance or class.

        :param sqlalchemy.schema.Column column: SQLAlchemy Column.
        :param bool instance: If `True`, return  `Field` instance, computing relevant kwargs
            from the given property. If `False`, return the `Field` class.
        :return: A `marshmallow.fields.Field` class or instance.
        """
        field_class = self._get_field_class_for_column(column)
        if not instance:
            return field_class
        field_kwargs = self.get_base_kwargs()
        self._add_column_kwargs(field_kwargs, column)
        return field_class(**{**field_kwargs, **kwargs})

    @overload
    def field_for(
        self,
        model: type[DeclarativeMeta],
        property_name: str,
        *,
        instance: Literal[True] = ...,
        field_class: type[fields.Field] | None = ...,
        **kwargs,
    ) -> fields.Field: ...

    @overload
    def field_for(
        self,
        model: type[DeclarativeMeta],
        property_name: str,
        *,
        instance: Literal[False] = ...,
        field_class: type[fields.Field] | None = None,
        **kwargs,
    ) -> type[fields.Field]: ...

    def field_for(
        self,
        model: type[DeclarativeMeta],
        property_name: str,
        *,
        instance: bool = True,
        field_class: type[fields.Field] | None = None,
        **kwargs,
    ) -> fields.Field | type[fields.Field]:
        """Convert a property for a mapped SQLAlchemy class to a marshmallow `Field`.
        Example: ::

            date_created = field_for(Author, "date_created", dump_only=True)
            author = field_for(Book, "author")

        :param type model: A SQLAlchemy mapped class.
        :param str property_name: The name of the property to convert.
        :param kwargs: Extra keyword arguments to pass to `property2field`
        :return: A `marshmallow.fields.Field` class or instance.
        """
        target_model = model
        prop_name = property_name
        attr = getattr(model, property_name)
        remote_with_local_multiplicity = False
        if hasattr(attr, "remote_attr"):
            target_model = attr.target_class
            prop_name = attr.value_attr
            remote_with_local_multiplicity = attr.local_attr.prop.uselist
        prop: MapperProperty = sa.inspect(target_model).attrs.get(prop_name)  # type: ignore[union-attr]
        converted_prop = self.property2field(
            prop,
            # To satisfy type checking, need to pass a literal bool
            instance=True if instance else False,  # noqa: SIM210
            field_class=field_class,
            **kwargs,
        )
        if remote_with_local_multiplicity:
            return RelatedList(converted_prop, **{**self.get_base_kwargs(), **kwargs})
        return converted_prop

    def composite2field(
        self,
        prop: CompositeProperty,
        *,
        instance: bool = True,
        field_class: type[fields.Field] | None = None,
        **kwargs,
    ) -> fields.Field | type[fields.Field]:
        """Convert a SQLAlchemy composite property to a nested field."""
        field_class = field_class or Composite
        if not instance:
            return field_class

        component_props = self._get_composite_component_properties(prop)
        component_keys = [component.key for component in component_props]
        field_kwargs = self._get_composite_field_kwargs(prop, component_props)
        field_kwargs.update(kwargs)
        registered_rule = self._get_registered_composite_rule(prop.composite_class)
        if self._is_composite_field_factory(registered_rule):
            registered_rule = cast("CompositeFieldFactory", registered_rule)(
                self, prop
            )

        if isinstance(registered_rule, ma.Schema) or (
            isinstance(registered_rule, type)
            and issubclass(registered_rule, ma.Schema)
        ):
            nested_schema = registered_rule
        else:
            inner_fields = self._get_composite_field_rules(prop, component_props)
            nested_schema = ma.Schema.from_dict(
                {
                    component.key: self._prepare_composite_inner_field(
                        inner_fields[component.key], component
                    )
                    for component in component_props
                },
                name=f"{prop.composite_class.__name__}Schema",
            )
        return field_class(
            nested_schema,
            prop.composite_class,
            component_keys,
            construct=prop._construct_composite,
            value_getter=prop._composite_values_from_instance,
            **field_kwargs,
        )

    def _get_field_name(self, prop_or_column: PropertyOrColumn) -> str:
        return prop_or_column.key

    def _get_field_class_for_column(self, column: sa.Column) -> type[fields.Field]:
        return self._get_field_class_for_data_type(column.type)

    def _get_field_class_for_data_type(
        self, data_type: TypeEngine
    ) -> type[fields.Field]:
        field_cls: type[fields.Field] | _FieldPartial | None = None
        types = inspect.getmro(type(data_type))
        # First search for a field class from self.SQLA_TYPE_MAPPING
        for col_type in types:
            if col_type in self.SQLA_TYPE_MAPPING:
                field_or_factory = self.SQLA_TYPE_MAPPING[col_type]
                if _is_field(field_or_factory):
                    field_cls = field_or_factory
                else:
                    field_cls = cast("_FieldClassFactory", field_or_factory)(
                        self, data_type
                    )
                break
        else:
            # Try to find a field class based on the column's python_type
            try:
                python_type = data_type.python_type
            except NotImplementedError:
                python_type = None

            if python_type in self.type_mapping:
                field_cls = self.type_mapping[python_type]
            else:
                if hasattr(data_type, "impl"):
                    return self._get_field_class_for_data_type(data_type.impl)
                raise ModelConversionError(
                    f"Could not find field column of type {types[0]}."
                )
        return cast("type[fields.Field]", field_cls)

    def _get_field_class_for_property(self, prop) -> type[fields.Field]:
        field_cls: type[fields.Field]
        if hasattr(prop, "direction"):
            field_cls = Related
        elif isinstance(prop, CompositeProperty):
            field_cls = Composite
        else:
            column = _base_column(prop.columns[0])
            field_cls = self._get_field_class_for_column(column)
        return field_cls

    def _get_composite_component_properties(
        self, prop: CompositeProperty
    ) -> list[Any]:
        if getattr(prop, "props", None):
            return list(prop.props)
        mapper = prop.parent
        return [mapper.attrs[attribute] for attribute in prop.attrs]

    def _get_active_composite_component_keys(
        self,
        model: type[DeclarativeMeta],
        *,
        fields: Iterable[str] | None,
        exclude: Iterable[str] | None,
    ) -> set[str]:
        keys: set[str] = set()
        for prop in sa.inspect(model).attrs:  # type: ignore[union-attr]
            if isinstance(prop, CompositeProperty) and not self._should_exclude_field(
                prop, fields=fields, exclude=exclude
            ):
                keys.update(
                    component.key
                    for component in self._get_composite_component_properties(prop)
                )
        return keys

    def _get_composite_field_rules(
        self,
        prop: CompositeProperty,
        component_props: list[Any],
    ) -> dict[str, CompositeFieldRule]:
        explicit_rule = self._get_registered_composite_rule(prop.composite_class)
        if self._is_composite_field_factory(explicit_rule):
            explicit_rule = cast("CompositeFieldFactory", explicit_rule)(
                self, prop
            )

        if explicit_rule is None:
            return {
                component.key: self.property2field(component)
                for component in component_props
            }

        if isinstance(explicit_rule, dict):
            rules = {
                component.key: self.property2field(component)
                for component in component_props
            }
            rules.update(explicit_rule)
        else:
            field = self._resolve_field_rule(explicit_rule)
            rules = {component.key: field for component in component_props}

        return rules

    def _get_registered_composite_rule(
        self, composite_class: type
    ) -> CompositeFieldRule:
        for composite_type in inspect.getmro(composite_class):
            if composite_type in self.COMPOSITE_TYPE_MAPPING:
                return self.COMPOSITE_TYPE_MAPPING[composite_type]
        return None

    def _is_composite_field_factory(self, rule: Any) -> bool:
        if not callable(rule) or isinstance(rule, type):
            return False
        try:
            parameters = inspect.signature(rule).parameters
        except (TypeError, ValueError):
            return False
        required_positional = sum(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Parameter.empty
            for parameter in parameters.values()
        )
        has_var_positional = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters.values()
        )
        return required_positional >= 2 or has_var_positional

    def _resolve_field_rule(self, rule: CompositeFieldRule) -> fields.Field:
        if isinstance(rule, fields.Field):
            return copy.deepcopy(rule)
        if _is_field(rule):
            return cast("type[fields.Field]", rule)()
        if callable(rule):
            resolved = cast("_FieldPartial", rule)()
            if isinstance(resolved, fields.Field):
                return resolved
            return cast("type[fields.Field]", resolved)()
        raise ModelConversionError(f"Unsupported composite field rule: {rule!r}")

    def _prepare_composite_inner_field(
        self, rule: CompositeFieldRule, component: Any
    ) -> fields.Field:
        field = self._resolve_field_rule(rule)
        column = (
            _base_column(component.columns[0])
            if hasattr(component, "columns") and component.columns
            else None
        )
        if column is not None and column.nullable and field.load_default is missing:
            field.load_default = None
        return field

    def _get_composite_field_kwargs(
        self, prop: CompositeProperty, component_props: list[Any]
    ) -> dict[str, Any]:
        columns = [
            _base_column(component.columns[0])
            for component in component_props
            if hasattr(component, "columns") and component.columns
        ]
        nullable = bool(columns) and all(column.nullable for column in columns)
        required = bool(columns) and any(
            not column.nullable and not _has_default(column) for column in columns
        )
        kwargs = self.get_base_kwargs()
        kwargs.update(
            {
                "allow_none": nullable,
                "required": required,
            }
        )
        if getattr(prop, "doc", None):
            kwargs["metadata"]["description"] = prop.doc
        return kwargs

    @staticmethod
    def _is_generated_field(field: Any) -> bool:
        from .schema import SQLAlchemyAutoField  # noqa: PLC0415

        return isinstance(field, SQLAlchemyAutoField)

    def _get_field_kwargs_for_property(self, prop: PropertyOrColumn) -> dict[str, Any]:
        kwargs = self.get_base_kwargs()
        if hasattr(prop, "columns"):
            column = _base_column(prop.columns[0])
            self._add_column_kwargs(kwargs, column)
            prop = column
        if hasattr(prop, "direction"):  # Relationship property
            self._add_relationship_kwargs(kwargs, prop)
        if getattr(prop, "doc", None):  # Useful for documentation generation
            kwargs["metadata"]["description"] = prop.doc
        return kwargs

    def _add_column_kwargs(self, kwargs: dict[str, Any], column: sa.Column) -> None:
        """Add keyword arguments to kwargs (in-place) based on the passed in
        `Column <sqlalchemy.schema.Column>`.
        """
        if hasattr(column, "nullable"):
            if column.nullable:
                kwargs["allow_none"] = True
            kwargs["required"] = not column.nullable and not _has_default(column)
        # If there is no nullable attribute, we are dealing with a property
        # that does not derive from the Column class. Mark as dump_only.
        else:
            kwargs["dump_only"] = True

        if hasattr(column.type, "enum_class") and column.type.enum_class is not None:
            kwargs["enum"] = column.type.enum_class
        elif hasattr(column.type, "enums") and not kwargs.get("dump_only"):
            kwargs["validate"].append(validate.OneOf(choices=column.type.enums))

        # Add a length validator if a max length is set on the column
        # Skip UUID columns
        # (see https://github.com/marshmallow-code/marshmallow-sqlalchemy/issues/54)
        if hasattr(column.type, "length") and not kwargs.get("dump_only"):
            column_length = column.type.length
            if column_length is not None:
                try:
                    python_type = column.type.python_type
                except (AttributeError, NotImplementedError):
                    python_type = None
                if not python_type or not issubclass(python_type, uuid.UUID):
                    kwargs["validate"].append(validate.Length(max=column_length))

        if getattr(column.type, "asdecimal", False):
            kwargs["places"] = getattr(column.type, "scale", None)

    def _add_relationship_kwargs(
        self, kwargs: dict[str, Any], prop: PropertyOrColumn
    ) -> None:
        """Add keyword arguments to kwargs (in-place) based on the passed in
        relationship `Property`.
        """
        nullable = True
        for pair in prop.local_remote_pairs:
            if not pair[0].nullable:
                if (
                    prop.uselist is True
                    or self.DIRECTION_MAPPING[prop.direction.name] is False
                ):
                    nullable = False
                break
        kwargs.update({"allow_none": nullable, "required": not nullable})

    def _should_exclude_field(
        self,
        column: PropertyOrColumn,
        fields: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> bool:
        key = self._get_field_name(column)
        if fields and key not in fields:
            return True
        return bool(exclude and key in exclude)

    def get_base_kwargs(self):
        return {"validate": [], "metadata": {}}


default_converter = ModelConverter()

fields_for_model = default_converter.fields_for_model
property2field = default_converter.property2field
column2field = default_converter.column2field
field_for = default_converter.field_for
