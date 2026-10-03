.. _loaded_only:

Dumping only loaded data
========================

By default, serializing a SQLAlchemy instance accesses every field on the
schema. If an attribute was not loaded from the database -- a deferred column,
an expired attribute, or a relationship that was not eager-loaded -- accessing
it triggers one or more lazy ``SELECT`` statements. For a list of objects this
can easily mean dozens of extra queries (the N+1 problem).

``SQLAlchemySchema`` offers a *loaded-only* dump mode in which attributes are
checked against the ORM instance state *before* they are accessed. Attributes
that would require a query are either reported with an error or omitted, and
the dump itself never emits SQL.

Enabling the mode
-----------------

Set ``dump_loaded_only`` in ``class Meta``::

    class AuthorSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = Author
            dump_loaded_only = True

It can also be enabled per schema instance::

    schema = AuthorSchema(dump_loaded_only=True)

How unloaded attributes are handled is controlled by ``unloaded_attribute``:

- ``"raise"`` (default): raise
  :class:`marshmallow_sqlalchemy.UnloadedAttributeError` as soon as an
  unloaded attribute is encountered. The error message contains the complete
  dotted field path, including nested field names and list indices, for
  example ``"books[0].publisher.name"``.
- ``"omit"``: leave the field out of the serialized output.

The constants ``marshmallow_sqlalchemy.RAISE`` and
``marshmallow_sqlalchemy.OMIT`` can be used instead of the raw strings::

    from marshmallow_sqlalchemy import OMIT

    class AuthorSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = Author
            dump_loaded_only = True
            unloaded_attribute = OMIT

Guarantees
----------

- The check reads the SQLAlchemy ``InstanceState`` (``inspect(instance)``)
  and never calls ``getattr`` first, so no lazy query is triggered and no
  ``DetachedInstanceError`` is raised for instances detached from their
  session.
- A genuinely loaded ``NULL`` is serialized as ``None`` like any other
  loaded value.
- Field ``dump_default`` values and Python-side column defaults are never
  used as substitutes for unloaded database values: in ``"omit"`` mode the
  key is simply absent.
- The strategy selected on the root schema is inherited by nested schemas
  during a dump.

The mode only affects dumping; loading/deserialization is unchanged.
