from __future__ import annotations

import pytest
import sqlalchemy as sa
from marshmallow import fields
from sqlalchemy import event
from sqlalchemy.orm import (
    Mapped,
    declarative_base,
    joinedload,
    load_only,
    mapped_column,
    relationship,
    sessionmaker,
    undefer,
)

from marshmallow_sqlalchemy import (
    OMIT,
    RAISE,
    SQLAlchemyAutoSchema,
    SQLAlchemySchema,
    UnloadedAttributeError,
    auto_field,
)


# ---------------------------------------------------------------------------
# Models with a deferred column, relationships and a Python-side default
# ---------------------------------------------------------------------------


@pytest.fixture
def base():
    return declarative_base()


@pytest.fixture
def models(base):
    class Department(base):
        __tablename__ = "department"
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
        name: Mapped[str | None] = mapped_column(sa.String, nullable=True)

    class Employee(base):
        __tablename__ = "employee"
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
        full_name: Mapped[str] = mapped_column(sa.String, nullable=False)
        # A deferred column: it lives in the table but is not loaded by default.
        bio: Mapped[str] = mapped_column(
            sa.String, nullable=False, deferred=True, default="DEFAULT-BIO"
        )
        department_id: Mapped[int | None] = mapped_column(
            sa.ForeignKey("department.id"), nullable=True
        )
        department = relationship("Department", backref="employees")

    return type("Models", (), {"Department": Department, "Employee": Employee})


@pytest.fixture
def engine(base):
    engine = sa.create_engine("sqlite:///:memory:", future=True)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(models, engine, base):
    base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    session.add(
        models.Employee(
            id=1,
            full_name="Ada Lovelace",
            bio="first programmer",
            department=models.Department(id=10, name=None),
        )
    )
    session.commit()
    session.close()
    return factory


@pytest.fixture
def session(session_factory):
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def query_counter(engine):
    """Count the SQL statements executed against ``engine``."""
    counter = {"count": 0}

    def _tick(*args, **kwargs):
        counter["count"] += 1

    event.listen(engine, "after_cursor_execute", _tick)
    yield counter
    event.remove(engine, "after_cursor_execute", _tick)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@pytest.fixture
def department_schema_cls(models):
    class DepartmentSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = models.Department

    return DepartmentSchema


@pytest.fixture
def schemas(models, department_schema_cls):
    class EmployeeSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = models.Employee
            include_relationships = True
            include_fk = True

        department = fields.Nested(department_schema_cls)

    class RaisingSchema(EmployeeSchema):
        class Meta(EmployeeSchema.Meta):
            dump_loaded_only = True
            unloaded_attribute = RAISE

    class OmittingSchema(EmployeeSchema):
        class Meta(EmployeeSchema.Meta):
            dump_loaded_only = True
            unloaded_attribute = OMIT

    return EmployeeSchema, RaisingSchema, OmittingSchema


@pytest.fixture
def plain_schema(schemas):
    return schemas[0]()


@pytest.fixture
def raising_schema(schemas):
    return schemas[1]()


@pytest.fixture
def omitting_schema(schemas):
    return schemas[2]()


# ---------------------------------------------------------------------------
# Eager loading: no SQL at all, loaded NULL is emitted
# ---------------------------------------------------------------------------


def test_eager_dump_emits_no_sql(session, raising_schema, query_counter):
    model = raising_schema.opts.model
    employee = (
        session.query(model)
        .options(
            joinedload(model.department),
            load_only(
                model.id,
                model.full_name,
                model.bio,
                model.department_id,
            ),
        )
        .one()
    )
    query_counter["count"] = 0

    result = raising_schema.dump(employee)

    assert query_counter["count"] == 0
    assert result == {
        "id": 1,
        "full_name": "Ada Lovelace",
        "bio": "first programmer",
        "department_id": 10,
        "department": {"id": 10, "name": None},
    }


def test_loaded_null_is_emitted(session, models, raising_schema, query_counter):
    employee = (
        session.query(models.Employee)
        .options(joinedload(models.Employee.department), undefer(models.Employee.bio))
        .one()
    )
    query_counter["count"] = 0

    result = raising_schema.dump(employee)

    assert query_counter["count"] == 0
    assert result["department"]["name"] is None


# ---------------------------------------------------------------------------
# Deferred columns and lazy relationships
# ---------------------------------------------------------------------------


def test_deferred_column_raises_with_full_path(
    session, models, raising_schema, query_counter
):
    # Eager-load the relationship so that the deferred `bio` column is the
    # only unloaded attribute.
    employee = (
        session.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .one()
    )
    query_counter["count"] = 0

    # `bio` is configured as deferred on the mapper.
    with pytest.raises(UnloadedAttributeError) as exc_info:
        raising_schema.dump(employee)

    assert exc_info.value.path == "bio"
    assert exc_info.value.attribute == "bio"
    assert exc_info.value.model is models.Employee
    assert "bio" in str(exc_info.value)
    # No query happened before the error, and none on the way out.
    assert query_counter["count"] == 0


def test_deferred_column_is_omitted_not_replaced_by_default(
    session, models, omitting_schema, query_counter
):
    employee = session.get(models.Employee, 1)
    query_counter["count"] = 0

    result = omitting_schema.dump(employee)

    assert query_counter["count"] == 0
    assert "bio" not in result
    # The Python-side column default must never leak into the output.
    assert result.get("bio") != "DEFAULT-BIO"
    assert result["id"] == 1
    assert result["full_name"] == "Ada Lovelace"


def test_lazy_relationship_raises(session, models, department_schema_cls, query_counter):
    class EmployeeSchema(SQLAlchemySchema):
        class Meta:
            model = models.Employee
            dump_loaded_only = True

        id = auto_field()
        department = fields.Nested(department_schema_cls)

    employee = session.get(models.Employee, 1)
    query_counter["count"] = 0

    with pytest.raises(UnloadedAttributeError) as exc_info:
        EmployeeSchema().dump(employee)

    assert exc_info.value.path == "department"
    assert query_counter["count"] == 0


def test_marshmallow_dump_default_not_used_for_unloaded_attribute(
    session, models, query_counter
):
    class EmployeeSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = models.Employee
            dump_loaded_only = True
            unloaded_attribute = OMIT

        bio = auto_field(dump_default="FALLBACK")

    employee = session.get(models.Employee, 1)
    result = EmployeeSchema().dump(employee)

    assert "bio" not in result


# ---------------------------------------------------------------------------
# Expired instances
# ---------------------------------------------------------------------------


def test_expired_instance_raises(session, models, raising_schema, query_counter):
    employee = session.get(models.Employee, 1)
    session.expire(employee)
    query_counter["count"] = 0

    with pytest.raises(UnloadedAttributeError) as exc_info:
        raising_schema.dump(employee)

    # Every attribute is now unloaded; whichever field is visited first
    # reports an unloaded mapped attribute.
    assert exc_info.value.attribute in {
        "id",
        "full_name",
        "bio",
        "department_id",
        "department",
    }
    assert exc_info.value.path == exc_info.value.attribute
    assert "expired" in str(exc_info.value)
    assert query_counter["count"] == 0


def test_expired_nested_attribute_path(
    session, models, raising_schema, query_counter
):
    employee = (
        session.query(models.Employee)
        .options(
            joinedload(models.Employee.department), undefer(models.Employee.bio)
        )
        .one()
    )
    session.expire(employee.department, ["name"])
    query_counter["count"] = 0

    with pytest.raises(UnloadedAttributeError) as exc_info:
        raising_schema.dump(employee)

    assert exc_info.value.path == "department.name"
    assert query_counter["count"] == 0


def test_expired_nested_attribute_omitted(
    session, models, omitting_schema, query_counter
):
    employee = (
        session.query(models.Employee)
        .options(
            joinedload(models.Employee.department), undefer(models.Employee.bio)
        )
        .one()
    )
    session.expire(employee.department, ["name"])
    query_counter["count"] = 0

    result = omitting_schema.dump(employee)

    assert query_counter["count"] == 0
    assert result["department"] == {"id": 10}


# ---------------------------------------------------------------------------
# Detached instances behave the same without a Session
# ---------------------------------------------------------------------------


def test_detached_loaded_instance_dumps(session, models, raising_schema):
    employee = (
        session.query(models.Employee)
        .options(
            joinedload(models.Employee.department),
            undefer(models.Employee.bio),
        )
        .one()
    )
    session.expunge(employee)

    result = raising_schema.dump(employee)

    assert result["id"] == 1
    assert result["bio"] == "first programmer"
    assert result["department"]["name"] is None


def test_detached_expired_instance_raises(
    session, models, raising_schema, query_counter
):
    employee = session.get(models.Employee, 1)
    session.expire(employee)
    session.expunge(employee)
    query_counter["count"] = 0

    with pytest.raises(UnloadedAttributeError) as exc_info:
        raising_schema.dump(employee)

    assert exc_info.value.attribute in {
        "id",
        "full_name",
        "bio",
        "department_id",
        "department",
    }
    assert "detached" in str(exc_info.value)
    assert query_counter["count"] == 0


def test_detached_expired_omitted(session, models, omitting_schema, query_counter):
    employee = session.get(models.Employee, 1)
    session.expire(employee)
    session.expunge(employee)

    assert omitting_schema.dump(employee) == {}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def test_many_path_includes_index(session, models, raising_schema):
    session.add(
        models.Employee(
            id=2,
            full_name="Grace Hopper",
            bio="compiler",
            department_id=None,
        )
    )
    session.commit()
    employees = (
        session.query(models.Employee)
        .options(
            joinedload(models.Employee.department), undefer(models.Employee.bio)
        )
        .order_by(models.Employee.id)
        .all()
    )
    # Only the second row has an unloaded attribute.
    session.expire(employees[1], ["full_name"])

    with pytest.raises(UnloadedAttributeError) as exc_info:
        raising_schema.dump(employees, many=True)

    assert exc_info.value.path == "[1].full_name"


def test_many_omit_emits_no_sql(
    session, models, omitting_schema, query_counter
):
    employees = (
        session.query(models.Employee)
        .options(undefer(models.Employee.bio))
        .all()
    )
    session.expire(employees[0], ["full_name"])
    query_counter["count"] = 0

    result = omitting_schema.dump(employees, many=True)

    assert query_counter["count"] == 0
    assert len(result) == 1
    assert "full_name" not in result[0]
    assert result[0]["bio"] == "first programmer"


# ---------------------------------------------------------------------------
# Strategy inheritance and configuration overrides
# ---------------------------------------------------------------------------


def test_nested_schema_inherits_omit_from_root(
    session, models, plain_schema, query_counter
):
    employee = (
        session.query(models.Employee)
        .options(
            joinedload(models.Employee.department), undefer(models.Employee.bio)
        )
        .one()
    )
    session.expire(employee.department, ["name"])
    query_counter["count"] = 0

    result = type(plain_schema)(
        dump_loaded_only=True, unloaded_attribute=OMIT
    ).dump(employee)

    assert query_counter["count"] == 0
    assert result["department"] == {"id": 10}


def test_constructor_override_enables_mode(
    session, models, plain_schema, query_counter
):
    employee = session.get(models.Employee, 1)
    query_counter["count"] = 0

    with pytest.raises(UnloadedAttributeError):
        type(plain_schema)(dump_loaded_only=True).dump(employee)

    result = type(plain_schema)(
        dump_loaded_only=True, unloaded_attribute=OMIT
    ).dump(employee)
    assert query_counter["count"] == 0
    assert "bio" not in result


def test_invalid_strategy_meta_raises(models):
    with pytest.raises(ValueError, match="unloaded_attribute"):

        class BadSchema(SQLAlchemyAutoSchema):
            class Meta:
                model = models.Employee
                dump_loaded_only = True
                unloaded_attribute = "explode"


def test_invalid_strategy_constructor_raises(models):
    class EmployeeSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = models.Employee

    with pytest.raises(ValueError, match="unloaded_attribute"):
        EmployeeSchema(dump_loaded_only=True, unloaded_attribute="explode")


# ---------------------------------------------------------------------------
# Non-ORM objects and mode disabled keep marshmallow behavior
# ---------------------------------------------------------------------------


def test_non_mapped_objects_fall_through(raising_schema):
    assert raising_schema.dump(
        {
            "id": 7,
            "full_name": "Dict Duck",
            "bio": "typed",
            "department_id": None,
            "department": None,
        }
    )["full_name"] == "Dict Duck"


def test_none_input(raising_schema):
    assert raising_schema.dump(None) == {}


def test_mode_disabled_still_lazy_loads(
    session, models, plain_schema, query_counter
):
    employee = session.get(models.Employee, 1)
    query_counter["count"] = 0

    result = plain_schema.dump(employee)

    # Baseline marshmallow-sqlalchemy behavior: deferred/lazy access emits SQL.
    assert query_counter["count"] > 0
    assert result["bio"] == "first programmer"
    assert result["department"] == {"id": 10, "name": None}


def test_excluded_unloaded_field_does_not_raise(
    session, models, raising_schema, query_counter
):
    # `bio` is deferred and the relationship is lazy; excluding both leaves
    # only loaded columns.
    schema = type(raising_schema)(exclude=("bio", "department"))
    employee = session.get(models.Employee, 1)
    query_counter["count"] = 0

    result = schema.dump(employee)

    assert query_counter["count"] == 0
    assert result == {"id": 1, "full_name": "Ada Lovelace", "department_id": 10}


def test_only_loaded_fields_does_not_raise(
    session, models, raising_schema, query_counter
):
    schema = type(raising_schema)(only=("id", "full_name"))
    employee = session.get(models.Employee, 1)
    query_counter["count"] = 0

    result = schema.dump(employee)

    assert query_counter["count"] == 0
    assert result == {"id": 1, "full_name": "Ada Lovelace"}
