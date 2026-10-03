from __future__ import annotations

from importlib.metadata import version

import marshmallow
import pytest
import sqlalchemy as sa
from marshmallow import Schema, ValidationError, fields, validate
from pytest_lazy_fixtures import lf

import marshmallow_sqlalchemy
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, SQLAlchemySchema, auto_field
from marshmallow_sqlalchemy.exceptions import IncorrectSchemaTypeError
from marshmallow_sqlalchemy.fields import Related

# -----------------------------------------------------------------------------


@pytest.fixture
def teacher(models, session):
    school = models.School(id=42, name="Univ. Of Whales")
    teacher_ = models.Teacher(
        id=24, full_name="Teachy McTeachFace", current_school=school
    )
    session.add(teacher_)
    session.flush()
    return teacher_


@pytest.fixture
def school(models, session):
    school = models.School(id=42, name="Univ. Of Whales")
    students = [
        models.Student(id=35, full_name="Bob Smith", current_school=school),
        models.Student(id=53, full_name="John Johnson", current_school=school),
    ]

    session.add_all(students)
    session.flush()
    return school


class EntityMixin:
    id = auto_field(dump_only=True)


# Auto schemas with default options


@pytest.fixture
def sqla_auto_model_schema(models, request) -> SQLAlchemyAutoSchema:
    class TeacherSchema(SQLAlchemyAutoSchema[models.Teacher]):
        class Meta:
            model = models.Teacher

        full_name = auto_field(validate=validate.Length(max=20))

    return TeacherSchema()


@pytest.fixture
def sqla_auto_table_schema(models, request) -> SQLAlchemyAutoSchema:
    class TeacherSchema(SQLAlchemyAutoSchema):
        class Meta:
            table = models.Teacher.__table__

        full_name = auto_field(validate=validate.Length(max=20))

    return TeacherSchema()


# Schemas with relationships


@pytest.fixture
def sqla_schema_with_relationships(models, request) -> SQLAlchemySchema:
    class TeacherSchema(EntityMixin, SQLAlchemySchema[models.Teacher]):
        class Meta:
            model = models.Teacher

        full_name = auto_field(validate=validate.Length(max=20))
        current_school = auto_field()
        substitute = auto_field()
        data = auto_field()

    return TeacherSchema()


@pytest.fixture
def sqla_auto_model_schema_with_relationships(models, request) -> SQLAlchemyAutoSchema:
    class TeacherSchema(SQLAlchemyAutoSchema[models.Teacher]):
        class Meta:
            model = models.Teacher
            include_relationships = True

        full_name = auto_field(validate=validate.Length(max=20))

    return TeacherSchema()


# Schemas with foreign keys


@pytest.fixture
def sqla_schema_with_fks(models, request) -> SQLAlchemySchema:
    class TeacherSchema(EntityMixin, SQLAlchemySchema[models.Teacher]):
        class Meta:
            model = models.Teacher

        full_name = auto_field(validate=validate.Length(max=20))
        current_school_id = auto_field()
        data = auto_field()

    return TeacherSchema()


@pytest.fixture
def sqla_auto_model_schema_with_fks(models, request) -> SQLAlchemyAutoSchema:
    class TeacherSchema(SQLAlchemyAutoSchema[models.Teacher]):
        class Meta:
            model = models.Teacher
            include_fk = True
            include_relationships = False

        full_name = auto_field(validate=validate.Length(max=20))

    return TeacherSchema()


# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema",
    (
        lf("sqla_schema_with_relationships"),
        lf("sqla_auto_model_schema_with_relationships"),
    ),
)
def test_dump_with_relationships(teacher, schema):
    assert schema.dump(teacher) == {
        "id": teacher.id,
        "full_name": teacher.full_name,
        "current_school": 42,
        "substitute": None,
        "data": None,
    }


@pytest.mark.parametrize(
    "schema",
    (
        lf("sqla_schema_with_fks"),
        lf("sqla_auto_model_schema_with_fks"),
    ),
)
def test_dump_with_foreign_keys(teacher, schema):
    assert schema.dump(teacher) == {
        "id": teacher.id,
        "full_name": teacher.full_name,
        "current_school_id": 42,
        "data": None,
    }


def test_table_schema_dump(teacher, sqla_auto_table_schema):
    assert sqla_auto_table_schema.dump(teacher) == {
        "id": teacher.id,
        "full_name": teacher.full_name,
        "data": None,
    }


@pytest.mark.parametrize(
    "schema",
    (
        lf("sqla_schema_with_relationships"),
        lf("sqla_schema_with_fks"),
        lf("sqla_auto_model_schema"),
        lf("sqla_auto_table_schema"),
    ),
)
def test_load(schema):
    assert schema.load({"full_name": "Teachy T"}) == {"full_name": "Teachy T"}


class TestLoadInstancePerSchemaInstance:
    @pytest.fixture
    def schema_no_load_instance(self, models, session):
        class TeacherSchema(SQLAlchemySchema[models.Teacher]):  # type: ignore[name-defined]
            class Meta:
                model = models.Teacher
                sqla_session = session
                # load_instance = False is the default

            full_name = auto_field(validate=validate.Length(max=20))
            current_school = auto_field()
            substitute = auto_field()

        return TeacherSchema

    @pytest.fixture
    def schema_with_load_instance(self, schema_no_load_instance: type):
        class TeacherSchema(schema_no_load_instance):
            class Meta(schema_no_load_instance.Meta):  # type: ignore[name-defined]
                load_instance = True

        return TeacherSchema

    @pytest.fixture
    def auto_schema_no_load_instance(self, models, session):
        class TeacherSchema(SQLAlchemyAutoSchema[models.Teacher]):  # type: ignore[name-defined]
            class Meta:
                model = models.Teacher
                sqla_session = session
                # load_instance = False is the default

        return TeacherSchema

    @pytest.fixture
    def auto_schema_with_load_instance(self, auto_schema_no_load_instance: type):
        class TeacherSchema(auto_schema_no_load_instance):
            class Meta(auto_schema_no_load_instance.Meta):  # type: ignore[name-defined]
                load_instance = True

        return TeacherSchema

    @pytest.mark.parametrize(
        "Schema",
        (
            lf("schema_no_load_instance"),
            lf("schema_with_load_instance"),
            lf("auto_schema_no_load_instance"),
            lf("auto_schema_with_load_instance"),
        ),
    )
    def test_toggle_load_instance_per_schema(self, models, Schema):
        tname = "Teachy T"
        source = {"full_name": tname}

        # No per-instance override
        load_instance_default = Schema()
        result = load_instance_default.load(source)
        default = load_instance_default.opts.load_instance

        default_type = models.Teacher if default else dict
        assert isinstance(result, default_type)

        # Override the default
        override = Schema(load_instance=not default)
        result = override.load(source)

        override_type = dict if default else models.Teacher
        assert isinstance(result, override_type)


@pytest.mark.parametrize(
    "schema",
    (
        lf("sqla_schema_with_relationships"),
        lf("sqla_schema_with_fks"),
        lf("sqla_auto_model_schema"),
        lf("sqla_auto_table_schema"),
    ),
)
def test_load_validation_errors(schema):
    with pytest.raises(ValidationError):
        schema.load({"full_name": "x" * 21})


def test_auto_field_on_plain_schema_raises_error():
    class BadSchema(Schema):
        name = auto_field()

    with pytest.raises(IncorrectSchemaTypeError):
        BadSchema()


def test_cannot_set_both_model_and_table(models):
    with pytest.raises(ValueError, match="Cannot set both"):

        class BadWidgetSchema(SQLAlchemySchema):
            class Meta:
                model = models.Teacher
                table = models.Teacher


def test_passing_model_to_auto_field(models, teacher):
    class TeacherSchema(SQLAlchemySchema):
        current_school_id = auto_field(model=models.Teacher)

    schema = TeacherSchema()
    assert schema.dump(teacher) == {"current_school_id": teacher.current_school_id}


def test_passing_table_to_auto_field(models, teacher):
    class TeacherSchema(SQLAlchemySchema):
        current_school_id = auto_field(table=models.Teacher.__table__)

    schema = TeacherSchema()
    assert schema.dump(teacher) == {"current_school_id": teacher.current_school_id}


# https://github.com/marshmallow-code/marshmallow-sqlalchemy/issues/190
def test_auto_schema_skips_synonyms(models):
    class TeacherSchema(SQLAlchemyAutoSchema[models.Teacher]):  # type: ignore[name-defined]
        class Meta:
            model = models.Teacher
            include_fk = True

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields
    assert "curr_school_id" not in schema.fields


def test_auto_field_works_with_synonym(models):
    class TeacherSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = models.Teacher
            include_fk = True

        curr_school_id = auto_field()

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields
    assert "curr_school_id" in schema.fields


# Regresion test https://github.com/marshmallow-code/marshmallow-sqlalchemy/issues/306
def test_auto_field_works_with_ordered_flag(models):
    class StudentSchema(SQLAlchemyAutoSchema[models.Student]):  # type: ignore[name-defined]
        class Meta:
            model = models.Student
            ordered = True

        full_name = auto_field()

    schema = StudentSchema()
    # Declared fields precede auto-generated fields
    assert tuple(schema.fields.keys()) == (
        "full_name",
        "course_count",
        "id",
        "dob",
        "date_created",
    )


class TestAliasing:
    @pytest.fixture
    def aliased_schema(self, models):
        class TeacherSchema(SQLAlchemySchema):
            class Meta:
                model = models.Teacher

            # Generate field from "full_name", pull from "full_name" attribute, output to "name"
            name = auto_field("full_name")

        return TeacherSchema()

    @pytest.fixture
    def aliased_auto_schema(self, models):
        class TeacherSchema(SQLAlchemyAutoSchema):
            class Meta:
                model = models.Teacher
                exclude = ("full_name",)

            # Generate field from "full_name", pull from "full_name" attribute, output to "name"
            name = auto_field("full_name")

        return TeacherSchema()

    @pytest.fixture
    def aliased_attribute_schema(self, models):
        class TeacherSchema(SQLAlchemySchema):
            class Meta:
                model = models.Teacher

            # Generate field from "full_name", pull from "fname" attribute, output to "name"
            name = auto_field("full_name", attribute="fname")

        return TeacherSchema()

    @pytest.mark.parametrize(
        "schema",
        (
            lf("aliased_schema"),
            lf("aliased_auto_schema"),
        ),
    )
    def test_passing_column_name(self, schema, teacher):
        assert schema.fields["name"].attribute == "full_name"
        dumped = schema.dump(teacher)
        assert dumped["name"] == teacher.full_name

    def test_passing_column_name_and_attribute(self, teacher, aliased_attribute_schema):
        assert aliased_attribute_schema.fields["name"].attribute == "fname"
        dumped = aliased_attribute_schema.dump(teacher)
        assert dumped["name"] == teacher.fname


class TestModelInstanceDeserialization:
    @pytest.fixture
    def sqla_schema_class(self, models, session):
        class TeacherSchema(SQLAlchemySchema):
            class Meta:
                model = models.Teacher
                load_instance = True
                sqla_session = session

            full_name = auto_field(validate=validate.Length(max=20))
            current_school = auto_field()
            substitute = auto_field()

        return TeacherSchema

    @pytest.fixture
    def sqla_auto_schema_class(self, models, session):
        class TeacherSchema(SQLAlchemyAutoSchema):
            class Meta:
                model = models.Teacher
                include_relationships = True
                load_instance = True
                sqla_session = session

        return TeacherSchema

    @pytest.mark.parametrize(
        "SchemaClass",
        (
            lf("sqla_schema_class"),
            lf("sqla_auto_schema_class"),
        ),
    )
    def test_load(self, teacher, SchemaClass, models):
        schema = SchemaClass(unknown=marshmallow.INCLUDE)
        dump_data = schema.dump(teacher)
        load_data = schema.load(dump_data)

        assert isinstance(load_data, models.Teacher)

    def test_load_transient(self, models, teacher):
        class TeacherSchema(SQLAlchemyAutoSchema):
            class Meta:
                model = models.Teacher
                load_instance = True
                transient = True

        schema = TeacherSchema()
        dump_data = schema.dump(teacher)
        load_data = schema.load(dump_data)
        assert isinstance(load_data, models.Teacher)
        state = sa.inspect(load_data)
        assert state.transient

    def test_override_transient(self, models, teacher):
        # marshmallow-code/marshmallow-sqlalchemy#388
        class TeacherSchema(SQLAlchemyAutoSchema):
            class Meta:
                model = models.Teacher
                load_instance = True
                transient = True

        schema = TeacherSchema(transient=False)
        assert schema.transient is False


def test_related_when_model_attribute_name_distinct_from_column_name(
    models,
    session,
    teacher,
):
    class TeacherSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = models.Teacher
            load_instance = True
            sqla_session = session

        current_school = Related(["id", "name"])

    dump_data = TeacherSchema().dump(teacher)
    assert "school_id" not in dump_data["current_school"]
    assert dump_data["current_school"]["id"] == teacher.current_school.id
    assert dump_data["current_school"]["name"] == teacher.current_school.name
    new_teacher = TeacherSchema().load(dump_data, transient=True)
    assert new_teacher.current_school.id == teacher.current_school.id
    assert TeacherSchema().load(dump_data) is teacher


# https://github.com/marshmallow-code/marshmallow-sqlalchemy/issues/338
def test_auto_field_works_with_assoc_proxy(models):
    class StudentSchema(SQLAlchemySchema):
        class Meta:
            model = models.Student

        possible_teachers = auto_field()

    schema = StudentSchema()
    assert "possible_teachers" in schema.fields


def test_dump_and_load_with_assoc_proxy_multiplicity(models, session, school):
    class SchoolSchema(SQLAlchemySchema):
        class Meta:
            model = models.School
            load_instance = True
            sqla_session = session

        student_ids = auto_field()

    schema = SchoolSchema()
    assert "student_ids" in schema.fields
    dump_data = schema.dump(school)
    assert "student_ids" in dump_data
    assert dump_data["student_ids"] == list(school.student_ids)
    new_school = schema.load(dump_data, transient=True)
    assert list(new_school.student_ids) == list(school.student_ids)


def test_dump_and_load_with_assoc_proxy_multiplicity_dump_only_kwargs(
    models, session, school
):
    class SchoolSchema(SQLAlchemySchema):
        class Meta:
            model = models.School
            load_instance = True
            sqla_session = session

        student_ids = auto_field(dump_only=True, data_key="student_identifiers")

    schema = SchoolSchema()
    assert "student_ids" in schema.fields
    assert schema.fields["student_ids"] not in schema.load_fields.values()
    assert schema.fields["student_ids"] in schema.dump_fields.values()

    dump_data = schema.dump(school)
    assert "student_ids" not in dump_data
    assert "student_identifiers" in dump_data
    assert dump_data["student_identifiers"] == list(school.student_ids)

    with pytest.raises(ValidationError):
        schema.load(dump_data, transient=True)


def test_dump_and_load_with_assoc_proxy_multiplicity_load_only_only_kwargs(
    models, session, school
):
    class SchoolSchema(SQLAlchemySchema):
        class Meta:
            model = models.School
            load_instance = True
            sqla_session = session

        student_ids = auto_field(load_only=True, data_key="student_identifiers")

    schema = SchoolSchema()

    assert "student_ids" in schema.fields
    assert schema.fields["student_ids"] not in schema.dump_fields.values()
    assert schema.fields["student_ids"] in schema.load_fields.values()

    dump_data = schema.dump(school)
    assert "student_identifers" not in dump_data

    new_school = schema.load(
        {"student_identifiers": list(school.student_ids)}, transient=True
    )
    assert list(new_school.student_ids) == list(school.student_ids)


# https://github.com/marshmallow-code/marshmallow-sqlalchemy/issues/440
def test_auto_schema_with_model_allows_subclasses_to_override_include_fk(models):
    class TeacherSchema(SQLAlchemyAutoSchema):
        inherited_field = fields.String()

        class Meta:
            model = models.Teacher
            include_fk = True

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields

    class TeacherNoFkSchema(TeacherSchema):
        class Meta(TeacherSchema.Meta):
            include_fk = False

    schema2 = TeacherNoFkSchema()
    assert "id" in schema2.fields
    assert "inherited_field" in schema2.fields
    assert "current_school_id" not in schema2.fields


def test_auto_schema_with_model_allows_subclasses_to_override_exclude(models):
    class TeacherSchema(SQLAlchemyAutoSchema):
        inherited_field = fields.String()

        class Meta:
            model = models.Teacher
            include_fk = True

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields

    class TeacherNoFkSchema(TeacherSchema):
        class Meta(TeacherSchema.Meta):
            exclude = ("current_school_id",)

    schema2 = TeacherNoFkSchema()
    assert "id" in schema2.fields
    assert "inherited_field" in schema2.fields
    assert "current_school_id" not in schema2.fields


def test_auto_schema_with_model_allows_subclasses_to_override_include_fk_with_explicit_field(
    models,
):
    class TeacherSchema(SQLAlchemyAutoSchema):
        inherited_field = fields.String()

        class Meta:
            model = models.Teacher
            include_fk = True

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields

    class TeacherNoFkSchema(TeacherSchema):
        current_school_id = fields.Integer()

        class Meta(TeacherSchema.Meta):
            include_fk = False

    schema2 = TeacherNoFkSchema()
    assert "id" in schema2.fields
    assert "inherited_field" in schema2.fields
    assert "current_school_id" in schema2.fields


def test_auto_schema_with_table_allows_subclasses_to_override_include_fk(models):
    class TeacherSchema(SQLAlchemyAutoSchema):
        inherited_field = fields.Integer()

        class Meta:
            table = models.Teacher.__table__
            include_fk = True

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields

    class TeacherNoFkSchema(TeacherSchema):
        class Meta(TeacherSchema.Meta):
            include_fk = False

    schema2 = TeacherNoFkSchema()
    assert "id" in schema2.fields
    assert "inherited_field" in schema2.fields
    assert "current_school_id" not in schema2.fields


def test_auto_schema_with_table_allows_subclasses_to_override_include_fk_with_explicit_field(
    models,
):
    class TeacherSchema(SQLAlchemyAutoSchema):
        inherited_field = fields.Integer()

        class Meta:
            table = models.Teacher.__table__
            include_fk = True

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields

    class TeacherNoFkModelSchema(TeacherSchema):
        current_school_id = fields.Integer()

        class Meta(TeacherSchema.Meta):
            include_fk = False

    schema2 = TeacherNoFkModelSchema()
    assert "id" in schema2.fields
    assert "inherited_field" in schema2.fields
    assert "current_school_id" in schema2.fields


def test_auto_schema_with_model_can_inherit_declared_field_for_foreign_key_column_when_include_fk_is_false(
    models,
):
    class BaseTeacherSchema(Schema):
        current_school_id = fields.Integer()

    class TeacherSchema(BaseTeacherSchema, SQLAlchemyAutoSchema):
        class Meta:
            model = models.Teacher
            include_fk = False

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields


def test_auto_schema_with_table_can_inherit_declared_field_for_foreign_key_column_when_include_fk_is_false(
    models,
):
    class BaseTeacherSchema(Schema):
        current_school_id = fields.Integer()

    class TeacherSchema(BaseTeacherSchema, SQLAlchemyAutoSchema):
        class Meta:
            table = models.Teacher.__table__
            include_fk = False

    schema = TeacherSchema()
    assert "current_school_id" in schema.fields


def test_auto_field_does_not_accept_arbitrary_kwargs(models):
    if int(version("marshmallow")[0]) < 4:
        from marshmallow.warnings import RemovedInMarshmallow4Warning  # noqa: PLC0415

        with pytest.warns(
            RemovedInMarshmallow4Warning,
            match="Passing field metadata as keyword arguments is deprecated",
        ):

            class CourseSchema(SQLAlchemyAutoSchema):
                class Meta:
                    model = models.Course

                name = auto_field(description="A course name")

    else:
        with pytest.raises(TypeError, match="unexpected keyword argument"):

            class CourseSchema(SQLAlchemyAutoSchema):  # type: ignore[no-redef]
                class Meta:
                    model = models.Course

                name = auto_field(description="A course name")


# https://github.com/marshmallow-code/marshmallow-sqlalchemy/issues/394
def test_dumping_pickle_field(models, teacher):
    class TeacherSchema(SQLAlchemySchema):
        class Meta:
            model = models.Teacher

        data = auto_field()

    teacher.data = {"foo": "bar"}

    schema = TeacherSchema()
    assert schema.dump(teacher) == {
        "data": {"foo": "bar"},
    }


# ---------------------------------------------------------------------------
# Two-phase loading (validated update plans)
# ---------------------------------------------------------------------------


@pytest.fixture
def nested_data(models, session):
    school_a = models.School(id=1, name="School A")
    school_b = models.School(id=2, name="School B")
    student_a = models.Student(
        id=10, full_name="Alice", dob=None, current_school=school_a
    )
    student_b = models.Student(
        id=11, full_name="Bob", dob=None, current_school=school_b
    )
    session.add_all([school_a, school_b, student_a, student_b])
    session.commit()
    return {
        "schools": (school_a, school_b),
        "students": (student_a, student_b),
    }


@pytest.fixture
def nested_schemas(models, session):
    hook_calls = {"school": 0, "student": 0}

    class SchoolSchema(SQLAlchemyAutoSchema[models.School]):
        class Meta:
            model = models.School
            load_instance = True
            sqla_session = session

        name = auto_field(validate=validate.Length(max=20))

        @marshmallow.post_load
        def count_hook(self, data, **kwargs):
            hook_calls["school"] += 1
            return data

    class StudentSchema(SQLAlchemyAutoSchema[models.Student]):
        class Meta:
            model = models.Student
            load_instance = True
            include_relationships = True
            sqla_session = session

        full_name = auto_field(validate=validate.Length(max=20))
        current_school = marshmallow.fields.Nested(SchoolSchema)

        @marshmallow.post_load
        def count_hook(self, data, **kwargs):
            hook_calls["student"] += 1
            return data

    schema = StudentSchema()
    return schema, SchoolSchema, hook_calls


def _listen_flushes(session):
    flushes = []

    def after_flush(sess, ctx):
        flushes.append(len(sess.new) + len(sess.dirty))

    sa.event.listen(session, "after_flush", after_flush)
    return flushes


def test_two_phase_failed_validation_leaves_objects_untouched(
    models, session, nested_data, nested_schemas
):
    schema, _, _ = nested_schemas
    school_a, school_b = nested_data["schools"]
    student_a, student_b = nested_data["students"]
    flushes = _listen_flushes(session)

    payload = [
        {
            "id": 10,
            "full_name": "Alice X",
            "current_school": {"id": 1, "name": "School A+1"},
        },
        {
            "id": 11,
            "full_name": "Bob X",
            "current_school": {"id": 2, "name": "School B+1"},
        },
        # last record fails field validation
        {
            "id": 10,
            "full_name": "x" * 21,
            "current_school": {"id": 1, "name": "Never Applied"},
        },
    ]

    with pytest.raises(ValidationError) as exc_info:
        schema.load(payload, many=True, two_phase=True)
    assert 2 in exc_info.value.messages

    # scalars unchanged
    assert student_a.full_name == "Alice"
    assert student_b.full_name == "Bob"
    assert school_a.name == "School A"
    assert school_b.name == "School B"
    # relationships unchanged (same identities)
    assert student_a.current_school is school_a
    assert student_b.current_school is school_b
    # nothing dirty and nothing flushed
    assert len(session.dirty) == 0
    assert len(session.new) == 0
    assert flushes == []


def test_two_phase_success_applies_and_returns_identity_map_objects(
    models, session, nested_data, nested_schemas
):
    schema, _, hook_calls = nested_schemas
    school_a, school_b = nested_data["schools"]
    student_a, student_b = nested_data["students"]

    payload = [
        {
            "id": 10,
            "full_name": "Alice 2",
            "current_school": {"id": 1, "name": "School A 2"},
        },
        {
            "id": 11,
            "full_name": "Bob 2",
            "current_school": {"id": 2, "name": "School B 2"},
        },
    ]
    results = schema.load(payload, many=True, two_phase=True)

    assert results == [student_a, student_b]
    assert results[0] is student_a
    assert results[1] is student_b
    assert student_a.full_name == "Alice 2"
    assert student_b.full_name == "Bob 2"
    assert student_a.current_school is school_a
    assert school_a.name == "School A 2"
    assert student_b.current_school is school_b
    assert school_b.name == "School B 2"
    # hooks ran exactly once per record (including nested)
    assert hook_calls == {"school": 2, "student": 2}


def test_two_phase_retry_does_not_run_hooks_twice(
    models, session, nested_data, nested_schemas
):
    schema, _, hook_calls = nested_schemas

    bad_payload = [
        {
            "id": 10,
            "full_name": "Alice X",
            "current_school": {"id": 1, "name": "Tmp"},
        },
        {
            "id": 11,
            "full_name": "x" * 21,
            "current_school": {"id": 2, "name": "Tmp"},
        },
    ]
    with pytest.raises(ValidationError):
        schema.load(bad_payload, many=True, two_phase=True)

    hook_calls["school"] = 0
    hook_calls["student"] = 0
    good_payload = [
        {
            "id": 10,
            "full_name": "Alice R",
            "current_school": {"id": 1, "name": "School A R"},
        },
        {
            "id": 11,
            "full_name": "Bob R",
            "current_school": {"id": 2, "name": "School B R"},
        },
    ]
    schema.load(good_payload, many=True, two_phase=True)
    # successful run alone executed every hook exactly once
    assert hook_calls == {"school": 2, "student": 2}


def test_two_phase_does_not_roll_back_unrelated_session_work(
    models, session, nested_data, nested_schemas
):
    schema, _, _ = nested_schemas
    school_a, _ = nested_data["schools"]
    student_a, student_b = nested_data["students"]

    scratch = models.Student(
        id=77, full_name="Scratch", dob=None, current_school_id=school_a.id
    )
    session.add(scratch)
    session.flush()
    scratch_id = id(scratch)

    with pytest.raises(ValidationError):
        schema.load(
            [
                {
                    "id": 10,
                    "full_name": "ok",
                    "current_school": {"id": 1, "name": "x"},
                },
                {
                    "id": 11,
                    "full_name": "x" * 21,
                    "current_school": {"id": 1, "name": "y"},
                },
            ],
            many=True,
            two_phase=True,
        )

    assert session.get(models.Student, 77) is scratch
    assert id(session.get(models.Student, 77)) == scratch_id
    assert student_a.full_name == "Alice"
    assert student_b.full_name == "Bob"


def test_two_phase_duplicate_target_raises_by_default_and_changes_nothing(
    models, session, nested_data, nested_schemas
):
    schema, _, _ = nested_schemas
    school_a, _ = nested_data["schools"]
    student_a, _ = nested_data["students"]

    payload = [
        {
            "id": 10,
            "full_name": "First",
            "current_school": {"id": 1, "name": "S First"},
        },
        {
            "id": 10,
            "full_name": "Second",
            "current_school": {"id": 1, "name": "S Second"},
        },
    ]
    with pytest.raises(marshmallow_sqlalchemy.DuplicateTargetError) as exc_info:
        schema.load(payload, many=True, two_phase=True)
    # it is still catchable as a plain marshmallow ValidationError
    assert isinstance(exc_info.value, ValidationError)
    assert exc_info.value.model is models.Student
    assert exc_info.value.identity == (10,)

    assert student_a.full_name == "Alice"
    assert school_a.name == "School A"
    assert len(session.dirty) == 0


def test_two_phase_duplicate_policies(models, session, nested_data, nested_schemas):
    schema, _, _ = nested_schemas
    school_a, _ = nested_data["schools"]
    student_a, _ = nested_data["students"]

    def payload():
        return [
            {
                "id": 10,
                "full_name": "First",
                "current_school": {"id": 1, "name": "S First"},
            },
            {
                "id": 10,
                "full_name": "Second",
                "current_school": {"id": 1, "name": "S Second"},
            },
        ]

    results = schema.load(payload(), many=True, two_phase=True, on_duplicate="first")
    assert results == [student_a]
    assert student_a.full_name == "First"
    assert school_a.name == "S First"
    session.rollback()

    results = schema.load(payload(), many=True, two_phase=True, on_duplicate="last")
    assert results == [student_a]
    assert student_a.full_name == "Second"
    assert school_a.name == "S Second"
    session.rollback()

    results = schema.load(
        [
            {
                "id": 10,
                "full_name": "Merged",
                "current_school": {"id": 1, "name": "S Merged"},
            },
            {
                "id": 10,
                "full_name": "Merged Again",
                "current_school": {"id": 1, "name": "S Merged"},
            },
        ],
        many=True,
        two_phase=True,
        on_duplicate="merge",
    )
    assert results == [student_a]
    assert student_a.full_name == "Merged Again"
    assert school_a.name == "S Merged"
    session.rollback()


def test_two_phase_shared_nested_target_is_not_a_duplicate(
    models, session, nested_data, nested_schemas
):
    schema, _, _ = nested_schemas
    school_a, _ = nested_data["schools"]
    student_a, student_b = nested_data["students"]

    results = schema.load(
        [
            {
                "id": 10,
                "full_name": "Alice S",
                "current_school": {"id": 1, "name": "Shared"},
            },
            {
                "id": 11,
                "full_name": "Bob S",
                "current_school": {"id": 1, "name": "Shared"},
            },
        ],
        many=True,
        two_phase=True,
    )
    assert results == [student_a, student_b]
    assert student_a.current_school is school_a
    assert student_b.current_school is school_a
    assert school_a.name == "Shared"


def test_make_update_plan_is_applied_lazily_and_is_idempotent(
    models, session, nested_data, nested_schemas
):
    schema, _, _ = nested_schemas
    _, school_b = nested_data["schools"]
    student_a, student_b = nested_data["students"]

    plan = schema.make_update_plan(
        [
            {
                "id": 11,
                "full_name": "Bob P",
                "current_school": {"id": 2, "name": "School B P"},
            }
        ],
        many=True,
    )
    # building the plan changes nothing
    assert student_b.full_name == "Bob"
    assert school_b.name == "School B"
    assert len(session.dirty) == 0

    first = plan.apply()
    second = plan.apply()
    assert first == second == [student_b]
    assert first[0] is student_b


def test_two_phase_single_record_with_instance(
    models, session, nested_data, nested_schemas
):
    schema, _, _ = nested_schemas
    school_a, _ = nested_data["schools"]
    student_a, _ = nested_data["students"]

    result = schema.load(
        {
            "id": 10,
            "full_name": "Solo",
            "current_school": {"id": 1, "name": "Solo School"},
        },
        two_phase=True,
        instance=student_a,
    )
    assert result is student_a
    assert student_a.full_name == "Solo"
    assert student_a.current_school is school_a
    assert school_a.name == "Solo School"


def test_two_phase_invalid_on_duplicate_policy(models, session, nested_schemas):
    schema, _, _ = nested_schemas
    with pytest.raises(ValueError, match="on_duplicate"):
        schema.load([{"id": 1}], many=True, two_phase=True, on_duplicate="bogus")


def test_two_phase_requires_load_instance(models, session):
    class PlainSchema(SQLAlchemyAutoSchema[models.Teacher]):
        class Meta:
            model = models.Teacher
            sqla_session = session

    with pytest.raises(ValueError, match="load_instance"):
        PlainSchema().load([{"full_name": "x"}], many=True, two_phase=True)
    with pytest.raises(ValueError, match="load_instance"):
        PlainSchema().make_update_plan([{"full_name": "x"}], many=True)


def test_two_phase_failure_inside_nested_many_changes_nothing(
    models, session
):
    lecture = models.Lecture(id=3, topic="Old topic")
    kw_one = models.Keyword(id=1, keyword="one")
    kw_two = models.Keyword(id=2, keyword="two")
    session.add_all([lecture, kw_one, kw_two])
    session.flush()
    lecture.kw = [kw_one, kw_two]
    session.commit()

    class KeywordSchema(SQLAlchemyAutoSchema[models.Keyword]):
        class Meta:
            model = models.Keyword
            load_instance = True
            sqla_session = session

        keyword = auto_field(validate=validate.Length(max=10))

    class LectureSchema(SQLAlchemyAutoSchema[models.Lecture]):
        class Meta:
            model = models.Lecture
            load_instance = True
            include_relationships = True
            sqla_session = session

        kw = marshmallow.fields.Nested(KeywordSchema, many=True)

    schema = LectureSchema()
    payload = [
        {
            "id": 3,
            "topic": "New topic",
            "kw": [
                {"id": 1, "keyword": "updated one"},
                {"id": 2, "keyword": "x" * 11},  # fails validation
            ],
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        schema.load(payload, many=True, two_phase=True)
    assert "kw" in exc_info.value.messages[0]

    assert lecture.topic == "Old topic"
    assert kw_one.keyword == "one"
    assert kw_two.keyword == "two"
    assert len(session.dirty) == 0


def test_two_phase_transient_creates_objects_on_apply(models):
    class TeacherSchema(SQLAlchemyAutoSchema[models.Teacher]):
        class Meta:
            model = models.Teacher
            load_instance = True
            transient = True

    schema = TeacherSchema()
    results = schema.load(
        [{"full_name": "One"}, {"full_name": "Two"}],
        many=True,
        two_phase=True,
    )
    assert [teacher.full_name for teacher in results] == ["One", "Two"]
    assert all(sa.inspect(teacher).transient for teacher in results)


def test_two_phase_new_objects_are_not_created_during_planning(models, session):
    class TeacherSchema(SQLAlchemyAutoSchema[models.Teacher]):
        class Meta:
            model = models.Teacher
            load_instance = True
            sqla_session = session

    schema = TeacherSchema()
    plan = schema.make_update_plan(
        [{"full_name": "New One"}, {"full_name": "New Two"}], many=True
    )
    assert len(session.new) == 0
    results = plan.apply()
    assert [teacher.full_name for teacher in results] == ["New One", "New Two"]
    assert all(isinstance(teacher, models.Teacher) for teacher in results)


def test_two_phase_skips_flush_during_planning_with_related_field(
    models, session, teacher
):
    class WithRelated(SQLAlchemyAutoSchema[models.Teacher]):
        class Meta:
            model = models.Teacher
            load_instance = True
            sqla_session = session

        full_name = auto_field()
        current_school = Related()

    # unflushed caller work that an autoflush would otherwise emit
    session.add(
        models.Student(id=901, full_name="Pending", dob=None, current_school_id=42)
    )
    flushes = _listen_flushes(session)

    schema = WithRelated()
    results = schema.load(
        [
            {"id": 24, "full_name": "Teachy McTeachFace", "current_school": 42},
        ],
        many=True,
        two_phase=True,
    )
    assert results[0] is teacher
    assert flushes == []
    session.rollback()
