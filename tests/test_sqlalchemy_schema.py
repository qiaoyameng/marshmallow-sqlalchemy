from __future__ import annotations

import datetime as dt
from importlib.metadata import version

import marshmallow
import pytest
import sqlalchemy as sa
from marshmallow import Schema, ValidationError, fields, validate
from pytest_lazy_fixtures import lf

from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, SQLAlchemySchema, auto_field
from marshmallow_sqlalchemy.exceptions import IncorrectSchemaTypeError
from marshmallow_sqlalchemy.fields import Related, RelatedList

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


class TestRelatedRequireExisting:
    @pytest.fixture
    def student_schema(self, models, session):
        class StudentSchema(SQLAlchemySchema[models.Student]):
            class Meta:
                model = models.Student
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            current_school = Related(require_existing=True)

        return StudentSchema()

    def test_existing_object_is_bound(self, models, session, school, student_schema):
        student = student_schema.load({"full_name": "New Kid", "current_school": 42})
        assert student.current_school is school

    def test_missing_object_raises(self, models, session, school, student_schema):
        with pytest.raises(ValidationError) as excinfo:
            student_schema.load({"full_name": "New Kid", "current_school": 999})
        errors = excinfo.value.messages
        assert "current_school" in errors
        assert "Could not find existing related object" in str(
            errors["current_school"]
        )
        # The failed load must not leave anything pending in the session,
        # must not write new rows, and must not touch existing relations.
        assert len(session.new) == 0
        assert session.query(models.School).count() == 1
        assert session.query(models.Student).count() == 2

    def test_failed_load_preserves_existing_instance(
        self, models, session, school, student_schema
    ):
        student = models.Student(
            id=1, full_name="Existing", current_school=school
        )
        session.add(student)
        session.flush()
        with pytest.raises(ValidationError):
            student_schema.load(
                {"full_name": "Renamed", "current_school": 999}, instance=student
            )
        assert student.full_name == "Existing"
        assert student.current_school is school
        assert len(session.new) == 0

    def test_pending_object_is_referenceable(
        self, models, session, student_schema
    ):
        pending_school = models.School(id=77, name="Pending School")
        session.add(pending_school)
        student = student_schema.load({"full_name": "New Kid", "current_school": 77})
        assert student.current_school is pending_school

    def test_no_session_raises(self, models, school):
        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student

            full_name = auto_field()
            current_school = Related(require_existing=True)

        with pytest.raises(ValidationError) as excinfo:
            StudentSchema().load({"full_name": "New Kid", "current_school": 42})
        assert "no session is available" in str(excinfo.value.messages)

    def test_transient_raises(self, models, session, school, student_schema):
        with pytest.raises(ValidationError) as excinfo:
            student_schema.load(
                {"full_name": "New Kid", "current_school": 42}, transient=True
            )
        assert "no session is available" in str(excinfo.value.messages)

    def test_composite_key(self, models, session):
        seminar = models.Seminar(title="Algebra", semester="Fall")
        session.add(seminar)
        session.flush()

        class LectureSchema(SQLAlchemySchema[models.Lecture]):
            class Meta:
                model = models.Lecture
                sqla_session = session
                load_instance = True

            topic = auto_field()
            seminar = Related(require_existing=True)

        schema = LectureSchema()
        lecture = schema.load(
            {
                "topic": "Linear maps",
                "seminar": {"title": "Algebra", "semester": "Fall"},
            }
        )
        assert lecture.seminar is seminar

        # Incomplete composite key
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"topic": "Linear maps", "seminar": {"title": "Algebra"}})
        assert "missing values for keys" in str(excinfo.value.messages)

        # Complete key that does not exist
        with pytest.raises(ValidationError) as excinfo:
            schema.load(
                {
                    "topic": "Linear maps",
                    "seminar": {"title": "Algebra", "semester": "Spring"},
                }
            )
        assert "Could not find existing related object" in str(excinfo.value.messages)

    def test_custom_columns(self, models, session, school):
        class StudentSchema(SQLAlchemySchema[models.Student]):
            class Meta:
                model = models.Student
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            current_school = Related(["id", "name"], require_existing=True)

        schema = StudentSchema()
        student = schema.load(
            {
                "full_name": "New Kid",
                "current_school": {"id": 42, "name": "Univ. Of Whales"},
            }
        )
        assert student.current_school is school

        with pytest.raises(ValidationError) as excinfo:
            schema.load(
                {
                    "full_name": "New Kid",
                    "current_school": {"id": 42, "name": "Wrong Name"},
                }
            )
        assert "Could not find existing related object" in str(excinfo.value.messages)

    def test_custom_columns_find_pending_object(self, models, session):
        pending_school = models.School(id=77, name="Pending School")
        session.add(pending_school)

        class StudentSchema(SQLAlchemySchema[models.Student]):
            class Meta:
                model = models.Student
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            current_school = Related(["id", "name"], require_existing=True)

        student = StudentSchema().load(
            {
                "full_name": "New Kid",
                "current_school": {"id": 77, "name": "Pending School"},
            }
        )
        assert student.current_school is pending_school

    def test_lookup_does_not_autoflush(self, models, session, school):
        # A pending object that would fail on flush must not be flushed
        # as a side effect of deserializing a related value.
        session.add(models.Student(full_name=None, current_school=school))

        class StudentSchema(SQLAlchemySchema[models.Student]):
            class Meta:
                model = models.Student
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            current_school = Related(["id", "name"], require_existing=True)

        student = StudentSchema().load(
            {
                "full_name": "New Kid",
                "current_school": {"id": 42, "name": "Univ. Of Whales"},
            }
        )
        assert student.current_school is school

    def test_allow_none(self, models, session, school):
        class TeacherSchema(SQLAlchemySchema[models.Teacher]):
            class Meta:
                model = models.Teacher
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            current_school = Related(require_existing=True, allow_none=True)

        teacher = TeacherSchema().load(
            {"full_name": "No School", "current_school": None}
        )
        assert teacher.current_school is None

    def test_related_list_errors_keep_path_structure(
        self, models, session, school
    ):
        course = models.Course(
            id=1,
            name="Math",
            cost=1.0,
            level="Primary",
            level_with_enum_class="PRIMARY",
            has_prereqs=False,
            started=dt.datetime(2020, 1, 1),
            grade=1,
            transcription="t",
        )
        session.add(course)
        session.flush()

        class StudentSchema(SQLAlchemySchema[models.Student]):
            class Meta:
                model = models.Student
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            courses = RelatedList(Related(require_existing=True))

        schema = StudentSchema()
        student = schema.load({"full_name": "New Kid", "courses": [1]})
        assert student.courses == [course]

        with pytest.raises(ValidationError) as excinfo:
            schema.load({"full_name": "New Kid", "courses": [1, 999]})
        errors = excinfo.value.messages
        assert list(errors["courses"].keys()) == [1]
        assert "Could not find existing related object" in str(errors["courses"][1])

    def test_default_behavior_unchanged(self, models, session, school):
        class StudentSchema(SQLAlchemySchema[models.Student]):
            class Meta:
                model = models.Student
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            current_school = Related()

        student = StudentSchema().load({"full_name": "New Kid", "current_school": 999})
        # Without require_existing, a new transient instance is created
        assert student.current_school.id == 999
        assert student.current_school not in session

    def test_auto_generated_relationship_field(self, models, session, school):
        class StudentSchema(SQLAlchemySchema[models.Student]):
            class Meta:
                model = models.Student
                sqla_session = session
                load_instance = True

            full_name = auto_field()
            current_school = auto_field(require_existing=True)
            courses = auto_field(require_existing=True)

        schema = StudentSchema()
        assert schema.fields["current_school"].require_existing is True
        assert schema.fields["courses"].inner.require_existing is True

        student = schema.load({"full_name": "New Kid", "current_school": 42})
        assert student.current_school is school

        with pytest.raises(ValidationError):
            schema.load({"full_name": "New Kid", "current_school": 999})
        with pytest.raises(ValidationError):
            schema.load({"full_name": "New Kid", "courses": [999]})
