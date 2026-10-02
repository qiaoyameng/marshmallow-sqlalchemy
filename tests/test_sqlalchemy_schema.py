from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from importlib.metadata import version

import marshmallow
import pytest
import sqlalchemy as sa
from marshmallow import Schema, ValidationError, fields, validate
from pytest_lazy_fixtures import lf

from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, SQLAlchemySchema, auto_field
from marshmallow_sqlalchemy.exceptions import IncorrectSchemaTypeError
from marshmallow_sqlalchemy.fields import Related, RelatedBatch, RelatedList

from .conftest import CourseLevel

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


@contextmanager
def count_selects(engine):
    statements = []

    def before_cursor_execute(
        conn,
        cursor,
        statement,
        parameters,
        context,
        executemany,
    ):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append((statement, parameters))

    sa.event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        yield statements
    finally:
        sa.event.remove(engine, "before_cursor_execute", before_cursor_execute)


class TestBatchRelated:
    def test_reuses_results_for_duplicate_related_keys(self, models, session, engine):
        schools = [models.School(id=1, name="First"), models.School(id=2, name="Second")]
        session.add_all(schools)
        session.flush()

        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                load_instance = True
                sqla_session = session

            full_name = auto_field()
            current_school = Related()

        schema = StudentSchema()
        with count_selects(engine) as statements:
            students = schema.load(
                [
                    {"full_name": "One", "current_school": 1},
                    {"full_name": "Two", "current_school": 1},
                    {"full_name": "Three", "current_school": 2},
                    {"full_name": "Four", "current_school": 2},
                    {"full_name": "Missing", "current_school": 3},
                ],
                many=True,
                batch_related=True,
            )

        assert len(statements) == 1
        assert students[0].current_school is schools[0]
        assert students[1].current_school is schools[0]
        assert students[2].current_school is schools[1]
        assert students[3].current_school is schools[1]
        assert isinstance(students[4].current_school, models.School)

    def test_reuses_objects_already_in_session_identity_map(
        self, models, session, engine
    ):
        school = models.School(id=1, name="First")
        session.add(school)
        session.flush()

        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                load_instance = True
                sqla_session = session

            full_name = auto_field()
            current_school = Related()

        schema = StudentSchema()
        with count_selects(engine) as statements:
            students = schema.load(
                [
                    {"full_name": "One", "current_school": 1},
                    {"full_name": "Two", "current_school": 1},
                ],
                many=True,
                batch_related=True,
            )

        assert statements == []
        assert students[0].current_school is school
        assert students[1].current_school is school

    def test_can_enable_batch_related_in_meta(self, models, session, engine):
        session.add(models.School(id=1, name="First"))
        session.flush()

        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                load_instance = True
                sqla_session = session
                batch_related = True

            full_name = auto_field()
            current_school = Related()

        schema = StudentSchema()
        with count_selects(engine) as statements:
            students = schema.load(
                [
                    {"full_name": "One", "current_school": 1},
                    {"full_name": "Two", "current_school": 1},
                ],
                many=True,
            )

        assert len(statements) == 1
        assert students[0].current_school is students[1].current_school

    def test_resolves_related_list_in_one_query(self, models, session, engine):
        courses = [
            models.Course(
                id=1,
                name="First",
                cost=1,
                level="Primary",
                level_with_enum_class=CourseLevel.PRIMARY,
                has_prereqs=False,
                started=dt.datetime.now(),
                grade=1,
                transcription="one",
            ),
            models.Course(
                id=2,
                name="Second",
                cost=2,
                level="Primary",
                level_with_enum_class=CourseLevel.PRIMARY,
                has_prereqs=False,
                started=dt.datetime.now(),
                grade=2,
                transcription="two",
            ),
        ]
        session.add_all(courses)
        session.flush()
        for course in courses:
            session.expunge(course)

        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                load_instance = True
                sqla_session = session

            full_name = auto_field()
            current_school = Related()
            courses = RelatedList(Related())

        school = models.School(id=1, name="School")
        session.add(school)
        schema = StudentSchema()
        with count_selects(engine) as statements:
            students = schema.load(
                [
                    {
                        "full_name": "One",
                        "current_school": 1,
                        "courses": [1, 2],
                    },
                    {
                        "full_name": "Two",
                        "current_school": 1,
                        "courses": [1],
                    },
                ],
                many=True,
                batch_related=True,
            )

        assert len(statements) == 2
        assert students[0].courses[0] is students[1].courses[0]
        assert students[0].courses[1].id == courses[1].id
        assert students[0].courses[0].id == courses[0].id

    def test_resolves_composite_primary_keys(self, models, session, engine):
        seminar = models.Seminar(title="ORM", semester="Fall")
        session.add(seminar)
        session.flush()
        session.expunge(seminar)

        class LectureSchema(SQLAlchemySchema):
            class Meta:
                model = models.Lecture
                sqla_session = session

            topic = auto_field()
            seminar = Related()

        schema = LectureSchema()
        with count_selects(engine) as statements:
            lectures = schema.load(
                [
                    {"topic": "First", "seminar": {"title": "ORM", "semester": "Fall"}},
                    {"topic": "Second", "seminar": {"title": "ORM", "semester": "Fall"}},
                ],
                many=True,
                batch_related=True,
            )

        assert len(statements) == 1
        assert lectures[0]["seminar"] is lectures[1]["seminar"]
        assert lectures[0]["seminar"].title == seminar.title

    def test_resolves_custom_columns_in_one_query(self, models, session, engine):
        schools = [
            models.School(id=1, name="First"),
            models.School(id=2, name="Second"),
        ]
        session.add_all(schools)
        session.flush()

        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                sqla_session = session

            full_name = auto_field()
            current_school = Related(["id", "name"])

        schema = StudentSchema()
        with count_selects(engine) as statements:
            students = schema.load(
                [
                    {"full_name": "One", "current_school": {"id": 1, "name": "First"}},
                    {"full_name": "Two", "current_school": {"id": 2, "name": "Second"}},
                ],
                many=True,
                batch_related=True,
            )

        assert len(statements) == 1
        assert students[0]["current_school"] is schools[0]
        assert students[1]["current_school"] is schools[1]

    def test_limits_number_of_parameters_per_query(self, models, session, engine):
        session.add_all(
            [models.School(id=id_, name=f"School {id_}") for id_ in range(1, 7)]
        )
        session.flush()

        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                sqla_session = session

            full_name = auto_field()
            current_school = Related()

        schema = StudentSchema()
        with count_selects(engine) as statements:
            schema.load(
                [
                    {"full_name": str(id_), "current_school": id_}
                    for id_ in range(1, 7)
                ],
                many=True,
                batch_related=True,
                related_batch_size=3,
            )

        assert len(statements) == 2
        assert all(statement.count("?") <= 3 for statement, _ in statements)

    def test_limits_parameters_for_composite_keys(self, models, session, engine):
        session.add_all(
            [
                models.Seminar(title=f"Title {id_}", semester=f"Term {id_}")
                for id_ in range(1, 5)
            ]
        )
        session.flush()

        class LectureSchema(SQLAlchemySchema):
            class Meta:
                model = models.Lecture
                sqla_session = session

            topic = auto_field()
            seminar = Related()

        schema = LectureSchema()
        with count_selects(engine) as statements:
            schema.load(
                [
                    {
                        "topic": str(id_),
                        "seminar": {
                            "title": f"Title {id_}",
                            "semester": f"Term {id_}",
                        },
                    }
                    for id_ in range(1, 5)
                ],
                many=True,
                batch_related=True,
                related_batch_size=4,
            )

        assert len(statements) == 2
        assert all(len(parameters) <= 5 for _, parameters in statements)

    def test_missing_related_fallbacks_are_not_shared_across_loads(
        self, models, session
    ):
        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                sqla_session = session

            full_name = auto_field()
            current_school = Related()

        schema = StudentSchema()
        first = schema.load(
            [
                {"full_name": "One", "current_school": 99},
                {"full_name": "Two", "current_school": 99},
            ],
            many=True,
            batch_related=True,
        )
        second = schema.load(
            [{"full_name": "Three", "current_school": 99}],
            many=True,
            batch_related=True,
        )

        assert first[0]["current_school"] is first[1]["current_school"]
        assert second[0]["current_school"] is not first[0]["current_school"]

    def test_batch_related_override_does_not_persist_across_loads(
        self, models, session
    ):
        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                sqla_session = session

            full_name = auto_field()
            current_school = Related()

        schema = StudentSchema()
        schema.load(
            [{"full_name": "One", "current_school": 1}],
            many=True,
            batch_related=True,
        )

        assert schema._batch_related is False
        assert schema._related_batch is None

    def test_validation_errors_keep_original_indexes(self, models, session):
        class LectureSchema(SQLAlchemySchema):
            class Meta:
                model = models.Lecture
                sqla_session = session

            topic = auto_field()
            seminar = Related()

        schema = LectureSchema()
        with pytest.raises(ValidationError) as exc_info:
            schema.load(
                [
                    {"topic": "Valid", "seminar": {"title": "ORM", "semester": "Fall"}},
                    {"topic": "Invalid", "seminar": "ORM"},
                ],
                many=True,
                batch_related=True,
            )

        assert tuple(exc_info.value.messages) == (1,)
        assert "seminar" in exc_info.value.messages[1]

    def test_failed_query_does_not_store_partial_results(self, models, session):
        class StudentSchema(SQLAlchemySchema):
            class Meta:
                model = models.Student
                sqla_session = session

            full_name = auto_field()
            current_school = Related()

        schema = StudentSchema()
        original_query_chunk = RelatedBatch._query_chunk

        def failing_query_chunk(batch, field, related_model, keys, found):
            if len(keys) == 3 and keys[0] == (4,):
                raise RuntimeError("query failed")
            return original_query_chunk(
                batch, field, related_model, keys, found
            )

        RelatedBatch._query_chunk = failing_query_chunk
        try:
            with pytest.raises(RuntimeError, match="query failed"):
                schema.load(
                    [
                        {"full_name": str(id_), "current_school": id_}
                        for id_ in range(1, 7)
                    ],
                    many=True,
                    batch_related=True,
                    related_batch_size=3,
                )
        finally:
            RelatedBatch._query_chunk = original_query_chunk

        assert schema._related_batch is None
        assert not any(
            "current_school" in obj.__dict__ for obj in session.new
        )


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
