"""Tests for the "loaded only" dump mode.

Dumps with ``loaded_only=True`` must never emit SQL: deferred columns, expired
attributes and unloaded relationships are detected from the ORM instance state
and handled via the configured ``unloaded`` strategy.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.attributes import set_committed_value

from marshmallow import fields
from marshmallow_sqlalchemy import (
    OMIT,
    RAISE,
    SQLAlchemyAutoSchema,
    SQLAlchemySchema,
    UnloadedAttributeError,
    auto_field,
)
from marshmallow_sqlalchemy.fields import Nested, Related


class Base(DeclarativeBase):
    pass


class School(Base):
    __tablename__ = "school"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(50))
    motto: Mapped[str | None] = mapped_column(sa.String(50), nullable=True)
    dean_id = sa.Column(sa.ForeignKey("teacher.id"), nullable=True)
    dean = relationship("Teacher", foreign_keys=[dean_id])


class Teacher(Base):
    __tablename__ = "teacher"
    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(sa.String(50))
    secret: Mapped[str] = mapped_column(sa.String(50), deferred=True)
    school_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey("school.id"), nullable=True
    )
    school = relationship(School, foreign_keys=[school_id], back_populates="teachers")
    lessons = relationship("Lesson", back_populates="teacher")
    school_name = association_proxy("school", "name")


School.teachers = relationship(Teacher, foreign_keys=[Teacher.school_id], back_populates="school")


class Lesson(Base):
    __tablename__ = "lesson"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(sa.String(50))
    topic: Mapped[str | None] = mapped_column(sa.String(50), nullable=True)
    teacher_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey("teacher.id"), nullable=True
    )
    teacher = relationship(Teacher, back_populates="lessons")


@pytest.fixture(scope="module")
def engine():
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        if session.get(School, 1) is None:
            school = School(id=1, name="Univ. Of Whales", motto=None)
            school_with_lessons = School(id=2, name="MIT", motto="Mind and Hand")
            teacher = Teacher(
                id=1, full_name="Teachy McTeachFace", secret="classified", school=school
            )
            teacher2 = Teacher(id=2, full_name="Lazy Load", secret="s2")
            teacher3 = Teacher(id=3, full_name="Nested N.", secret="s3", school=school_with_lessons)
            lessons = [
                Lesson(id=1, title="Algebra", topic="math", teacher=teacher),
                Lesson(id=2, title="Geometry", topic=None, teacher=teacher),
            ]
            session.add_all(
                [school, school_with_lessons, teacher, teacher2, teacher3, *lessons]
            )
            session.commit()
        yield session


@pytest.fixture
def sql_counter(engine):
    statements: list[str] = []

    def listener(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", listener)
    yield statements
    event.remove(engine, "before_cursor_execute", listener)


class TeacherSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Teacher
        include_relationships = True
        include_fk = True
        loaded_only = True
        unloaded = RAISE


class SchoolSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = School
        include_relationships = True
        include_fk = True
        loaded_only = True
        unloaded = RAISE


class TeacherColumnsSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Teacher
        include_fk = True
        loaded_only = True
        unloaded = RAISE


# ---------------------------------------------------------------------------
# Eager
# ---------------------------------------------------------------------------


def test_eager_instance_dumps_loaded_columns_without_sql(session, sql_counter):
    teacher = session.get(Teacher, 1)
    assert "secret" in sa.inspect(teacher).unloaded

    sql_counter.clear()
    result = TeacherSchema().dump(teacher, unloaded=OMIT)
    assert sql_counter == []

    assert result["id"] == 1
    assert result["full_name"] == "Teachy McTeachFace"
    assert result["school_id"] == 1
    assert "secret" not in result
    assert "school" not in result
    assert "lessons" not in result


def test_loaded_null_is_serialized_as_null(session, sql_counter):
    school = session.get(School, 1)  # motto is NULL in the database
    sql_counter.clear()
    result = SchoolSchema().dump(school, unloaded=OMIT)
    assert sql_counter == []
    assert result["motto"] is None
    assert result["dean_id"] is None
    assert "dean" not in result


def test_eager_loaded_relationship_serializes_primary_key(session, sql_counter):
    teacher = session.get(Teacher, 1)
    _ = teacher.school  # eagerly touch the relationship before dumping
    sql_counter.clear()
    result = TeacherSchema().dump(teacher, unloaded=OMIT)
    assert sql_counter == []
    assert result["school"] == 1


def test_collection_of_eager_instances_emits_no_sql(session, sql_counter):
    teachers = session.scalars(sa.select(Teacher)).all()
    # leave relationships/deferred columns unloaded
    schema = TeacherSchema(unloaded=OMIT)
    sql_counter.clear()
    result = schema.dump(teachers, many=True)
    assert sql_counter == []
    assert len(result) == 3
    assert {row["full_name"] for row in result} == {
        "Teachy McTeachFace",
        "Lazy Load",
        "Nested N.",
    }


# ---------------------------------------------------------------------------
# Deferred
# ---------------------------------------------------------------------------


def test_deferred_column_raise_reports_field_path(session):
    teacher = session.get(Teacher, 1)
    with pytest.raises(UnloadedAttributeError, match=r"'secret'"):
        TeacherColumnsSchema().dump(teacher)


def test_deferred_column_omit_skips_it(session):
    teacher = session.get(Teacher, 1)
    result = TeacherColumnsSchema(unloaded=OMIT).dump(teacher)
    assert "secret" not in result
    # loaded columns are still there
    assert result["full_name"] == "Teachy McTeachFace"


def test_deferred_relationship_raise_and_omit(session):
    teacher = session.get(Teacher, 1)
    with pytest.raises(UnloadedAttributeError, match=r"'school'"):
        TeacherSchema().dump(teacher)
    result = TeacherSchema(unloaded=OMIT).dump(teacher)
    assert "school" not in result
    assert "lessons" not in result


def test_dump_default_is_not_used_as_fake_value(session):
    # An unloaded attribute must never fall back to a field dump_default;
    # with "raise" it errors, with "omit" the key is absent entirely.
    class TeacherWithDefaultSchema(SQLAlchemySchema):
        class Meta:
            model = Teacher
            loaded_only = True

        full_name = auto_field()
        secret = fields.String(dump_default="NOT-IN-DB")

    teacher = session.get(Teacher, 1)
    with pytest.raises(UnloadedAttributeError, match=r"'secret'"):
        TeacherWithDefaultSchema().dump(teacher)
    result = TeacherWithDefaultSchema(unloaded=OMIT).dump(teacher)
    assert "secret" not in result


# ---------------------------------------------------------------------------
# Expired
# ---------------------------------------------------------------------------


def test_expired_instance_does_not_refresh(session, sql_counter):
    teacher = session.get(Teacher, 1)
    session.expire(teacher)
    sql_counter.clear()
    with pytest.raises(UnloadedAttributeError, match=r"'id'"):
        TeacherColumnsSchema().dump(teacher)
    assert sql_counter == []

    result = TeacherColumnsSchema(unloaded=OMIT).dump(teacher)
    assert sql_counter == []
    # Every column/relationship is expired, so nothing remains.
    assert result == {}


def test_partially_expired_attribute_is_detected(session, sql_counter):
    teacher = session.get(Teacher, 1)
    session.expire(teacher, ["full_name"])
    sql_counter.clear()
    with pytest.raises(UnloadedAttributeError, match=r"'full_name'"):
        TeacherColumnsSchema().dump(teacher)
    assert sql_counter == []
    result = TeacherColumnsSchema(unloaded=OMIT).dump(teacher)
    assert result["id"] == 1
    assert "full_name" not in result


# ---------------------------------------------------------------------------
# Detached
# ---------------------------------------------------------------------------


def test_detached_instance_with_loaded_data_dumps_without_sql(session, sql_counter):
    teacher = session.get(Teacher, 1, options=[sa.orm.load_only(Teacher.id, Teacher.full_name)])
    session.expunge(teacher)
    assert sa.inspect(teacher).detached
    sql_counter.clear()
    result = TeacherColumnsSchema(unloaded=OMIT).dump(teacher)
    assert sql_counter == []
    assert result["id"] == 1
    assert result["full_name"] == "Teachy McTeachFace"
    assert "school_id" not in result
    assert "secret" not in result

    # ... but with all columns loaded, expunge preserves them and emits no SQL.
    full = session.get(Teacher, 1)
    _ = full.secret  # explicitly load the deferred column
    session.expunge(full)
    full_result = TeacherColumnsSchema(unloaded=OMIT).dump(full)
    assert full_result["school_id"] == 1
    assert full_result["secret"] == "classified"


def test_detached_expired_instance_raises_without_detached_error(session, sql_counter):
    teacher = session.get(Teacher, 1)
    session.expire(teacher)
    session.expunge(teacher)
    sql_counter.clear()
    with pytest.raises(UnloadedAttributeError, match=r"detached"):
        TeacherColumnsSchema().dump(teacher)
    assert sql_counter == []
    result = TeacherColumnsSchema(unloaded=OMIT).dump(teacher)
    assert result == {}
    assert sql_counter == []


def test_detached_loaded_relationship_is_serialized(session, sql_counter):
    teacher = session.get(Teacher, 1)
    _ = teacher.school
    session.expunge(teacher)
    sql_counter.clear()
    result = TeacherSchema(unloaded=OMIT).dump(teacher)
    assert sql_counter == []
    assert result["school"] == 1
    assert "secret" not in result


# ---------------------------------------------------------------------------
# Nested schemas inherit the policy; paths are complete
# ---------------------------------------------------------------------------


class NestedTeacherSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Teacher
        include_relationships = True
        include_fk = True
        # NOTE: no loaded_only here -- it inherits the parent dump policy.

    lessons = fields.List(Nested(lambda: LessonSchema()))


class LessonSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Lesson
        include_relationships = True
        include_fk = True

    title = auto_field()


class NestedSchoolSchema(SQLAlchemySchema):
    class Meta:
        model = School
        loaded_only = True
        unloaded = RAISE

    id = auto_field()
    name = auto_field()
    teachers = fields.List(Nested(NestedTeacherSchema))


def test_nested_schema_inherits_policy_and_reports_full_path(session):
    school = session.get(School, 2)
    _ = school.teachers  # load the collection, but teachers' lessons stay unloaded
    with pytest.raises(UnloadedAttributeError, match=r"'teachers\[0\]\.lessons'"):
        NestedSchoolSchema().dump(school)


def test_nested_schema_omit_inherits_omit(session, sql_counter):
    school = session.get(School, 2)
    _ = school.teachers
    sql_counter.clear()
    result = NestedSchoolSchema(unloaded=OMIT).dump(school)
    assert sql_counter == []
    assert result["id"] == 2
    assert result["teachers"][0]["id"] == 3
    assert "lessons" not in result["teachers"][0]


def test_nested_loaded_data_serializes_normally(session, sql_counter):
    teacher = session.get(Teacher, 2)
    set_committed_value(teacher, "lessons", [])
    sql_counter.clear()

    class TeacherWithLessonsSchema(SQLAlchemySchema):
        class Meta:
            model = Teacher
            loaded_only = True
            unloaded = OMIT

        id = auto_field()
        lessons = fields.List(Nested(LessonSchema()))

    result = TeacherWithLessonsSchema().dump(teacher)
    assert sql_counter == []
    assert result == {"id": 2, "lessons": []}


def test_top_level_many_path_includes_index(session):
    teachers = session.scalars(
        sa.select(Teacher).order_by(Teacher.id),
        execution_options={"populate_existing": True},
    ).all()
    # First row fully loaded; second row keeps `secret` deferred.
    _ = teachers[0].secret
    with pytest.raises(UnloadedAttributeError, match=r"\[1\]\.secret"):
        TeacherColumnsSchema().dump(teachers, many=True)


def test_related_field_with_columns_checks_loaded_state(session, sql_counter):
    class TeacherRelatedSchema(SQLAlchemySchema):
        class Meta:
            model = Teacher
            loaded_only = True

        id = auto_field()
        school = Related(["id", "name"])

    teacher = session.get(Teacher, 1)
    _ = teacher.school
    # expire only the related school's name to simulate partially loaded data
    session.expire(teacher.school, ["name"])
    sql_counter.clear()
    with pytest.raises(UnloadedAttributeError, match=r"'school\.name'"):
        TeacherRelatedSchema().dump(teacher)
    assert sql_counter == []
    result = TeacherRelatedSchema(unloaded=OMIT).dump(teacher)
    assert "school" not in result


# ---------------------------------------------------------------------------
def test_association_proxy_depends_on_underlying_relationship(session, sql_counter):
    class TeacherProxySchema(SQLAlchemySchema):
        class Meta:
            model = Teacher
            loaded_only = True

        school_name = auto_field()

    teacher = session.get(Teacher, 1)  # school relationship unloaded
    sql_counter.clear()
    with pytest.raises(UnloadedAttributeError, match=r"'school_name'"):
        TeacherProxySchema().dump(teacher)
    assert sql_counter == []
    assert TeacherProxySchema(unloaded=OMIT).dump(teacher) == {}
    assert sql_counter == []


def test_association_proxy_emits_value_when_relationship_loaded(session, sql_counter):
    class TeacherProxySchema(SQLAlchemySchema):
        class Meta:
            model = Teacher
            loaded_only = True
            unloaded = OMIT

        school_name = auto_field()

    teacher = session.get(Teacher, 1)
    _ = teacher.school
    sql_counter.clear()
    assert TeacherProxySchema().dump(teacher) == {"school_name": "Univ. Of Whales"}
    assert sql_counter == []


def test_plain_marshmallow_parent_nesting_sqla_schema_inherits_policy(session):
    from marshmallow import Schema as PlainSchema

    class InnerTeacherSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = Teacher
            include_fk = True
            loaded_only = True
            unloaded = RAISE

    class OuterSchema(PlainSchema):
        label = fields.String()
        teacher = Nested(InnerTeacherSchema)

    teacher = session.get(Teacher, 1)
    with pytest.raises(UnloadedAttributeError, match=r"'teacher\.secret'"):
        OuterSchema().dump({"label": "x", "teacher": teacher})

    class InnerOmitSchema(InnerTeacherSchema):
        class Meta(InnerTeacherSchema.Meta):
            unloaded = OMIT

    class OuterOmitSchema(PlainSchema):
        teacher = Nested(InnerOmitSchema)

    result = OuterOmitSchema().dump({"teacher": teacher})
    assert "secret" not in result["teacher"]
    assert result["teacher"]["full_name"] == "Teachy McTeachFace"


def test_nested_many_path_and_index_without_parent_mode(session):
    class InnerTeacherSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = Teacher
            include_fk = True
            loaded_only = True

    class SchoolNestedManySchema(SQLAlchemySchema):
        class Meta:
            model = School
            loaded_only = True

        teachers = Nested(InnerTeacherSchema, many=True)

    school = session.get(School, 2)
    _ = school.teachers
    with pytest.raises(UnloadedAttributeError, match=r"'teachers\[0\]\.secret'"):
        SchoolNestedManySchema().dump(school)


def test_field_default_value_is_not_confused_with_loaded_value(session):
    # A Python-side column default on an unset (transient) instance must not be
    # emitted as if it came from the database.
    class TeacherNameSchema(SQLAlchemySchema):
        class Meta:
            model = Teacher
            loaded_only = True

        id = auto_field()
        full_name = auto_field()

    pending = Teacher(id=99)
    with pytest.raises(UnloadedAttributeError, match=r"'full_name'"):
        TeacherNameSchema().dump(pending)
    assert TeacherNameSchema(unloaded=OMIT).dump(pending) == {"id": 99}


# Misc
# ---------------------------------------------------------------------------


def test_unloaded_option_is_validated():
    with pytest.raises(ValueError, match="unloaded"):

        class BadSchema(SQLAlchemySchema):
            class Meta:
                model = Teacher
                loaded_only = True
                unloaded = "explode"


def test_unloaded_keyword_argument_is_validated(session):
    teacher = session.get(Teacher, 1)
    with pytest.raises(ValueError, match="unloaded"):
        TeacherSchema().dump(teacher, unloaded="nope")


def test_mode_disabled_keeps_lazy_loading_behavior(session, sql_counter):
    class PlainSchema(SQLAlchemyAutoSchema):
        class Meta:
            model = Teacher
            include_fk = True

    teacher = session.get(Teacher, 1)
    sql_counter.clear()
    result = PlainSchema().dump(teacher)
    # default behavior lazy-loads deferred columns -> SQL is emitted
    assert sql_counter
    assert result["secret"] == "classified"


def test_plain_mapping_objects_are_unaffected():
    class PlainSchema(SQLAlchemySchema):
        class Meta:
            model = Teacher
            loaded_only = True
            unloaded = RAISE

        full_name = auto_field()

    # Dicts/mappings are not ORM instances: normal access applies.
    assert PlainSchema().dump({"full_name": "dict value"}) == {"full_name": "dict value"}
