"""Tests for bulk resolution of Related fields on many=True loads."""

from __future__ import annotations

from unittest import mock

import pytest
import sqlalchemy as sa
from marshmallow import ValidationError

from marshmallow_sqlalchemy import SQLAlchemySchema, auto_field
from marshmallow_sqlalchemy.fields import Related, RelatedList


class SelectCounter:
    def __init__(self):
        self.count = 0

    def __call__(self, conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            self.count += 1


@pytest.fixture
def select_counter(engine):
    counter = SelectCounter()
    sa.event.listen(engine, "before_cursor_execute", counter)
    yield counter
    sa.event.remove(engine, "before_cursor_execute", counter)


@pytest.fixture
def school(models, session):
    school_ = models.School(id=1, name="Harvard")
    session.add(school_)
    session.flush()
    return school_


@pytest.fixture
def student_schema_class(models, session):
    class StudentSchema(SQLAlchemySchema):
        class Meta:
            model = models.Student
            load_instance = True
            sqla_session = session
            bulk_related = True

        full_name = auto_field()
        current_school = Related()

    return StudentSchema


@pytest.fixture
def lecture_schema_class(models, session):
    class LectureSchema(SQLAlchemySchema):
        class Meta:
            model = models.Lecture
            load_instance = True
            sqla_session = session
            bulk_related = True

        topic = auto_field()
        seminar = Related()

    return LectureSchema


def make_students_data(school_id, count=100):
    return [
        {"full_name": f"Student {i}", "current_school": school_id}
        for i in range(count)
    ]


def test_bulk_related_reuses_session_instance(
    models, session, school, student_schema_class, select_counter
):
    """Objects already in the session are reused without any SELECT."""
    schema = student_schema_class()
    students = schema.load(make_students_data(school.id), many=True)

    assert select_counter.count == 0
    assert len(students) == 100
    assert all(student.current_school is school for student in students)


def test_bulk_related_dedupes_repeated_keys(
    models, session, school, student_schema_class, select_counter
):
    """Repeated references to one key issue a single SELECT and share an instance."""
    school_id = school.id
    session.expunge_all()

    schema = student_schema_class()
    students = schema.load(make_students_data(school_id), many=True)

    assert select_counter.count == 1
    first = students[0].current_school
    assert first.id == school_id
    assert all(student.current_school is first for student in students)


def test_bulk_related_load_kwarg_opt_in(
    models, session, school, select_counter
):
    """Bulk resolution can be enabled per load call."""
    school_id = school.id
    session.expunge_all()

    class StudentSchema(SQLAlchemySchema):
        class Meta:
            model = models.Student
            load_instance = True
            sqla_session = session

        full_name = auto_field()
        current_school = Related()

    students = StudentSchema().load(
        make_students_data(school_id), many=True, bulk_related=True
    )

    assert select_counter.count == 1
    first = students[0].current_school
    assert all(student.current_school is first for student in students)


def test_bulk_related_composite_primary_key(
    models, session, lecture_schema_class, select_counter
):
    seminar = models.Seminar(title="Math", semester="Fall")
    session.add(seminar)
    session.flush()

    key = {"title": "Math", "semester": "Fall"}
    data = [
        {"topic": "Algebra", "seminar": key},
        {"topic": "Geometry", "seminar": key},
        {"topic": "Calculus", "seminar": dict(key)},
    ]
    lectures = lecture_schema_class().load(data, many=True)

    assert select_counter.count == 0
    assert all(lecture.seminar is seminar for lecture in lectures)


def test_bulk_related_composite_primary_key_expunged(
    models, session, lecture_schema_class, select_counter
):
    seminar = models.Seminar(title="Math", semester="Fall")
    session.add(seminar)
    session.flush()
    session.expunge_all()

    key = {"title": "Math", "semester": "Fall"}
    data = [{"topic": f"Topic {i}", "seminar": key} for i in range(10)]
    lectures = lecture_schema_class().load(data, many=True)

    assert select_counter.count == 1
    first = lectures[0].seminar
    assert (first.title, first.semester) == ("Math", "Fall")
    assert all(lecture.seminar is first for lecture in lectures)


def test_bulk_related_missing_related(
    models, session, school, student_schema_class, select_counter
):
    """Missing related rows still deserialize per the field's configuration."""
    data = [
        {"full_name": "Known", "current_school": school.id},
        {"full_name": "Unknown", "current_school": 9999},
    ]
    students = student_schema_class().load(data, many=True)

    assert students[0].current_school is school
    missing = students[1].current_school
    assert isinstance(missing, models.School)
    assert missing.id == 9999
    assert sa.inspect(missing).transient
    # Only the missing key required a query.
    assert select_counter.count == 1


def test_bulk_related_errors_keep_input_indices(
    models, session, lecture_schema_class
):
    seminar = models.Seminar(title="Math", semester="Fall")
    session.add(seminar)
    session.flush()

    key = {"title": "Math", "semester": "Fall"}
    data = [
        {"topic": "Algebra", "seminar": key},
        # A scalar is invalid for a composite-key Related field.
        {"topic": "Geometry", "seminar": 123},
        {"topic": "Calculus", "seminar": key},
    ]
    with pytest.raises(ValidationError) as exc_info:
        lecture_schema_class().load(data, many=True)

    assert set(exc_info.value.messages) == {1}
    assert "seminar" in exc_info.value.messages[1]
    valid_data = exc_info.value.valid_data
    assert len(valid_data) == 3
    # Related fields of valid items are still resolved from the bulk cache.
    assert valid_data[0]["seminar"] is seminar
    assert valid_data[2]["seminar"] is seminar


def test_bulk_related_cache_not_retained_between_loads(
    models, session, school, student_schema_class
):
    schema = student_schema_class()
    data = make_students_data(school.id, count=2)

    students = schema.load(data, many=True)
    assert students[0].current_school is school
    assert schema._related_cache is None

    # Once the row is gone, a later load must not reuse the stale instance.
    session.delete(school)
    session.flush()
    students = schema.load(data, many=True)
    assert schema._related_cache is None
    assert students[0].current_school is not school
    assert students[0].current_school.id == 1
    assert sa.inspect(students[0].current_school).transient


def test_bulk_related_batch_size_limits_parameters(
    models, session, select_counter
):
    schools = [models.School(name=f"School {i}") for i in range(25)]
    session.add_all(schools)
    session.flush()
    school_ids = [school.id for school in schools]
    session.expunge_all()

    class StudentSchema(SQLAlchemySchema):
        class Meta:
            model = models.Student
            load_instance = True
            sqla_session = session
            bulk_related = True
            bulk_related_batch_size = 10

        full_name = auto_field()
        current_school = Related()

    data = [
        {"full_name": f"Student {i}", "current_school": school_id}
        for i, school_id in enumerate(school_ids)
    ]
    students = StudentSchema().load(data, many=True)

    # 25 distinct keys, at most 10 parameters per query -> 3 SELECTs.
    assert select_counter.count == 3
    assert [student.current_school.id for student in students] == school_ids


def test_bulk_related_query_failure_installs_no_partial_cache(
    models, session, school, student_schema_class
):
    school_id = school.id
    session.expunge_all()
    schema = student_schema_class()
    data = make_students_data(school_id, count=5)

    with mock.patch.object(
        session, "execute", side_effect=RuntimeError("boom")
    ), pytest.raises(RuntimeError, match="boom"):
        schema.load(data, many=True)

    assert schema._related_cache is None


def test_bulk_related_custom_columns(models, session, school, select_counter):
    """Non-PK lookups are batched too, instead of one query per item."""
    session.expunge_all()

    class StudentSchema(SQLAlchemySchema):
        class Meta:
            model = models.Student
            load_instance = True
            sqla_session = session
            bulk_related = True

        full_name = auto_field()
        current_school = Related(columns=["name"])

    data = [
        {"full_name": f"Student {i}", "current_school": "Harvard"}
        for i in range(100)
    ]
    students = StudentSchema().load(data, many=True)

    assert select_counter.count == 1
    first = students[0].current_school
    assert first.name == "Harvard"
    assert all(student.current_school is first for student in students)


def test_bulk_related_list(models, session, select_counter):
    keywords = [models.Keyword(id=i, keyword=f"keyword{i}") for i in range(1, 4)]
    session.add_all(keywords)
    session.flush()

    class LectureSchema(SQLAlchemySchema):
        class Meta:
            model = models.Lecture
            load_instance = True
            sqla_session = session
            bulk_related = True

        topic = auto_field()
        kw = RelatedList(Related())

    data = [{"topic": f"Topic {i}", "kw": [1, 2, 3]} for i in range(10)]
    lectures = LectureSchema().load(data, many=True)

    assert select_counter.count == 0
    for lecture in lectures:
        assert all(lecture.kw[j] is keywords[j] for j in range(3))
