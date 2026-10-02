"""Tests for the ``require_existing`` option of `Related` fields."""

from __future__ import annotations

import pytest
from marshmallow import ValidationError

from marshmallow_sqlalchemy import (
    SQLAlchemyAutoSchema,
    SQLAlchemySchema,
    auto_field,
)
from marshmallow_sqlalchemy.fields import Related, RelatedList


@pytest.fixture
def school(models, session):
    school_ = models.School(id=1, name="Existing School")
    session.add(school_)
    session.flush()
    return school_


@pytest.fixture
def seminar(models, session):
    seminar_ = models.Seminar(title="Math", semester="Fall")
    session.add(seminar_)
    session.flush()
    return seminar_


def make_teacher_schema(models, session, **related_kwargs):
    class TeacherSchema(SQLAlchemySchema):
        class Meta:
            model = models.Teacher
            load_instance = True
            sqla_session = session

        full_name = auto_field()
        current_school = Related(**related_kwargs)

    return TeacherSchema


def make_lecture_schema(models, session):
    class LectureSchema(SQLAlchemySchema):
        class Meta:
            model = models.Lecture
            load_instance = True
            sqla_session = session

        topic = auto_field()
        seminar = Related(require_existing=True)

    return LectureSchema


class TestRequireExisting:
    def test_binds_existing_by_scalar_and_dict(self, models, session, school):
        TeacherSchema = make_teacher_schema(models, session, require_existing=True)
        for value in (school.id, {"id": school.id}):
            teacher = TeacherSchema().load(
                {"full_name": "Mr. A", "current_school": value}
            )
            assert teacher.current_school is school

    def test_not_found_raises_and_creates_nothing(self, models, session, school):
        TeacherSchema = make_teacher_schema(models, session, require_existing=True)
        with pytest.raises(ValidationError) as excinfo:
            TeacherSchema().load({"full_name": "Mr. A", "current_school": 999})
        messages = excinfo.value.messages["current_school"]
        assert "Could not find related object" in messages[0]
        # No new object was created or registered in the session
        assert len(session.new) == 0
        assert session.query(models.School).count() == 1

    def test_failed_load_leaves_instance_untouched(self, models, session, school):
        teacher = models.Teacher(full_name="Mr. T", current_school=school)
        session.add(teacher)
        session.flush()
        TeacherSchema = make_teacher_schema(models, session, require_existing=True)
        with pytest.raises(ValidationError):
            TeacherSchema().load({"current_school": 999}, instance=teacher)
        assert teacher.current_school is school
        assert len(session.new) == 0
        assert session.query(models.School).count() == 1

    def test_incomplete_key(self, models, session, school):
        TeacherSchema = make_teacher_schema(models, session, require_existing=True)
        for value in ({}, {"id": None}):
            with pytest.raises(ValidationError) as excinfo:
                TeacherSchema().load({"full_name": "Mr. A", "current_school": value})
            messages = excinfo.value.messages["current_school"]
            assert "Missing related key(s)" in messages[0]

    def test_no_session(self, models):
        class TeacherSchema(SQLAlchemySchema):
            class Meta:
                model = models.Teacher

            full_name = auto_field()
            current_school = Related(require_existing=True)

        with pytest.raises(ValidationError) as excinfo:
            TeacherSchema().load({"full_name": "Mr. A", "current_school": 1})
        messages = excinfo.value.messages["current_school"]
        assert "session" in messages[0]

    def test_allow_none(self, models, session, school):
        TeacherSchema = make_teacher_schema(
            models, session, require_existing=True, allow_none=True
        )
        teacher = TeacherSchema().load({"full_name": "Mr. A", "current_school": None})
        assert teacher.current_school is None

    def test_none_not_allowed_by_default(self, models, session, school):
        TeacherSchema = make_teacher_schema(models, session, require_existing=True)
        with pytest.raises(ValidationError) as excinfo:
            TeacherSchema().load({"full_name": "Mr. A", "current_school": None})
        assert "current_school" in excinfo.value.messages

    def test_pending_object_is_referenceable(self, models, session):
        pending_school = models.School(id=2, name="Pending School")
        session.add(pending_school)
        assert pending_school in session.new
        TeacherSchema = make_teacher_schema(models, session, require_existing=True)
        teacher = TeacherSchema().load(
            {"full_name": "Mr. A", "current_school": pending_school.id}
        )
        assert teacher.current_school is pending_school
        # The lookup must not flush the pending object
        assert pending_school in session.new

    def test_lookup_does_not_autoflush(self, models, session, school):
        keyword = models.Keyword(keyword="pending")
        session.add(keyword)
        assert keyword in session.new
        TeacherSchema = make_teacher_schema(models, session, require_existing=True)
        # A failing lookup ...
        with pytest.raises(ValidationError):
            TeacherSchema().load({"full_name": "Mr. A", "current_school": 999})
        assert keyword in session.new
        # ... and a successful lookup both leave pending objects unflushed
        TeacherSchema().load({"full_name": "Mr. A", "current_school": school.id})
        assert keyword in session.new
        with session.no_autoflush:
            assert session.query(models.Keyword).count() == 0

    def test_default_behavior_creates_transient_instance(
        self, models, session, school
    ):
        TeacherSchema = make_teacher_schema(models, session)
        teacher = TeacherSchema().load({"full_name": "Mr. A", "current_school": 999})
        assert isinstance(teacher.current_school, models.School)
        assert teacher.current_school.id == 999
        assert teacher.current_school not in session


class TestRequireExistingCompositeKey:
    def test_binds_existing_composite_key(self, models, session, seminar):
        LectureSchema = make_lecture_schema(models, session)
        lecture = LectureSchema().load(
            {"topic": "Derivatives", "seminar": {"title": "Math", "semester": "Fall"}}
        )
        assert lecture.seminar is seminar

    def test_scalar_value_invalid_for_composite_key(self, models, session, seminar):
        LectureSchema = make_lecture_schema(models, session)
        with pytest.raises(ValidationError) as excinfo:
            LectureSchema().load({"topic": "Derivatives", "seminar": "Math"})
        assert "seminar" in excinfo.value.messages

    def test_incomplete_composite_key(self, models, session, seminar):
        LectureSchema = make_lecture_schema(models, session)
        for value in (
            {"title": "Math"},
            {"semester": "Fall"},
            {"title": "Math", "semester": None},
        ):
            with pytest.raises(ValidationError) as excinfo:
                LectureSchema().load({"topic": "Derivatives", "seminar": value})
            messages = excinfo.value.messages["seminar"]
            assert "Missing related key(s)" in messages[0]

    def test_composite_key_not_found(self, models, session, seminar):
        LectureSchema = make_lecture_schema(models, session)
        with pytest.raises(ValidationError) as excinfo:
            LectureSchema().load(
                {
                    "topic": "Derivatives",
                    "seminar": {"title": "Art", "semester": "Fall"},
                }
            )
        messages = excinfo.value.messages["seminar"]
        assert "Could not find related object" in messages[0]
        assert len(session.new) == 0
        assert session.query(models.Seminar).count() == 1

    def test_pending_composite_key_object_is_referenceable(self, models, session):
        pending_seminar = models.Seminar(title="Art", semester="Spring")
        session.add(pending_seminar)
        LectureSchema = make_lecture_schema(models, session)
        lecture = LectureSchema().load(
            {"topic": "History", "seminar": {"title": "Art", "semester": "Spring"}}
        )
        assert lecture.seminar is pending_seminar
        assert pending_seminar in session.new


class TestRequireExistingMany:
    @pytest.fixture
    def students(self, models, session, school):
        students_ = [
            models.Student(full_name="Alice", current_school_id=school.id),
            models.Student(full_name="Bob", current_school_id=school.id),
        ]
        session.add_all(students_)
        session.flush()
        return students_

    def make_school_schema(self, models, session):
        class SchoolSchema(SQLAlchemySchema):
            class Meta:
                model = models.School
                load_instance = True
                sqla_session = session

            name = auto_field()
            students = RelatedList(Related(require_existing=True))

        return SchoolSchema

    def test_binds_existing_many(self, models, session, school, students):
        SchoolSchema = self.make_school_schema(models, session)
        result = SchoolSchema().load(
            {"name": "s", "students": [students[0].id, students[1].id]}
        )
        assert result.students == students

    def test_many_errors_keep_index_paths(self, models, session, school, students):
        SchoolSchema = self.make_school_schema(models, session)
        with pytest.raises(ValidationError) as excinfo:
            SchoolSchema().load(
                {"name": "s", "students": [students[0].id, 999, students[1].id, 998]}
            )
        errors = excinfo.value.messages["students"]
        assert set(errors) == {1, 3}
        assert "Could not find related object" in errors[1][0]
        # Nothing was created for the missing keys
        assert len(session.new) == 0
        assert session.query(models.Student).count() == 2


class TestRequireExistingAutoFields:
    def test_auto_field_passes_require_existing(self, models, session, school):
        class StudentSchema(SQLAlchemyAutoSchema):
            class Meta:
                model = models.Student
                load_instance = True
                sqla_session = session
                include_relationships = True

            current_school = auto_field(require_existing=True)
            courses = auto_field(require_existing=True)

        schema = StudentSchema()
        assert isinstance(schema.fields["current_school"], Related)
        assert schema.fields["current_school"].require_existing is True
        courses_field = schema.fields["courses"]
        assert isinstance(courses_field, RelatedList)
        assert courses_field.inner.require_existing is True

    def test_auto_field_enforces_existing_only(self, models, session, school):
        class StudentSchema(SQLAlchemyAutoSchema):
            class Meta:
                model = models.Student
                load_instance = True
                sqla_session = session
                include_relationships = True

            current_school = auto_field(require_existing=True)
            courses = auto_field(require_existing=True)

        with pytest.raises(ValidationError) as excinfo:
            StudentSchema().load(
                {"full_name": "Alice", "current_school": 999, "courses": [998]}
            )
        messages = excinfo.value.messages
        assert "Could not find related object" in messages["current_school"][0]
        assert set(messages["courses"]) == {0}
        assert len(session.new) == 0
        assert session.query(models.Student).count() == 0
        assert session.query(models.Course).count() == 0
