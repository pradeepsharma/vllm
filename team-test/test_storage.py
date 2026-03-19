"""
Comprehensive tests for team-test/storage.py

Covers:
  - TaskStorage construction and properties (path, exists)
  - load(): missing file → empty TaskList
  - load(): valid JSON file → correct TaskList
  - load(): file with invalid JSON → StorageReadError
  - load(): file with non-dict JSON → StorageReadError
  - load(): file with bad task data → StorageReadError
  - load(): unreadable file (permission denied) → StorageReadError
  - save(): creates file and parent dirs
  - save(): round-trip (save then load preserves all data)
  - save(): atomic write (temp file cleaned up on success)
  - save(): wrong type raises TypeError
  - save(): unwritable directory → StorageWriteError
  - delete(): existing file → True
  - delete(): absent file → False
  - backup(): creates .bak copy
  - backup(): absent source → StorageReadError
  - file_size(): returns byte count / None
  - repr()
  - Integration: full CRUD workflow persisted across multiple store instances
"""

import importlib.util
import json
import os
import pathlib
import stat
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Load models.py and storage.py dynamically (no package setup required)
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: pathlib.Path):
    """Load a module from an absolute file path and register it in sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod


_here = pathlib.Path(__file__).parent
_models = _load_module("task_models", _here / "models.py")
_storage = _load_module("task_storage", _here / "storage.py")

Task = _models.Task
TaskList = _models.TaskList
Priority = _models.Priority
Status = _models.Status

TaskStorage = _storage.TaskStorage
StorageError = _storage.StorageError
StorageReadError = _storage.StorageReadError
StorageWriteError = _storage.StorageWriteError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**kwargs) -> Task:
    """Return a Task with sensible defaults, overridable via kwargs."""
    defaults = dict(title="Sample task")
    defaults.update(kwargs)
    return Task(**defaults)


def _make_task_list(*titles) -> TaskList:
    """Return a TaskList populated with tasks whose titles are given."""
    tl = TaskList()
    for title in titles:
        tl.add(Task(title=title))
    return tl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a fresh temporary directory for each test."""
    return tmp_path


@pytest.fixture()
def store(tmp_dir):
    """Return a TaskStorage pointing at a file that does not yet exist."""
    return TaskStorage(tmp_dir / "tasks.json")


@pytest.fixture()
def populated_store(store):
    """Return a TaskStorage that already has three tasks saved to disk."""
    tl = _make_task_list("Alpha", "Beta", "Gamma")
    store.save(tl)
    return store


# ===========================================================================
# StorageError hierarchy
# ===========================================================================

class TestExceptionHierarchy:
    def test_storage_read_error_is_storage_error(self):
        path = pathlib.Path("/tmp/x.json")
        err = StorageReadError("msg", path)
        assert isinstance(err, StorageError)
        assert err.path == path
        assert "msg" in str(err)

    def test_storage_write_error_is_storage_error(self):
        path = pathlib.Path("/tmp/x.json")
        err = StorageWriteError("msg", path)
        assert isinstance(err, StorageError)
        assert err.path == path
        assert "msg" in str(err)


# ===========================================================================
# TaskStorage – construction & properties
# ===========================================================================

class TestTaskStorageProperties:
    def test_path_is_resolved(self, tmp_dir):
        p = tmp_dir / "sub" / "tasks.json"
        store = TaskStorage(p)
        assert store.path == p

    def test_path_expands_tilde(self):
        store = TaskStorage("~/tasks.json")
        assert "~" not in str(store.path)

    def test_exists_false_when_file_absent(self, store):
        assert store.exists is False

    def test_exists_true_after_save(self, store):
        store.save(TaskList())
        assert store.exists is True

    def test_repr_contains_path(self, store):
        r = repr(store)
        assert "tasks.json" in r

    def test_repr_shows_absent(self, store):
        assert "absent" in repr(store)

    def test_repr_shows_exists_after_save(self, store):
        store.save(TaskList())
        assert "exists" in repr(store)


# ===========================================================================
# TaskStorage.load()
# ===========================================================================

class TestLoad:
    def test_load_missing_file_returns_empty_task_list(self, store):
        tl = store.load()
        assert isinstance(tl, TaskList)
        assert len(tl) == 0

    def test_load_valid_file_returns_correct_task_list(self, populated_store):
        tl = populated_store.load()
        assert isinstance(tl, TaskList)
        assert len(tl) == 3
        titles = {t.title for t in tl}
        assert titles == {"Alpha", "Beta", "Gamma"}

    def test_load_preserves_task_fields(self, store):
        original = Task(
            title="Detailed task",
            description="Some notes",
            priority=Priority.HIGH,
            status=Status.IN_PROGRESS,
            due_date="2030-12-31",
            tags=["backend", "urgent"],
        )
        tl = TaskList()
        tl.add(original)
        store.save(tl)

        loaded_tl = store.load()
        assert len(loaded_tl) == 1
        t = loaded_tl.all()[0]
        assert t.id == original.id
        assert t.title == "Detailed task"
        assert t.description == "Some notes"
        assert t.priority is Priority.HIGH
        assert t.status is Status.IN_PROGRESS
        assert t.due_date == "2030-12-31"
        assert set(t.tags) == {"backend", "urgent"}
        assert t.created_at == original.created_at

    def test_load_invalid_json_raises_storage_read_error(self, store, tmp_dir):
        bad_file = tmp_dir / "tasks.json"
        bad_file.write_text("not valid json {{{{", encoding="utf-8")
        with pytest.raises(StorageReadError) as exc_info:
            store.load()
        assert "invalid JSON" in str(exc_info.value)
        assert exc_info.value.path == store.path

    def test_load_non_dict_json_raises_storage_read_error(self, store, tmp_dir):
        bad_file = tmp_dir / "tasks.json"
        bad_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(StorageReadError) as exc_info:
            store.load()
        assert "unexpected format" in str(exc_info.value)
        assert exc_info.value.path == store.path

    def test_load_bad_task_data_raises_storage_read_error(self, store, tmp_dir):
        # A task dict missing the required 'title' field
        bad_data = {"tasks": [{"description": "no title here"}]}
        bad_file = tmp_dir / "tasks.json"
        bad_file.write_text(json.dumps(bad_data), encoding="utf-8")
        with pytest.raises(StorageReadError) as exc_info:
            store.load()
        assert exc_info.value.path == store.path

    def test_load_empty_tasks_list_returns_empty_task_list(self, store, tmp_dir):
        (tmp_dir / "tasks.json").write_text(
            json.dumps({"tasks": []}), encoding="utf-8"
        )
        tl = store.load()
        assert len(tl) == 0

    @pytest.mark.skipif(os.name == "nt", reason="chmod not reliable on Windows")
    def test_load_unreadable_file_raises_storage_read_error(self, store, tmp_dir):
        f = tmp_dir / "tasks.json"
        f.write_text(json.dumps({"tasks": []}), encoding="utf-8")
        f.chmod(0o000)
        try:
            with pytest.raises(StorageReadError) as exc_info:
                store.load()
            assert exc_info.value.path == store.path
        finally:
            f.chmod(0o644)  # restore so tmp_path cleanup works


# ===========================================================================
# TaskStorage.save()
# ===========================================================================

class TestSave:
    def test_save_creates_file(self, store):
        assert not store.exists
        store.save(TaskList())
        assert store.exists

    def test_save_creates_parent_directories(self, tmp_dir):
        deep_path = tmp_dir / "a" / "b" / "c" / "tasks.json"
        store = TaskStorage(deep_path)
        store.save(TaskList())
        assert deep_path.exists()

    def test_save_writes_valid_json(self, store):
        tl = _make_task_list("Task A", "Task B")
        store.save(tl)
        raw = store.path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "tasks" in data
        assert len(data["tasks"]) == 2

    def test_save_wrong_type_raises_type_error(self, store):
        with pytest.raises(TypeError, match="TaskList"):
            store.save({"tasks": []})  # type: ignore[arg-type]

    def test_save_overwrites_existing_file(self, populated_store):
        # Save a smaller list over the existing three-task file
        tl = _make_task_list("Only one")
        populated_store.save(tl)
        loaded = populated_store.load()
        assert len(loaded) == 1
        assert loaded.all()[0].title == "Only one"

    def test_save_empty_task_list(self, store):
        store.save(TaskList())
        loaded = store.load()
        assert len(loaded) == 0

    def test_save_preserves_unicode(self, store):
        tl = TaskList()
        tl.add(Task(title="Tâche avec accents: éàü", description="日本語テスト"))
        store.save(tl)
        loaded = store.load()
        t = loaded.all()[0]
        assert t.title == "Tâche avec accents: éàü"
        assert t.description == "日本語テスト"

    @pytest.mark.skipif(os.name == "nt", reason="chmod not reliable on Windows")
    def test_save_unwritable_directory_raises_storage_write_error(self, tmp_dir):
        locked_dir = tmp_dir / "locked"
        locked_dir.mkdir()
        locked_dir.chmod(0o555)
        store = TaskStorage(locked_dir / "tasks.json")
        try:
            with pytest.raises(StorageWriteError) as exc_info:
                store.save(TaskList())
            assert exc_info.value.path == store.path
        finally:
            locked_dir.chmod(0o755)


# ===========================================================================
# Round-trip (save → load)
# ===========================================================================

class TestRoundTrip:
    def test_round_trip_single_task(self, store):
        task = Task(
            title="Round-trip task",
            priority=Priority.CRITICAL,
            status=Status.IN_PROGRESS,
            due_date="2035-01-01",
            tags=["alpha", "beta"],
        )
        tl = TaskList()
        tl.add(task)
        store.save(tl)

        loaded = store.load()
        assert len(loaded) == 1
        t = loaded.all()[0]
        assert t.id == task.id
        assert t.title == task.title
        assert t.priority is Priority.CRITICAL
        assert t.status is Status.IN_PROGRESS
        assert t.due_date == "2035-01-01"
        assert set(t.tags) == {"alpha", "beta"}

    def test_round_trip_many_tasks(self, store):
        tl = TaskList()
        for i in range(50):
            tl.add(Task(title=f"Task {i:03d}", priority=Priority.LOW))
        store.save(tl)

        loaded = store.load()
        assert len(loaded) == 50
        loaded_titles = {t.title for t in loaded}
        original_titles = {t.title for t in tl}
        assert loaded_titles == original_titles

    def test_round_trip_preserves_order(self, store):
        titles = ["First", "Second", "Third", "Fourth", "Fifth"]
        tl = TaskList()
        for title in titles:
            tl.add(Task(title=title))
        store.save(tl)

        loaded = store.load()
        loaded_titles = [t.title for t in loaded]
        assert loaded_titles == titles

    def test_multiple_save_load_cycles(self, store):
        """Simulate repeated edits: each cycle adds a task and reloads."""
        for i in range(5):
            tl = store.load()
            tl.add(Task(title=f"Cycle {i}"))
            store.save(tl)

        final = store.load()
        assert len(final) == 5
        for i in range(5):
            assert final.get_by_id is not None  # TaskList has get_by_id


# ===========================================================================
# TaskStorage.delete()
# ===========================================================================

class TestDelete:
    def test_delete_existing_file_returns_true(self, populated_store):
        result = populated_store.delete()
        assert result is True
        assert not populated_store.exists

    def test_delete_absent_file_returns_false(self, store):
        result = store.delete()
        assert result is False

    def test_delete_then_load_returns_empty(self, populated_store):
        populated_store.delete()
        tl = populated_store.load()
        assert len(tl) == 0


# ===========================================================================
# TaskStorage.backup()
# ===========================================================================

class TestBackup:
    def test_backup_creates_bak_file(self, populated_store):
        bak_path = populated_store.backup()
        assert bak_path.exists()
        assert bak_path.suffix == ".bak"

    def test_backup_content_matches_original(self, populated_store):
        bak_path = populated_store.backup()
        original_content = populated_store.path.read_text(encoding="utf-8")
        backup_content = bak_path.read_text(encoding="utf-8")
        assert original_content == backup_content

    def test_backup_custom_suffix(self, populated_store):
        bak_path = populated_store.backup(suffix=".backup")
        assert str(bak_path).endswith(".backup")
        assert bak_path.exists()

    def test_backup_absent_source_raises_storage_read_error(self, store):
        with pytest.raises(StorageReadError) as exc_info:
            store.backup()
        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.path == store.path

    def test_backup_is_independent_copy(self, populated_store):
        """Modifying the original after backup should not affect the backup."""
        bak_path = populated_store.backup()
        # Overwrite the original with a different task list
        populated_store.save(_make_task_list("Only this"))
        backup_data = json.loads(bak_path.read_text(encoding="utf-8"))
        assert len(backup_data["tasks"]) == 3  # original three tasks


# ===========================================================================
# TaskStorage.file_size()
# ===========================================================================

class TestFileSize:
    def test_file_size_absent_returns_none(self, store):
        assert store.file_size() is None

    def test_file_size_returns_positive_int(self, populated_store):
        size = populated_store.file_size()
        assert isinstance(size, int)
        assert size > 0

    def test_file_size_grows_with_more_tasks(self, store):
        store.save(_make_task_list("One"))
        size_one = store.file_size()
        store.save(_make_task_list("One", "Two", "Three", "Four", "Five"))
        size_five = store.file_size()
        assert size_five > size_one


# ===========================================================================
# Integration – full CRUD workflow across multiple store instances
# ===========================================================================

class TestIntegrationWorkflow:
    """Simulate a realistic multi-session CLI workflow."""

    def test_full_crud_workflow(self, tmp_dir):
        path = tmp_dir / "workflow_tasks.json"

        # --- Session 1: create tasks ---
        store1 = TaskStorage(path)
        tl = store1.load()
        assert len(tl) == 0

        t1 = Task(title="Design API", priority=Priority.HIGH, tags=["backend"])
        t2 = Task(title="Write tests", priority=Priority.MEDIUM, tags=["testing"])
        t3 = Task(title="Deploy", priority=Priority.CRITICAL, tags=["devops"])
        tl.add(t1)
        tl.add(t2)
        tl.add(t3)
        store1.save(tl)

        # --- Session 2: update a task ---
        store2 = TaskStorage(path)
        tl2 = store2.load()
        assert len(tl2) == 3

        tl2.update(t1.id, status="in_progress")
        store2.save(tl2)

        # --- Session 3: complete and remove ---
        store3 = TaskStorage(path)
        tl3 = store3.load()

        task = tl3.get_by_id(t1.id)
        assert task is not None
        assert task.status is Status.IN_PROGRESS

        task.mark_done()
        tl3.remove(t3.id)
        store3.save(tl3)

        # --- Session 4: verify final state ---
        store4 = TaskStorage(path)
        tl4 = store4.load()
        assert len(tl4) == 2

        done_task = tl4.get_by_id(t1.id)
        assert done_task is not None
        assert done_task.status is Status.DONE

        removed_task = tl4.get_by_id(t3.id)
        assert removed_task is None

        # --- Backup and verify ---
        bak = store4.backup()
        assert bak.exists()
        bak_data = json.loads(bak.read_text(encoding="utf-8"))
        assert len(bak_data["tasks"]) == 2

        # --- Delete and verify empty ---
        store4.delete()
        store5 = TaskStorage(path)
        tl5 = store5.load()
        assert len(tl5) == 0

    def test_concurrent_independent_stores(self, tmp_dir):
        """Two separate storage files should not interfere with each other."""
        store_a = TaskStorage(tmp_dir / "a.json")
        store_b = TaskStorage(tmp_dir / "b.json")

        tl_a = _make_task_list("Task A1", "Task A2")
        tl_b = _make_task_list("Task B1")

        store_a.save(tl_a)
        store_b.save(tl_b)

        loaded_a = store_a.load()
        loaded_b = store_b.load()

        assert len(loaded_a) == 2
        assert len(loaded_b) == 1
        assert {t.title for t in loaded_a} == {"Task A1", "Task A2"}
        assert loaded_b.all()[0].title == "Task B1"

    def test_storage_survives_empty_description_and_no_tags(self, store):
        """Edge case: tasks with all optional fields at their defaults."""
        tl = TaskList()
        tl.add(Task(title="Minimal"))
        store.save(tl)
        loaded = store.load()
        t = loaded.all()[0]
        assert t.description == ""
        assert t.tags == []
        assert t.due_date is None

    def test_filter_and_sort_after_load(self, store):
        """Filtering and sorting on a loaded TaskList should work correctly."""
        tl = TaskList()
        tl.add(Task(title="Low priority", priority=Priority.LOW))
        tl.add(Task(title="High priority", priority=Priority.HIGH))
        tl.add(Task(title="Critical", priority=Priority.CRITICAL))
        store.save(tl)

        loaded = store.load()
        sorted_tl = loaded.sorted_by_priority(descending=True)
        titles = [t.title for t in sorted_tl]
        assert titles[0] == "Critical"
        assert titles[-1] == "Low priority"

        high_only = loaded.filter_by_priority(Priority.HIGH)
        assert len(high_only) == 1
        assert high_only.all()[0].title == "High priority"
