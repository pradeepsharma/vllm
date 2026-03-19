"""Business logic layer for the Task Management CLI Tool.

This module sits between the raw :class:`~storage.Storage` layer and the
CLI/API presentation layer.  It encapsulates all domain rules that go beyond
simple CRUD, such as:

* Bulk status transitions (e.g. complete all tasks in a project).
* Formatted summary reports for display.
* Moving tasks between projects.
* Searching tasks by keyword across title and description.
* Archiving (marking all DONE tasks) and purging completed tasks.
* Aggregated project health metrics.

Typical usage::

    from storage import Storage
    from services import TaskService, ProjectService

    store = Storage()
    task_svc = TaskService(store)
    proj_svc = ProjectService(store)

    # Create a project and some tasks
    project = proj_svc.create("Sprint 1", description="First sprint")
    task_svc.add("Implement login", project_id=project.id, priority="high")
    task_svc.add("Write tests", project_id=project.id)

    # Get a formatted summary
    summary = proj_svc.summary(project.id)
    print(summary["completion_pct"])  # e.g. 0.0

    # Complete all tasks in the project
    completed = task_svc.complete_all(project_id=project.id)
    print(len(completed))  # 2
"""
from __future__ import annotations

from typing import Dict, List, Optional

from models import Priority, Project, Status, Task
from storage import NotFoundError, Storage, ValidationError

# ---------------------------------------------------------------------------
# Re-export storage exceptions so callers only need to import from services
# ---------------------------------------------------------------------------
__all__ = [
    "TaskService",
    "ProjectService",
    "ServiceError",
    "NotFoundError",
    "ValidationError",
]


class ServiceError(Exception):
    """Base class for service-layer errors that don't map to storage errors."""


# ---------------------------------------------------------------------------
# TaskService
# ---------------------------------------------------------------------------


class TaskService:
    """High-level operations on :class:`~models.Task` objects.

    Parameters
    ----------
    storage:
        A :class:`~storage.Storage` instance to delegate persistence to.
    """

    def __init__(self, storage: Storage) -> None:
        self._store = storage

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------

    def add(
        self,
        title: str,
        *,
        description: str = "",
        status: str | Status = Status.TODO,
        priority: str | Priority = Priority.MEDIUM,
        due_date: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Task:
        """Create a new task, accepting enum values *or* their string names.

        This is a thin convenience wrapper around
        :meth:`~storage.Storage.create_task` that coerces string arguments to
        the appropriate enum types before delegating.

        Parameters
        ----------
        title:
            Short human-readable summary.  Must be non-empty.
        description:
            Optional longer description.
        status:
            Initial lifecycle status.  Accepts a :class:`~models.Status`
            instance or a string value such as ``"todo"``, ``"in_progress"``,
            or ``"done"``.  Defaults to ``Status.TODO``.
        priority:
            Urgency level.  Accepts a :class:`~models.Priority` instance or a
            string value such as ``"low"``, ``"medium"``, ``"high"``, or
            ``"critical"``.  Defaults to ``Priority.MEDIUM``.
        due_date:
            Optional ISO-8601 date string ``"YYYY-MM-DD"``.
        project_id:
            Optional id of an existing :class:`~models.Project`.

        Returns
        -------
        Task
            The newly created and persisted task.

        Raises
        ------
        ValidationError
            If *title* is empty, *status*/*priority* are invalid strings,
            *due_date* has an invalid format, or *project_id* does not exist.
        """
        status = self._coerce_status(status)
        priority = self._coerce_priority(priority)
        return self._store.create_task(
            title,
            description=description,
            status=status,
            priority=priority,
            due_date=due_date,
            project_id=project_id,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> Task:
        """Return the task with *task_id*.

        Raises
        ------
        NotFoundError
            If no task with that id exists.
        """
        return self._store.get_task(task_id)

    def list_all(
        self,
        *,
        project_id: Optional[str] = None,
        status: Optional[str | Status] = None,
        priority: Optional[str | Priority] = None,
        overdue_only: bool = False,
    ) -> List[Task]:
        """Return tasks, optionally filtered.

        Accepts enum instances *or* their string values for *status* and
        *priority*.

        Parameters
        ----------
        project_id:
            If given, only return tasks belonging to this project.
        status:
            If given, only return tasks with this status.
        priority:
            If given, only return tasks with this priority.
        overdue_only:
            If ``True``, only return tasks that are overdue.

        Returns
        -------
        list[Task]
            Matching tasks ordered by creation time (oldest first).
        """
        status_enum = self._coerce_status(status) if status is not None else None
        priority_enum = (
            self._coerce_priority(priority) if priority is not None else None
        )
        return self._store.list_tasks(
            project_id=project_id,
            status=status_enum,
            priority=priority_enum,
            overdue_only=overdue_only,
        )

    def search(self, keyword: str, *, project_id: Optional[str] = None) -> List[Task]:
        """Return tasks whose title or description contains *keyword*.

        The search is case-insensitive.  An empty *keyword* returns all tasks
        (optionally filtered by *project_id*).

        Parameters
        ----------
        keyword:
            Substring to search for in task title and description.
        project_id:
            If given, restrict the search to tasks in this project.

        Returns
        -------
        list[Task]
            Matching tasks ordered by creation time (oldest first).
        """
        tasks = self._store.list_tasks(project_id=project_id)
        if not keyword:
            return tasks
        needle = keyword.lower()
        return [
            t
            for t in tasks
            if needle in t.title.lower() or needle in t.description.lower()
        ]

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update(
        self,
        task_id: str,
        *,
        title: object = None,
        description: object = None,
        status: object = None,
        priority: object = None,
        due_date: object = None,
        project_id: object = None,
    ) -> Task:
        """Update a task's fields, coercing string enum values automatically.

        Only fields that are explicitly passed as non-``None`` are changed,
        *except* for ``due_date`` and ``project_id`` which can be explicitly
        set to ``None`` to clear them.  To distinguish "not passed" from
        "passed as None" for those two fields, use the sentinel-aware
        :meth:`~storage.Storage.update_task` directly.

        Parameters
        ----------
        task_id:
            Id of the task to update.
        title:
            New title.  Must be non-empty if supplied.
        description:
            New description.
        status:
            New lifecycle status (string or :class:`~models.Status`).
        priority:
            New urgency level (string or :class:`~models.Priority`).
        due_date:
            New due date (``"YYYY-MM-DD"``), or ``None`` to clear it.
        project_id:
            New project id, or ``None`` to unlink from any project.

        Returns
        -------
        Task
            The updated task.

        Raises
        ------
        NotFoundError
            If no task with *task_id* exists.
        ValidationError
            If any supplied value fails validation.
        """
        from storage import _UNSET  # noqa: PLC0415 (local import to avoid circular)

        kwargs: dict = {}
        if title is not None:
            kwargs["title"] = title
        if description is not None:
            kwargs["description"] = description
        if status is not None:
            kwargs["status"] = self._coerce_status(status)
        if priority is not None:
            kwargs["priority"] = self._coerce_priority(priority)
        # due_date and project_id can be explicitly None to clear them.
        # We use _UNSET as the default in storage.update_task, so we only
        # pass them when the caller explicitly provided a value.
        if due_date is not _UNSET:
            kwargs["due_date"] = due_date
        if project_id is not _UNSET:
            kwargs["project_id"] = project_id

        return self._store.update_task(task_id, **kwargs)

    def complete(self, task_id: str) -> Task:
        """Mark a single task as :attr:`~models.Status.DONE`.

        Parameters
        ----------
        task_id:
            Id of the task to complete.

        Returns
        -------
        Task
            The updated task with ``status == Status.DONE``.

        Raises
        ------
        NotFoundError
            If no task with *task_id* exists.
        """
        return self._store.update_task(task_id, status=Status.DONE)

    def start(self, task_id: str) -> Task:
        """Mark a single task as :attr:`~models.Status.IN_PROGRESS`.

        Parameters
        ----------
        task_id:
            Id of the task to start.

        Returns
        -------
        Task
            The updated task with ``status == Status.IN_PROGRESS``.

        Raises
        ------
        NotFoundError
            If no task with *task_id* exists.
        """
        return self._store.update_task(task_id, status=Status.IN_PROGRESS)

    def reopen(self, task_id: str) -> Task:
        """Reset a task's status back to :attr:`~models.Status.TODO`.

        Useful for re-opening a task that was previously completed or started.

        Parameters
        ----------
        task_id:
            Id of the task to reopen.

        Returns
        -------
        Task
            The updated task with ``status == Status.TODO``.

        Raises
        ------
        NotFoundError
            If no task with *task_id* exists.
        """
        return self._store.update_task(task_id, status=Status.TODO)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def complete_all(self, *, project_id: Optional[str] = None) -> List[Task]:
        """Mark all non-DONE tasks as DONE.

        Parameters
        ----------
        project_id:
            If given, only complete tasks belonging to this project.

        Returns
        -------
        list[Task]
            The tasks that were updated (previously non-DONE tasks).

        Raises
        ------
        NotFoundError
            If *project_id* is given but does not exist.
        """
        if project_id is not None:
            # Validate the project exists first.
            self._store.get_project(project_id)

        pending = self._store.list_tasks(
            project_id=project_id,
            status=None,
        )
        updated: List[Task] = []
        for task in pending:
            if task.status != Status.DONE:
                updated.append(self._store.update_task(task.id, status=Status.DONE))
        return updated

    def purge_completed(self, *, project_id: Optional[str] = None) -> int:
        """Delete all DONE tasks and return the count of deleted tasks.

        Parameters
        ----------
        project_id:
            If given, only purge DONE tasks belonging to this project.

        Returns
        -------
        int
            Number of tasks deleted.

        Raises
        ------
        NotFoundError
            If *project_id* is given but does not exist.
        """
        if project_id is not None:
            self._store.get_project(project_id)

        done_tasks = self._store.list_tasks(
            project_id=project_id,
            status=Status.DONE,
        )
        for task in done_tasks:
            self._store.delete_task(task.id)
        return len(done_tasks)

    def move_to_project(
        self,
        task_id: str,
        new_project_id: Optional[str],
    ) -> Task:
        """Move a task to a different project (or unlink it from any project).

        Parameters
        ----------
        task_id:
            Id of the task to move.
        new_project_id:
            Id of the destination project, or ``None`` to unlink the task
            from its current project.

        Returns
        -------
        Task
            The updated task.

        Raises
        ------
        NotFoundError
            If *task_id* does not exist.
        ValidationError
            If *new_project_id* is given but does not refer to an existing
            project.
        """
        return self._store.update_task(task_id, project_id=new_project_id)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def remove(self, task_id: str) -> None:
        """Delete a task by id.

        Raises
        ------
        NotFoundError
            If no task with *task_id* exists.
        """
        self._store.delete_task(task_id)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def overdue_report(self) -> List[Dict]:
        """Return a list of dicts describing all overdue tasks.

        Each dict has the shape::

            {
                "id":          str,
                "title":       str,
                "due_date":    str,          # "YYYY-MM-DD"
                "priority":    str,          # e.g. "high"
                "project_id":  str | None,
            }

        Returns
        -------
        list[dict]
            Overdue tasks ordered by due date (earliest first).
        """
        overdue = self._store.list_tasks(overdue_only=True)
        overdue_sorted = sorted(overdue, key=lambda t: t.due_date or "")
        return [
            {
                "id": t.id,
                "title": t.title,
                "due_date": t.due_date,
                "priority": t.priority.value,
                "project_id": t.project_id,
            }
            for t in overdue_sorted
        ]

    def format_task(self, task: Task) -> str:
        """Return a single-line human-readable representation of *task*.

        Format::

            [STATUS] TITLE (priority: PRIORITY) [due: DUE_DATE] [overdue!]

        The ``[due: ...]`` and ``[overdue!]`` parts are omitted when not
        applicable.

        Parameters
        ----------
        task:
            The task to format.

        Returns
        -------
        str
            A single-line string suitable for CLI output.
        """
        parts = [f"[{task.status.value.upper()}]", task.title]
        parts.append(f"(priority: {task.priority.value})")
        if task.due_date:
            parts.append(f"[due: {task.due_date}]")
        if task.is_overdue():
            parts.append("[overdue!]")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_status(value: str | Status) -> Status:
        """Convert *value* to a :class:`~models.Status` enum member.

        Raises
        ------
        ValidationError
            If *value* is a string that does not match any Status value.
        """
        if isinstance(value, Status):
            return value
        try:
            return Status(value)
        except ValueError:
            valid = ", ".join(s.value for s in Status)
            raise ValidationError(
                f"Invalid status '{value}'. Valid values: {valid}."
            ) from None

    @staticmethod
    def _coerce_priority(value: str | Priority) -> Priority:
        """Convert *value* to a :class:`~models.Priority` enum member.

        Raises
        ------
        ValidationError
            If *value* is a string that does not match any Priority value.
        """
        if isinstance(value, Priority):
            return value
        try:
            return Priority(value)
        except ValueError:
            valid = ", ".join(p.value for p in Priority)
            raise ValidationError(
                f"Invalid priority '{value}'. Valid values: {valid}."
            ) from None


# ---------------------------------------------------------------------------
# ProjectService
# ---------------------------------------------------------------------------


class ProjectService:
    """High-level operations on :class:`~models.Project` objects.

    Parameters
    ----------
    storage:
        A :class:`~storage.Storage` instance to delegate persistence to.
    """

    def __init__(self, storage: Storage) -> None:
        self._store = storage

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------

    def create(self, name: str, *, description: str = "") -> Project:
        """Create and persist a new project.

        Parameters
        ----------
        name:
            Human-readable project name.  Must be non-empty.
        description:
            Optional longer description.

        Returns
        -------
        Project
            The newly created project.

        Raises
        ------
        ValidationError
            If *name* is empty or not a string.
        """
        return self._store.create_project(name, description=description)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get(self, project_id: str) -> Project:
        """Return the project with *project_id*.

        Raises
        ------
        NotFoundError
            If no project with that id exists.
        """
        return self._store.get_project(project_id)

    def list_all(self) -> List[Project]:
        """Return all projects ordered by creation time (oldest first)."""
        return self._store.list_projects()

    def find_by_name(self, name: str) -> List[Project]:
        """Return projects whose name contains *name* (case-insensitive).

        Parameters
        ----------
        name:
            Substring to search for in project names.

        Returns
        -------
        list[Project]
            Matching projects ordered by creation time (oldest first).
        """
        needle = name.lower()
        return [p for p in self._store.list_projects() if needle in p.name.lower()]

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def rename(self, project_id: str, new_name: str) -> Project:
        """Rename a project.

        Parameters
        ----------
        project_id:
            Id of the project to rename.
        new_name:
            New name.  Must be non-empty.

        Returns
        -------
        Project
            The updated project.

        Raises
        ------
        NotFoundError
            If no project with *project_id* exists.
        ValidationError
            If *new_name* is empty.
        """
        return self._store.update_project(project_id, name=new_name)

    def update_description(self, project_id: str, description: str) -> Project:
        """Update a project's description.

        Parameters
        ----------
        project_id:
            Id of the project to update.
        description:
            New description (may be empty string to clear it).

        Returns
        -------
        Project
            The updated project.

        Raises
        ------
        NotFoundError
            If no project with *project_id* exists.
        """
        return self._store.update_project(project_id, description=description)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def remove(self, project_id: str, *, cascade: bool = False) -> None:
        """Delete a project.

        Parameters
        ----------
        project_id:
            Id of the project to delete.
        cascade:
            If ``True``, also delete all tasks that belong to this project.
            If ``False`` (default), tasks are left intact but unlinked.

        Raises
        ------
        NotFoundError
            If no project with *project_id* exists.
        """
        self._store.delete_project(project_id, cascade=cascade)

    # ------------------------------------------------------------------
    # Reporting / metrics
    # ------------------------------------------------------------------

    def summary(self, project_id: str) -> Dict:
        """Return a health-metrics dict for a single project.

        The returned dict has the shape::

            {
                "project_id":     str,
                "name":           str,
                "description":    str,
                "total_tasks":    int,
                "done":           int,
                "in_progress":    int,
                "todo":           int,
                "overdue":        int,
                "completion_pct": float,   # 0.0 – 100.0
            }

        ``completion_pct`` is ``0.0`` when there are no tasks.

        Parameters
        ----------
        project_id:
            Id of the project to summarise.

        Returns
        -------
        dict
            Health metrics for the project.

        Raises
        ------
        NotFoundError
            If no project with *project_id* exists.
        """
        project = self._store.get_project(project_id)
        tasks = self._store.list_tasks(project_id=project_id)

        total = len(tasks)
        done = sum(1 for t in tasks if t.status == Status.DONE)
        in_progress = sum(1 for t in tasks if t.status == Status.IN_PROGRESS)
        todo = sum(1 for t in tasks if t.status == Status.TODO)
        overdue = sum(1 for t in tasks if t.is_overdue())
        completion_pct = (done / total * 100.0) if total > 0 else 0.0

        return {
            "project_id": project.id,
            "name": project.name,
            "description": project.description,
            "total_tasks": total,
            "done": done,
            "in_progress": in_progress,
            "todo": todo,
            "overdue": overdue,
            "completion_pct": round(completion_pct, 2),
        }

    def all_summaries(self) -> List[Dict]:
        """Return :meth:`summary` dicts for every project.

        Returns
        -------
        list[dict]
            One summary dict per project, ordered by project creation time
            (oldest first).
        """
        return [self.summary(p.id) for p in self._store.list_projects()]

    def format_project(self, project: Project) -> str:
        """Return a single-line human-readable representation of *project*.

        Format::

            PROJECT_NAME [id: PROJECT_ID]

        Parameters
        ----------
        project:
            The project to format.

        Returns
        -------
        str
            A single-line string suitable for CLI output.
        """
        return f"{project.name} [id: {project.id}]"
