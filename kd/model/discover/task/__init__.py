__all__ = ["make_task", "set_task", "Task", "HierarchicalTask", "SequentialTask"]


def __getattr__(name: str):
    if name in __all__:
        from .task import make_task, set_task, Task, HierarchicalTask, SequentialTask
        mapping = {
            "make_task": make_task,
            "set_task": set_task,
            "Task": Task,
            "HierarchicalTask": HierarchicalTask,
            "SequentialTask": SequentialTask,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
