from rmbench_sim import list_tasks


def test_expected_tasks_are_discoverable() -> None:
    tasks = list_tasks()
    assert "cover_blocks" in tasks
    assert "press_button" in tasks
    assert "_base_task" not in tasks
