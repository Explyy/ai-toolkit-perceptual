import json

from training_automation.unified_supervisor import supervise


def test_supervisor_records_real_result_and_polls_again(tmp_path):
    calls = []

    def run(path, env):
        calls.append(path)
        return {"status": "idle", "pending": []}

    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    status = tmp_path / "status.json"
    try:
        supervise(
            tmp_path / "config.yaml",
            env={"TRAINING_UNIFIED_STATUS": str(status), "TRAINING_UNIFIED_POLL_SECONDS": "2"},
            run=run, sleep=sleep,
        )
    except KeyboardInterrupt:
        pass
    assert len(calls) == 2
    assert json.loads(status.read_text())["status"] == "idle"


def test_supervisor_holds_error_without_simulating_success(tmp_path):
    status = tmp_path / "status.json"

    def fail(path, env):
        raise RuntimeError("trainer failed")

    code = supervise(
        tmp_path / "config.yaml",
        env={"TRAINING_UNIFIED_STATUS": str(status), "TRAINING_UNIFIED_POLL_SECONDS": "2"},
        run=fail, once=True,
    )
    document = json.loads(status.read_text())
    assert code == 1
    assert document["status"] == "held"
    assert "trainer failed" in document["error"]
