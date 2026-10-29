"""Offline CLI and argument checks; no credential lookup or paid requests."""

import asyncio
import json
from copy import deepcopy

import pytest

pytest.importorskip("langgraph.checkpoint.sqlite")

from dataverify.graph_checkpoint import GraphResumeError
from dataverify.probe_cases import get_case
from tools import run_graph as cli


def read_report(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_arguments(tmp_path, *extra):
    return [
        "--checkpoint",
        str(tmp_path / "checkpoints.sqlite3"),
        "--output",
        str(tmp_path / "run.json"),
        "--thread",
        "cli-test",
        *extra,
    ]


def test_offline_default_repairs_disclosed_fault_without_environment_or_network(
    tmp_path, monkeypatch
):
    def forbidden():
        pytest.fail("Offline replay must not look up credentials")

    monkeypatch.setattr(cli.QwenConfig, "from_local_environment", forbidden)
    assert cli.main(run_arguments(tmp_path)) == 0
    report = read_report(tmp_path / "run.json")
    assert report["runtime"] == "langgraph"
    assert report["status"] == "completed"
    assert report["track"] == "controlled"
    assert report["model_attempts"] == 2
    assert report["publication"]["status"] == "verified"
    assert report["publication"]["verified_sql"] == get_case("d06").reference_sql
    assert report["real_model_used"] is False
    assert report["http_events"] == []
    assert report["this_run_http_attempts"] == 0
    assert report["budget_start"] is report["budget_end"] is None
    assert report["audit_ledger_complete"] is True


@pytest.mark.parametrize("pause", ["verify", "feedback"])
def test_offline_resume_consumes_only_remaining_scripted_response(tmp_path, pause):
    assert cli.main(run_arguments(tmp_path, "--pause-after", pause)) == 0
    first = read_report(tmp_path / "run.json")
    assert first["status"] == "paused"
    assert first["model_attempts"] == 1
    assert first["publication"]["rows"] is None
    assert cli.main(run_arguments(tmp_path, "--resume")) == 0
    second = read_report(tmp_path / "run.json")
    assert second["status"] == "completed"
    assert second["model_attempts"] == 2
    assert second["model_calls_this_process"] == 1
    assert second["publication"]["status"] == "verified"
    assert len(second["actions"]) == 2


def test_completed_resume_revalidates_without_another_model_call(tmp_path):
    assert cli.main(run_arguments(tmp_path)) == 0
    assert cli.main(run_arguments(tmp_path, "--resume")) == 0
    report = read_report(tmp_path / "run.json")
    assert report["model_calls_this_process"] == 0
    assert report["publication"]["status"] == "verified"
    assert report["model_attempts"] == 2


def test_offline_natural_is_labeled_as_authored_not_model_performance(tmp_path):
    assert cli.main(run_arguments(tmp_path, "--track", "natural")) == 0
    report = read_report(tmp_path / "run.json")
    assert report["model_attempts"] == 1
    assert report["real_model_used"] is False
    assert "authored" in report["evaluation_scope"]


def test_exhausted_attempt_allowance_returns_failure_instead_of_claiming_success(tmp_path):
    assert cli.main(run_arguments(tmp_path, "--max-attempts", "1")) == 1
    report = read_report(tmp_path / "run.json")
    assert report["publication"]["status"] == "unverified"
    assert report["publication"]["rows"] is None


def test_new_run_cannot_replace_an_existing_thread(tmp_path, capsys):
    assert cli.main(run_arguments(tmp_path)) == 0
    capsys.readouterr()
    assert cli.main(run_arguments(tmp_path)) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure == {"status": "error", "error_code": "THREAD_ALREADY_EXISTS"}


def test_missing_resume_thread_has_a_clear_safe_error(tmp_path, capsys):
    assert cli.main(run_arguments(tmp_path, "--resume")) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "THREAD_NOT_FOUND"


def test_resume_rejects_a_changed_case_configuration(tmp_path, capsys):
    assert cli.main(run_arguments(tmp_path, "--pause-after", "verify")) == 0
    capsys.readouterr()
    assert cli.main(run_arguments(tmp_path, "--resume", "--case", "d10")) == 2
    assert (
        json.loads(capsys.readouterr().out)["error_code"] == "TASK_SOURCE_RUNTIME_OR_CONFIG_CHANGED"
    )


@pytest.mark.parametrize(
    "extra,expected",
    [
        (["--live"], "requires --budget-file"),
        (["--live", "--budget-file", "journal.json"], "requires an explicit --track"),
        (["--budget-file", "journal.json"], "only valid with --live"),
        (["--prior-budget", "prior.json"], "only valid with --live"),
        (
            ["--live", "--track", "natural", "--budget-file", "journal.json"],
            "requires --prior-budget",
        ),
        (["--thread", "bad/thread"], "--thread must contain"),
        (["--max-attempts", "4"], "invalid choice"),
        (["--case", "missing"], "invalid choice"),
    ],
)
def test_invalid_live_or_run_arguments_stop_before_credentials(extra, expected, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(extra)
    assert caught.value.code == 2
    assert expected in capsys.readouterr().err


def test_existing_journal_cannot_be_reseeded(tmp_path, capsys):
    journal = tmp_path / "budget.json"
    journal.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--live",
                "--track",
                "controlled",
                "--budget-file",
                str(journal),
                "--prior-budget",
                str(tmp_path / "prior.json"),
            ]
        )
    assert "cannot be reseeded" in capsys.readouterr().err


@pytest.mark.parametrize("amount", ["0", "-1", "5.01", "nan", "inf"])
def test_first_use_money_allowance_must_be_finite_positive_and_bounded(tmp_path, amount, capsys):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--live",
                "--track",
                "natural",
                "--budget-file",
                str(tmp_path / "budget.json"),
                "--new-budget-cny",
                amount,
            ]
        )
    assert "at most CNY 5" in capsys.readouterr().err
    assert not (tmp_path / "budget.json").exists()


def test_first_use_and_prior_ledger_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--new-budget-cny", "1", "--prior-budget", "prior.json"])
    assert "not allowed with argument" in capsys.readouterr().err


def test_first_use_budget_option_is_not_an_offline_side_effect(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--new-budget-cny", "1"])
    assert "only valid with --live" in capsys.readouterr().err


def test_new_user_can_create_and_reuse_a_smaller_durable_allowance_without_a_prior_report(tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--live",
            "--track",
            "natural",
            "--budget-file",
            str(tmp_path / "budget.json"),
            "--new-budget-cny",
            "1",
        ]
    )
    cli._validate_arguments(parser, args)
    with cli.exclusive_run(args.budget_file):
        budget = cli._load_live_budget(args)
        first = budget.snapshot()
    assert first["money_limit_cny"] == 1
    assert first["http_attempt_limit"] == 6
    assert first["http_attempts"] == 0
    args.new_budget_cny = None
    cli._validate_arguments(parser, args)
    with cli.exclusive_run(args.budget_file):
        assert cli._load_live_budget(args).snapshot() == first


def test_existing_journal_cannot_receive_a_new_money_allowance(tmp_path, capsys):
    journal = tmp_path / "budget.json"
    journal.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--live",
                "--track",
                "natural",
                "--budget-file",
                str(journal),
                "--new-budget-cny",
                "5",
            ]
        )
    assert "cannot be reseeded" in capsys.readouterr().err


def test_output_checkpoint_collision_is_rejected_before_sqlite_is_written(tmp_path, capsys):
    path = str(tmp_path / "shared.sqlite3")
    with pytest.raises(SystemExit):
        cli.main(["--checkpoint", path, "--output", path])
    assert "different paths" in capsys.readouterr().err
    assert not (tmp_path / "shared.sqlite3").exists()


def test_live_wrapper_skips_injection_on_resume_without_calling_a_real_model(tmp_path, monkeypatch):
    """Inspect wiring only; the fake runner never invokes either model adapter."""
    args = cli.build_parser().parse_args(
        run_arguments(
            tmp_path,
            "--live",
            "--track",
            "controlled",
            "--budget-file",
            str(tmp_path / "budget.json"),
            "--resume",
        )
    )
    seen = []

    class FakeBudget:
        def snapshot(self):
            return {"stopped_reason": None}

    class FakeLiveModel:
        def __init__(self, config, budget):
            self.config, self.budget = config, budget

    class FakeInjectedModel:
        def __init__(self, delegate, candidate, *, already_injected=False):
            seen.append(already_injected)
            self.delegate = delegate

    async def completed(*_):
        return 1

    async def fake_run(*_):
        return {"status": "wiring_only_no_calls"}

    monkeypatch.setattr(cli, "_completed_attempts", completed)
    monkeypatch.setattr(cli, "_load_live_budget", lambda _: FakeBudget())
    monkeypatch.setattr(cli.QwenConfig, "from_local_environment", lambda: object())
    monkeypatch.setattr(cli, "validate_pricing", lambda _: None)
    monkeypatch.setattr(cli, "QwenGraphModel", FakeLiveModel)
    monkeypatch.setattr(cli, "InjectedFirstGraphModel", FakeInjectedModel)
    monkeypatch.setattr(cli, "_run_with_model", fake_run)
    assert asyncio.run(cli.run_from_args(args))["status"] == "wiring_only_no_calls"
    assert seen == [True]


def test_accounting_requires_each_new_http_attempt_and_complete_usage():
    start = {"http_attempts": 107}
    end = {"http_attempts": 108, "reported_cost_complete": True}
    events = [
        {"attempt": 108, "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
    ]
    assert cli._accounting_complete(events, start, end)
    assert not cli._accounting_complete([], start, end)
    wrong = deepcopy(events)
    wrong[0]["attempt"] = 1
    assert not cli._accounting_complete(wrong, start, end)
    wrong[0]["attempt"] = 108
    wrong[0]["usage"]["total_tokens"] = None
    assert not cli._accounting_complete(wrong, start, end)


def test_non_code_errors_cannot_print_private_paths_or_credentials():
    assert cli._error_code(ValueError("sk-private-key")) == "GRAPH_RUN_FAILED"
    assert cli._error_code(OSError("C:/Users/private/file")) == "GRAPH_RUN_FAILED"
    assert (
        cli._error_code(GraphResumeError("MODEL_CALL_OUTCOME_UNKNOWN"))
        == "MODEL_CALL_OUTCOME_UNKNOWN"
    )
