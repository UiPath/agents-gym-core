"""The Salesforce mock keeps each agent run's changes, and only that run sees them."""

import asyncio
from typing import Any

from agents_gym_core.agents.salesforce_agent import tools

BIOPULSE_KEEP, BIOPULSE_DUPLICATE = "0015f00000WEF7FAAX", "0015f00000WEF7GAAX"


def _call(tool: Any, run: str, **args: Any) -> Any:
    return asyncio.run(tool.ainvoke(args, config={"configurable": {"thread_id": run}}))


def _biopulse(run: str) -> list[dict[str, Any]]:
    return _call(tools.retrieve_accounts, run, account_name="BioPulse Solutions")


def test_updates_and_removals_change_the_run_records() -> None:
    _call(tools.update_account, "merge", account_id=BIOPULSE_KEEP, account_rating="Hot")
    assert "Successfully removed" in _call(
        tools.remove_account, "merge", account_id=BIOPULSE_DUPLICATE
    )

    (kept,) = _biopulse("merge")
    assert kept["id"] == BIOPULSE_KEEP and kept["account_rating"] == "Hot"
    assert "Failed" in _call(tools.remove_account, "merge", account_id=BIOPULSE_DUPLICATE)


def test_created_records_can_be_read_and_updated() -> None:
    reply = _call(tools.create_account, "create", account_name="EarthWear", industry_name="Retail")
    new_id = reply.rsplit(" ", 1)[-1]

    assert "Successfully updated" in _call(
        tools.update_account, "create", account_id=new_id, employee_count=12
    )
    (created,) = _call(tools.retrieve_accounts, "create", account_name="EarthWear")
    assert created == {
        "id": new_id,
        "account_name": "EarthWear",
        "account_rating": "Hot",
        "account_type": "Prospect",
        "industry_name": "Retail",
        "employee_count": 12,
    }


def test_runs_do_not_see_each_others_changes() -> None:
    _call(tools.remove_account, "first", account_id=BIOPULSE_DUPLICATE)

    assert len(_biopulse("first")) == 1
    assert len(_biopulse("second")) == 2


def test_an_update_keeps_the_fields_it_does_not_name() -> None:
    opportunity = _call(tools.retrieve_opportunities, "stage")[0]

    _call(
        tools.update_opportunity, "stage", opportunity_id=opportunity["id"], opportunity_amount=1.0
    )

    (after,) = [
        o for o in _call(tools.retrieve_opportunities, "stage") if o["id"] == opportunity["id"]
    ]
    assert after["opportunity_stage"] == opportunity["opportunity_stage"]
    assert after["opportunity_amount"] == 1.0
