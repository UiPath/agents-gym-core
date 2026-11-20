import copy
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import tool


class AccountRating(StrEnum):
    HOT = "Hot"
    WARM = "Warm"
    COLD = "Cold"


class AccountType(StrEnum):
    PROSPECT = "Prospect"
    CUSTOMER_DIRECT = "Customer - Direct"
    CUSTOMER_CHANNEL = "Customer - Channel"
    CHANNEL_PARTNER_RESELLER = "Channel Partner  Reseller"
    INSTALLATION_PARTNER = "Installation Partner"
    TECHNOLOGY_PARTNER = "Technology Partner"
    OTHER = "Other"


class Industry(StrEnum):
    AGRICULTURE = "Agriculture"
    APPAREL = "Apparel"
    BANKING = "Banking"
    BIOTECHNOLOGY = "Biotechnology"
    CHEMICALS = "Chemicals"
    COMMUNICATIONS = "Communications"
    CONSTRUCTION = "Construction"
    CONSULTING = "Consulting"
    EDUCATION = "Education"
    ELECTRONICS = "Electronics"
    ENERGY = "Energy"
    ENGINEERING = "Engineering"
    ENTERTAINMENT = "Entertainment"
    ENVIRONMENTAL = "Environmental"
    FINANCE = "Finance"
    FOOD_BEVERAGE = "Food & Beverage"
    GOVERNMENT = "Government"
    HEALTHCARE = "Healthcare"
    HOSPITALITY = "Hospitality"
    INSURANCE = "Insurance"
    MACHINERY = "Machinery"
    MANUFACTURING = "Manufacturing"
    MEDIA = "Media"
    NOT_FOR_PROFIT = "Not For Profit"
    RECREATION = "Recreation"
    RETAIL = "Retail"
    SHIPPING = "Shipping"
    TECHNOLOGY = "Technology"
    TELECOMMUNICATIONS = "Telecommunications"
    TRANSPORTATION = "Transportation"
    UTILITIES = "Utilities"
    OTHER = "Other"
    NONE = "None"


class OpportunityStage(StrEnum):
    PROSPECTING = "Prospecting"
    QUALIFICATION = "Qualification"
    PROPOSAL_PRICE_QUOTE = "ProposalPrice Quote"
    VALUE_PROPOSITION = "Value Proposition"
    CLOSED_LOST = "Closed Lost"
    CLOSED_WON = "Closed Won"
    ID_DECISION_MAKERS = "Id. Decision Makers"
    NEEDS_ANALYSIS = "Needs Analysis"
    NEGOTIATION_REVIEW = "NegotiationReview"
    PERCEPTION_ANALYSIS = "Perception Analysis"


# Records live per agent run: each run (its LangGraph thread) starts from documents/test_data.json
# and sees only its own creations, updates and removals, so runs that share this process,
# concurrently or one after another, cannot affect each other.
_KINDS = ("accounts", "contacts", "opportunities", "leads")
_ID_PREFIX = {"accounts": "001", "contacts": "003", "opportunities": "006", "leads": "00Q"}
_MAX_RUNS = 512


@dataclass
class _RunState:
    records: dict[str, list[dict[str, Any]]]
    created: int = 0


_runs: OrderedDict[str, _RunState] = OrderedDict()
_runs_lock = threading.Lock()


@cache
def _initial_records() -> dict[str, list[dict[str, Any]]]:
    with open(Path(__file__).parent / "documents" / "test_data.json") as f:
        loaded = json.load(f)
    return {kind: loaded.get(kind, []) for kind in _KINDS}


def _run_state() -> _RunState:
    """The state of the current agent run, created from the test data on first use."""
    config = var_child_runnable_config.get(None) or {}
    run = str((config.get("configurable") or {}).get("thread_id", ""))
    with _runs_lock:
        if run not in _runs:
            records = copy.deepcopy(_initial_records())
            _runs[run] = _RunState({**records, "action_center_tasks": []})
            while len(_runs) > _MAX_RUNS:
                _runs.popitem(last=False)
        return _runs[run]


def run_data() -> dict[str, list[dict[str, Any]]]:
    """The records of the current agent run."""
    return _run_state().records


def get_record(record_type: str, record_id: str) -> dict[str, Any] | None:
    """Get a record of the current run by its ID."""
    return next((r for r in run_data()[record_type] if r.get("id") == record_id), None)


def _create(kind: str, fields: dict[str, Any]) -> str:
    """Store a new record built from a create tool's arguments and return its ID."""
    state = _run_state()
    state.created += 1
    record_id = f"{_ID_PREFIX[kind]}5f00000NEW{state.created:03d}AAX"
    values = {k: v for k, v in fields.items() if v is not None}
    state.records[kind].append({"id": record_id, **values})
    return record_id


def _update(kind: str, id_field: str, fields: dict[str, Any]) -> bool:
    """Apply an update tool's non-empty arguments to the record named by ``fields[id_field]``."""
    record = get_record(kind, fields[id_field])
    if record is None:
        return False
    record.update({k: v for k, v in fields.items() if v is not None and k != id_field})
    return True


def _remove(kind: str, record_id: str) -> bool:
    record = get_record(kind, record_id)
    if record is None:
        return False
    run_data()[kind].remove(record)
    return True


@tool
async def create_account(
    account_name: str,
    account_rating: str = "Hot",
    account_type: str = "Prospect",
    account_description: str | None = None,
    annual_revenue: float | None = None,
    employee_count: int | None = None,
    industry_name: str = "None",
) -> str:
    """Create a new account in Salesforce.

    Args:
        account_name: Name of the account
        account_rating: Rating of the account (Hot, Warm, or Cold)
        account_type: Type of the account
        account_description: Description of the account
        annual_revenue: Annual revenue of the account
        employee_count: Number of employees in the account
        industry_name: Industry name of the account

    Returns:
        Confirmation message with the created account ID
    """
    sf_id = _create("accounts", locals())  # first statement: locals() holds only the arguments
    return f"Successfully created account with ID: {sf_id}"


@tool
async def remove_account(account_id: str) -> str:
    """Remove an account from Salesforce.

    Args:
        account_id: ID of the account to remove

    Returns:
        Confirmation message for the removed account
    """
    if _remove("accounts", account_id):
        return f"Successfully removed account with ID: {account_id}"
    return f"Failed to remove account with ID: {account_id}"


@tool
async def update_account(
    account_id: str,
    account_name: str | None = None,
    account_rating: str | None = None,
    account_type: str | None = None,
    annual_revenue: float | None = None,
    employee_count: int | None = None,
    industry_name: str | None = None,
    account_owner_email_address: str | None = None,
) -> str:
    """Update an existing account in Salesforce.

    Args:
        account_id: ID of the account to update
        account_name: Name of the account
        account_rating: Rating of the account (Hot, Warm, or Cold)
        account_type: Type of the account
        annual_revenue: Annual revenue of the account
        employee_count: Number of employees in the account
        industry_name: Industry name of the account
        account_owner_email_address: Email address of the account owner

    Returns:
        Confirmation message for the updated account
    """
    if _update("accounts", "account_id", locals()):  # first statement, as above
        return f"Successfully updated account with ID: {account_id}"
    return f"Failed to update account with ID: {account_id}"


@tool
async def create_contact(
    first_name: str,
    last_name: str,
    email_address: str,
    contact_title: str | None = None,
    phone_number: str | None = None,
    account_name: str | None = None,
    contact_description: str | None = None,
) -> str:
    """Create a new contact in Salesforce.

    Args:
        first_name: First name of the contact
        last_name: Last name of the contact
        email_address: Email address of the contact
        contact_title: Title of the contact
        phone_number: Phone number of the contact
        account_name: Name of the account the contact belongs to
        contact_description: Description of the contact

    Returns:
        Confirmation message with the created contact ID
    """
    sf_id = _create("contacts", locals())  # first statement: locals() holds only the arguments
    return f"Successfully created contact with ID: {sf_id}"


@tool
async def remove_contact(contact_id: str) -> str:
    """Remove a contact from Salesforce.

    Args:
        contact_id: ID of the contact to remove

    Returns:
        Confirmation message for the removed contact
    """
    if _remove("contacts", contact_id):
        return f"Successfully removed contact with ID: {contact_id}"
    return f"Failed to remove contact with ID: {contact_id}"


@tool
async def update_contact(
    contact_id: str,
    phone_number: str | None = None,
    email_address: str | None = None,
    contact_title: str | None = None,
    contact_description: str | None = None,
    contact_owner_email_address: str | None = None,
) -> str:
    """Update an existing contact in Salesforce.

    Args:
        contact_id: ID of the contact to update
        phone_number: Phone number of the contact
        email_address: Email address of the contact
        contact_title: Title of the contact
        contact_description: Description of the contact
        contact_owner_email_address: Email address of the contact owner

    Returns:
        Confirmation message for the updated contact
    """
    if _update("contacts", "contact_id", locals()):  # first statement, as above
        return f"Successfully updated contact with ID: {contact_id}"
    return f"Failed to update contact with ID: {contact_id}"


@tool
async def create_opportunity(
    opportunity_name: str,
    opportunity_close_date: str,
    opportunity_stage: str = "Prospecting",
    opportunity_description: str | None = None,
    account_name: str | None = None,
    opportunity_amount: float | None = None,
    opportunity_probability: float | None = None,
) -> str:
    """Create a new opportunity in Salesforce.

    Args:
        opportunity_name: Name of the opportunity
        opportunity_close_date: Close date of the opportunity (ISO format string)
        opportunity_stage: Stage of the opportunity
        opportunity_description: Description of the opportunity
        account_name: Name of the account the opportunity belongs to
        opportunity_amount: Amount of the opportunity
        opportunity_probability: Probability of the opportunity

    Returns:
        Confirmation message with the created opportunity ID
    """
    sf_id = _create("opportunities", locals())  # first statement: locals() holds only the arguments
    return f"Successfully created opportunity with ID: {sf_id}"


@tool
async def remove_opportunity(opportunity_id: str) -> str:
    """Remove an opportunity from Salesforce.

    Args:
        opportunity_id: ID of the opportunity to remove

    Returns:
        Confirmation message for the removed opportunity
    """
    if _remove("opportunities", opportunity_id):
        return f"Successfully removed opportunity with ID: {opportunity_id}"
    return f"Failed to remove opportunity with ID: {opportunity_id}"


@tool
async def update_opportunity(
    opportunity_id: str,
    opportunity_stage: str | None = None,
    opportunity_name: str | None = None,
    opportunity_probability: float | None = None,
    opportunity_amount: float | None = None,
    opportunity_close_date: str | None = None,
    opportunity_owner_email_address: str | None = None,
) -> str:
    """Update an existing opportunity in Salesforce.

    Args:
        opportunity_id: ID of the opportunity to update
        opportunity_stage: Stage of the opportunity
        opportunity_name: Name of the opportunity
        opportunity_probability: Probability of the opportunity
        opportunity_amount: Amount of the opportunity
        opportunity_close_date: Close date of the opportunity (ISO format string)
        opportunity_owner_email_address: Email address of the opportunity owner

    Returns:
        Confirmation message for the updated opportunity
    """
    if _update("opportunities", "opportunity_id", locals()):  # first statement, as above
        return f"Successfully updated opportunity with ID: {opportunity_id}"
    return f"Failed to update opportunity with ID: {opportunity_id}"


@tool
async def create_lead(
    first_name: str,
    last_name: str,
    company_name: str,
    lead_title: str | None = None,
    lead_status: str | None = None,
    lead_email_address: str | None = None,
    lead_country: str | None = None,
) -> str:
    """Create a new lead in Salesforce.

    Args:
        first_name: First name of the lead
        last_name: Last name of the lead
        company_name: Company name of the lead
        lead_title: Title of the lead
        lead_status: Status of the lead
        lead_email_address: Email address of the lead
        lead_country: Country of the lead

    Returns:
        Confirmation message with the created lead ID
    """
    sf_id = _create("leads", locals())  # first statement: locals() holds only the arguments
    return f"Successfully created lead with ID: {sf_id}"


@tool
async def remove_lead(lead_id: str) -> str:
    """Remove a lead from Salesforce.

    Args:
        lead_id: ID of the lead to remove

    Returns:
        Confirmation message for the removed lead
    """
    if _remove("leads", lead_id):
        return f"Successfully removed lead with ID: {lead_id}"
    return f"Failed to remove lead with ID: {lead_id}"


@tool
async def update_lead(
    lead_id: str,
    company_name: str | None = None,
    lead_email_address: str | None = None,
    lead_industry: str | None = None,
    lead_owner_email_address: str | None = None,
    lead_status: str | None = None,
    lead_title: str | None = None,
    lead_country: str | None = None,
) -> str:
    """Update an existing lead in Salesforce.

    Args:
        lead_id: ID of the lead to update
        company_name: Company name of the lead
        lead_email_address: Email address of the lead
        lead_industry: Industry of the lead
        lead_owner_email_address: Email address of the lead owner
        lead_status: Status of the lead
        lead_title: Job title of the lead
        lead_country: Country of the lead

    Returns:
        Confirmation message for the updated lead
    """
    if _update("leads", "lead_id", locals()):  # first statement, as above
        return f"Successfully updated lead with ID: {lead_id}"
    return f"Failed to update lead with ID: {lead_id}"


@tool
async def retrieve_accounts(account_name: str | None = None) -> list[dict[str, Any]]:
    """Retrieve accounts with optional name matching (contains).

    Args:
        account_name: Optional name to match against. If None, returns all accounts.

    Returns:
        List of account records matching the criteria
    """
    accounts = run_data()["accounts"]
    if not account_name:
        return accounts

    matches = [
        account for account in accounts if account_name.lower() in account["account_name"].lower()
    ]
    return matches if matches else accounts


@tool
async def retrieve_contacts(
    account_name: str | None = None, account_id: str | None = None
) -> list[dict[str, Any]]:
    """Retrieve contacts with optional filtering by account.

    Args:
        account_name: Optional account name to filter by
        account_id: Optional account ID to filter by (takes precedence)

    Returns:
        List of contact records matching the criteria
    """
    contacts = run_data()["contacts"]
    if account_id:
        for account in run_data()["accounts"]:
            if account["id"] == account_id:
                return [c for c in contacts if c.get("account_name") == account["account_name"]]
        return []

    if account_name:
        return [c for c in contacts if c.get("account_name") == account_name]

    return contacts


@tool
async def retrieve_opportunities(
    account_name: str | None = None, account_id: str | None = None
) -> list[dict[str, Any]]:
    """Retrieve opportunities with optional filtering by account.

    Args:
        account_name: Optional account name to filter by
        account_id: Optional account ID to filter by (takes precedence)

    Returns:
        List of opportunity records matching the criteria
    """
    opportunities = run_data()["opportunities"]
    if account_id:
        for account in run_data()["accounts"]:
            if account["id"] == account_id:
                return [
                    o for o in opportunities if o.get("account_name") == account["account_name"]
                ]
        return []

    if account_name:
        return [o for o in opportunities if o.get("account_name") == account_name]

    return opportunities


@tool
async def retrieve_leads(lead_name: str | None = None) -> list[dict[str, Any]]:
    """Retrieve leads with optional name matching (contains).

    Args:
        lead_name: Optional name to match against. If None, returns all leads.

    Returns:
        List of lead records matching the criteria
    """
    leads = run_data()["leads"]
    if not lead_name:
        return leads

    matches = []
    for lead in leads:
        full_name = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()
        if lead_name.lower() in full_name.lower():
            matches.append(lead)

    return matches if matches else leads


@tool
async def perform_escalation(
    task_title: str,
    task_description: str,
    task_priority: str = "Medium",
    assigned_to: str | None = None,
    due_date: str | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Create a task in UiPath Action Center for escalation and return the requested data.

    Args:
        task_title: Title of the action center task
        task_description: Description of the action center task
        task_priority: Priority of the task (Low, Medium, High)
        assigned_to: Email of the person to assign the task to
        due_date: Due date for the task (ISO format string)

    Returns:
        Tuple containing:
        - ActionCenterTaskCreated (bool): Whether the task was successfully created
        - ActionCenterTaskURL (str): URL of the created task, or error message if creation failed
        - RequestedData (Dict): Dictionary containing the requested data
    """
    tasks = run_data()["action_center_tasks"]
    task_id = f"TASK{len(tasks) + 1:04d}"
    try:
        tasks.append({"id": task_id, "title": task_title, "description": task_description})
        task_url = f"https://example.com/actioncenter/tasks/{task_id}"
        mentioned = f"{task_title} {task_description}".lower().replace(" ", "")

        def mentions(company: str) -> bool:
            return company.lower().replace(" ", "") in mentioned

        response_data = {}
        if mentions("Greenfield"):
            response_data = {"opportunity_probability": 75}
        elif mentions("BrightSphere"):
            response_data = {
                "annual_revenue": 5000000,
                "employee_count": 250,
                "account_type": AccountType.CUSTOMER_DIRECT,
                "industry_name": Industry.ENERGY,
                "account_rating": AccountRating.HOT,
                "account_description": "Renewable energy solutions company",
            }
        elif mentions("TechEd"):
            response_data = {
                "contact_title": "Operations Lead",
                "phone_number": "+1-555-123-4567",
                "email_address": "matthew.baker@teched.example",
                "opportunity_name": "TechEd Solutions - Process Automation",
                "opportunity_close_date": (datetime(2025, 5, 1) + timedelta(days=90)).isoformat(),
                "opportunity_stage": OpportunityStage.PROSPECTING,
                "opportunity_description": "Process automation implementation for TechEd Solutions",
                "account_name": "TechEd Solutions",
                "opportunity_amount": 200000,
                "opportunity_probability": 60,
            }
        elif mentions("FusionHealth"):
            response_data = {
                "industry_name": Industry.HEALTHCARE,
                "phone_number": "+1-555-0129",
                "email_address": "alex.martinez@fusionhealth.example",
                "first_name": "Alex",
                "last_name": "Martinez",
                "contact_title": "IT Director",
                "account_name": "FusionHealth",
                "contact_description": "Primary IT contact",
            }
        elif mentions("HarvestCo"):
            response_data = {
                "lead_title": "Chief Operations Officer",
                "lead_status": "Qualified",
                "lead_country": "United States",
            }
        elif mentions("BioPulse"):
            response_data = {
                "response": "Yes, indeed, there are 2 duplicate accounts, First update the account with the proper data and then remove the duplicate one."
            }
        elif mentions("Happy Trails"):
            response_data = {"response": "They are not in the system"}

        return True, task_url, response_data

    except Exception as e:
        return False, f"Failed to create task: {str(e)}", {}
