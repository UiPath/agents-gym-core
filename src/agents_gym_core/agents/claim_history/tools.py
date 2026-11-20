from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ClaimHistoryLookupOutput(BaseModel):
    """Output schema for claim history lookup."""

    AppealTips: list[str] = Field(description="Appeal Tips")
    ErrorMessage: str = Field(description="Error Message")


def get_claim_history_output_schema() -> type[BaseModel]:
    """Return the output schema class for Hydra instantiation."""
    return ClaimHistoryLookupOutput


class ClaimHistoryLookupMatchResult(BaseModel):
    Member_ID: str
    Claim_Date: str
    Denial_Code: str
    Denial_Description: str
    Remittance_Code: str
    Remittance_Description: str
    Claim_Amount: float
    Paid_Amount: float
    Payer_Name: str
    Claim_Status: str
    Notes: str


sample_claim_history_lookup_match_results: list[ClaimHistoryLookupMatchResult] = [
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00052",
        Claim_Date="2024-02-03",
        Denial_Code="CO-50",
        Denial_Description="These are non-covered services because this is not deemed a 'medical necessity' by the payer",
        Remittance_Code="M54",
        Remittance_Description="Missing/incomplete/invalid place of service",
        Claim_Amount=7222.22,
        Paid_Amount=6866.03,
        Payer_Name="Aetna",
        Claim_Status="Overturned",
        Notes="Provided medical necessity documentation and corrected place of service.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00124",
        Claim_Date="2024-01-15",
        Denial_Code="CO-16",
        Denial_Description="Claim/service lacks information or has submission/billing error(s)",
        Remittance_Code="M51",
        Remittance_Description="Missing/incomplete/invalid procedure codes",
        Claim_Amount=1250.75,
        Paid_Amount=850.25,
        Payer_Name="UnitedHealthcare",
        Claim_Status="Overturned",
        Notes="Corrected procedure codes and resubmitted with supporting documentation.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00078",
        Claim_Date="2024-02-22",
        Denial_Code="CO-97",
        Denial_Description="The benefit for this service is included in the payment/allowance for another service/procedure",
        Remittance_Code="N95",
        Remittance_Description="This provider type/specialty may not bill this service",
        Claim_Amount=450.00,
        Paid_Amount=325.50,
        Payer_Name="Cigna",
        Claim_Status="Overturned",
        Notes="Clarified benefit inclusion and provided coverage documentation.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00103",
        Claim_Date="2024-03-05",
        Denial_Code="CO-22",
        Denial_Description="This care may be covered by another payer per coordination of benefits",
        Remittance_Code="N30",
        Remittance_Description="Patient ineligible for this service",
        Claim_Amount=3500.50,
        Paid_Amount=2800.25,
        Payer_Name="Blue Cross Blue Shield",
        Claim_Status="Overturned",
        Notes="Verified patient eligibility and submitted coordination of benefits information.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00091",
        Claim_Date="2024-01-30",
        Denial_Code="CO-29",
        Denial_Description="The time limit for filing has expired",
        Remittance_Code="M52",
        Remittance_Description="Missing/incomplete/invalid 'from' date(s) of service",
        Claim_Amount=5678.90,
        Paid_Amount=4325.75,
        Payer_Name="Humana",
        Claim_Status="Overturned",
        Notes="Appealed time limit expiration with proof of timely filing and corrected date of service.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00145",
        Claim_Date="2024-02-18",
        Denial_Code="PR-96",
        Denial_Description="Non-covered charge(s)",
        Remittance_Code="N130",
        Remittance_Description="Consult plan benefit documents/guidelines for information about restrictions for this service",
        Claim_Amount=875.33,
        Paid_Amount=735.25,
        Payer_Name="Anthem",
        Claim_Status="Overturned",
        Notes="Provided plan benefit documents showing service coverage.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00067",
        Claim_Date="2024-03-12",
        Denial_Code="CO-45",
        Denial_Description="Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement",
        Remittance_Code="M53",
        Remittance_Description="Missing/incomplete/invalid days or units of service",
        Claim_Amount=12450.75,
        Paid_Amount=9750.80,
        Payer_Name="Medicare",
        Claim_Status="Overturned",
        Notes="Corrected date of service errors and resubmitted claim.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00112",
        Claim_Date="2024-01-25",
        Denial_Code="CO-16",
        Denial_Description="Claim/service lacks information or has submission/billing error(s)",
        Remittance_Code="M51",
        Remittance_Description="Missing/incomplete/invalid procedure codes",
        Claim_Amount=2125.50,
        Paid_Amount=1875.45,
        Payer_Name="Molina Healthcare",
        Claim_Status="Overturned",
        Notes="Corrected procedure codes and resubmitted with documentation.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00039",
        Claim_Date="2024-02-28",
        Denial_Code="CO-252",
        Denial_Description="An attachment/other documentation is required to adjudicate this claim/service",
        Remittance_Code="N26",
        Remittance_Description="Missing itemized bill/statement",
        Claim_Amount=2345.67,
        Paid_Amount=2100.50,
        Payer_Name="Kaiser Permanente",
        Claim_Status="Overturned",
        Notes="Provided itemized bill and additional documentation as requested.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00083",
        Claim_Date="2024-03-01",
        Denial_Code="CO-45",
        Denial_Description="Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement",
        Remittance_Code="M53",
        Remittance_Description="Missing/incomplete/invalid days or units of service",
        Claim_Amount=8975.25,
        Paid_Amount=7856.44,
        Payer_Name="Centene",
        Claim_Status="Overturned",
        Notes="Corrected date of service information and resubmitted claim.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00156",
        Claim_Date="2024-01-22",
        Denial_Code="CO-16",
        Denial_Description="Claim/service lacks information or has submission/billing error(s)",
        Remittance_Code="M51",
        Remittance_Description="Missing/incomplete/invalid procedure codes",
        Claim_Amount=3245.75,
        Paid_Amount=2875.25,
        Payer_Name="Aetna",
        Claim_Status="Overturned",
        Notes="Corrected procedure codes and provided supporting documentation.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00178",
        Claim_Date="2024-02-15",
        Denial_Code="CO-252",
        Denial_Description="An attachment/other documentation is required to adjudicate this claim/service",
        Remittance_Code="N26",
        Remittance_Description="Missing itemized bill/statement",
        Claim_Amount=5678.90,
        Paid_Amount=4895.65,
        Payer_Name="UnitedHealthcare",
        Claim_Status="Overturned",
        Notes="Submitted requested itemized bill and additional supporting documents.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00204",
        Claim_Date="2024-03-07",
        Denial_Code="CO-97",
        Denial_Description="The benefit for this service is included in the payment/allowance for another service/procedure",
        Remittance_Code="N95",
        Remittance_Description="This provider type/specialty may not bill this service",
        Claim_Amount=1750.45,
        Paid_Amount=1350.75,
        Payer_Name="Cigna",
        Claim_Status="Overturned",
        Notes="Clarified benefit inclusion and provided documentation of coverage details.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00187",
        Claim_Date="2024-01-28",
        Denial_Code="CO-50",
        Denial_Description="These are non-covered services because this is not deemed a 'medical necessity' by the payer",
        Remittance_Code="M54",
        Remittance_Description="Missing/incomplete/invalid place of service",
        Claim_Amount=9350.25,
        Paid_Amount=8250.75,
        Payer_Name="Humana",
        Claim_Status="Overturned",
        Notes="Provided medical necessity documentation and corrected place of service information.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00215",
        Claim_Date="2024-02-12",
        Denial_Code="CO-22",
        Denial_Description="This care may be covered by another payer per coordination of benefits",
        Remittance_Code="N30",
        Remittance_Description="Patient ineligible for this service",
        Claim_Amount=4275.50,
        Paid_Amount=3825.25,
        Payer_Name="Blue Cross Blue Shield",
        Claim_Status="Overturned",
        Notes="Verified patient eligibility and submitted coordination of benefits details.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00233",
        Claim_Date="2024-03-10",
        Denial_Code="CO-29",
        Denial_Description="The time limit for filing has expired",
        Remittance_Code="M52",
        Remittance_Description="Missing/incomplete/invalid 'from' date(s) of service",
        Claim_Amount=3125.75,
        Paid_Amount=2750.50,
        Payer_Name="Anthem",
        Claim_Status="Overturned",
        Notes="Provided proof of timely filing and corrected date of service information.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00172",
        Claim_Date="2024-01-20",
        Denial_Code="PR-96",
        Denial_Description="Non-covered charge(s)",
        Remittance_Code="N130",
        Remittance_Description="Consult plan benefit documents/guidelines for information about restrictions for this service",
        Claim_Amount=6250.45,
        Paid_Amount=5325.30,
        Payer_Name="Kaiser Permanente",
        Claim_Status="Overturned",
        Notes="Reviewed plan benefit documents and provided evidence of coverage.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00196",
        Claim_Date="2024-02-25",
        Denial_Code="CO-45",
        Denial_Description="Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement",
        Remittance_Code="M53",
        Remittance_Description="Missing/incomplete/invalid days or units of service",
        Claim_Amount=7825.35,
        Paid_Amount=6950.20,
        Payer_Name="Medicare",
        Claim_Status="Overturned",
        Notes="Corrected date of service errors and provided supporting documentation.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00221",
        Claim_Date="2024-03-15",
        Denial_Code="CO-16",
        Denial_Description="Claim/service lacks information or has submission/billing error(s)",
        Remittance_Code="M51",
        Remittance_Description="Missing/incomplete/invalid procedure codes",
        Claim_Amount=4350.75,
        Paid_Amount=3825.50,
        Payer_Name="Molina Healthcare",
        Claim_Status="Overturned",
        Notes="Corrected procedure codes and resubmitted with supporting documentation.",
    ),
    ClaimHistoryLookupMatchResult(
        Member_ID="MID00245",
        Claim_Date="2024-01-18",
        Denial_Code="CO-252",
        Denial_Description="An attachment/other documentation is required to adjudicate this claim/service",
        Remittance_Code="N26",
        Remittance_Description="Missing itemized bill/statement",
        Claim_Amount=3850.25,
        Paid_Amount=3375.40,
        Payer_Name="Centene",
        Claim_Status="Overturned",
        Notes="Provided itemized bill and all requested additional documentation.",
    ),
]


claim_history: list[ClaimHistoryLookupMatchResult] = sample_claim_history_lookup_match_results


@tool
async def claim_history_lookup(
    DenialCode: Annotated[str, "Denial Code to lookup"],  # noqa: N803
    RemittanceCode: Annotated[str, "Remittance Code to lookup"],  # noqa: N803
) -> list[ClaimHistoryLookupMatchResult]:
    """Process to retrieve claim denial history by DenialCode and RemittanceCode.

    Args:
        DenialCode: Denial Code
        RemittanceCode: Remittance Code

    Returns:
       List of claim denial history match results
    """
    return [
        claim
        for claim in claim_history
        if claim.Denial_Code == DenialCode and claim.Remittance_Code == RemittanceCode
    ]
