Inputs:
- DenialCode: {DenialCode}
- RemittanceCode: {RemittanceCode}

Do:
1) Call claim_history_lookup once with both inputs.
2) Produce only these fields:
   AppealTips: [concise, actionable steps]
   ErrorMessage: "" if tips exist; otherwise brief reason.
No narrative text; no extra fields.
