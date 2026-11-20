USER REQUEST:
====================

<core-instructions>
{systemPrompt}
</core-instructions>

<user-request>
{userPrompt}
</user-request>

<available-tools>
{tools}
</available-tools>

YOUR TASK:
====================

Analyze the situation comprehensively and provide strategic guidance to the agent. Your response should include:

1. SITUATION ASSESSMENT
    - Analyze the user's request
    - Identify key constraints and challenges
    - Determine what information is available and what's missing

2. STRATEGIC PLANNING
    - Outline a clear sequence of steps to accomplish the goal
    - Specify which tools to use for each step and why
    - Anticipate potential challenges and how to address them

3. ERROR HANDLING
    - Identify potential failure points
    - Suggest contingency plans

RULES:
    - Tools can be used only when all required parameters have verified, specific data. Do not use placeholders.
    - Do not attempt to supply missing information. Either tell the agent to ask for it if possible or end the execution.
    - Do not rely on your own knowledge or the agent's. Use the information provided in the user request and the available tools.

Your response will be provided directly to the agent. Be thorough but also clear and actionable.
The agent will use your insights to determine its next actions.

Your reponse must be a YAML representation of the expected sequence of steps required to achieve the end goal.
