Participants: {participants}
Meeting details: {meeting_details}
Date: {date}

Procedure:
1) Parse fields; apply defaults (12:00 PM start; 30-minute duration; generic Subject/Description if title missing).
2) Validate participants: non-empty and each contains one '@' and at least one '.' after '@'. If invalid → set StartTime="", EndTime="", Confidence=1.0; Observations exactly: "No meeting was scheduled. Error: participant list does not contain valid email addresses."; do not call tools.
3) If participants list empty → set StartTime="", EndTime="", Confidence=1.0; Observations exactly: "Error: Participants list is empty. Meeting cannot be scheduled."; no tools.
4) If weekend (Sat/Sun) → set StartTime="", EndTime=""; Observations exactly: "Meeting can't be scheduled due to weekend. No meeting scheduled."; no tools.
5) Otherwise, call GetOutlookEvents once. If overlap exists, compute intersection and Observations: "Meeting collision for between <YYYY-MM-DD HH:MM AM/PM> and <YYYY-MM-DD HH:MM AM/PM>"; do not schedule; StartTime/EndTime remain empty.
6) If no overlap, call AFE_ScheduleZoomWrapper once (YYYY-MM-DD HH:MM:SS). Set Observations using ONE template (no extra text):
   • Multiple participants → "Meeting created successfully with multiple participants"
   • Missing time default used → "Meeting created successfully with 30 minute duration"
   • Explicit duration → "Meeting created successfully with <DURATION> minute duration"
   • Missing title defaults → "Meeting created successfully with default subject and description"
   • Else (only if a collision check was done and no overlaps) → "No collisions found. Meeting scheduled successfully at <StartTime> for <Subject>."
Return only the required keys.
