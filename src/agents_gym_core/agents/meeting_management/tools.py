# ruff: noqa: N802, N803
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import TypedDict

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents_gym_core.agents.meeting_management.settings import (
    meeting_management_settings,
)


class AgentInputSchema(BaseModel):
    meeting_details: str = Field(description="Details about the meeting")
    participants: list[str] = Field(description="Participants in the meeting")
    date: str = Field(description="Date of the meeting")


class AgentOutputSchema(BaseModel):
    Description: str | None = Field(
        default=None,
        description="Body of the created meeting",
    )
    Subject: str | None = Field(
        default=None,
        description="Subject of the created meeting",
    )
    StartTime: str | None = Field(
        default=None,
        description="Start time of the meeting in string format",
    )
    EndTime: str | None = Field(
        default=None,
        description="End time of the meeting in string format",
    )
    Confidence: float | None = Field(
        default=None,
        description="Score from 0 to 1 resulted from execution confidence",
    )
    Observations: str | None = Field(
        default=None,
        description="Execution observation such as errors",
    )


class MeetingTimeConstraintsSchema(BaseModel):
    ExactStartTime: str | None = Field(
        default=None,
        description="Start time of the meeting in string format",
    )
    ExactEndTime: str | None = Field(
        default=None,
        description="End time of the meeting in string format",
    )
    ExactDuration: int | None = Field(
        default=None,
        description="Duration of the meeting in minutes",
    )
    AllowedIntervalTime: tuple[str, str] | None = Field(
        default=None,
        description="Allowed interval time for the meeting",
    )
    NoMeeting: bool | None = Field(
        default=None,
        description="No meeting should be scheduled",
    )

    @property
    def exact_start_time(self) -> pd.Timestamp | None:
        return pd.to_datetime(self.ExactStartTime) if self.ExactStartTime else None

    @property
    def exact_end_time(self) -> pd.Timestamp | None:
        return pd.to_datetime(self.ExactEndTime) if self.ExactEndTime else None


class OutlookEvent(TypedDict):
    start_time: datetime
    end_time: datetime


@lru_cache
def _load_events() -> dict[str, list[OutlookEvent]]:
    events: dict[str, list[OutlookEvent]] = {}
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "documents",
            meeting_management_settings.GET_OUTLOOK_EVENTS_TOOL_DOCUMENT,
        ),
    ) as f:
        list_events = json.load(f)["events"]
        for event in list_events:
            day_of_event = event["start_time"].split(" ")[0]
            if day_of_event not in events:
                events[day_of_event] = []
            events[day_of_event].append(
                OutlookEvent(
                    start_time=datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S"),
                    end_time=datetime.strptime(event["end_time"], "%Y-%m-%d %H:%M:%S"),
                )
            )
    return events


def is_valid_email(email: str) -> bool:
    """Validate if the provided string is an email address."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def is_overlap(start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> bool:
    """Check if two time intervals overlap."""
    return max(start1, start2) < min(end1, end2)


@tool("GetOutlookEvents")
async def GetOutlookEvents(
    From_Date: str,
    To_Date: str,
    Number_Of_Events: int = 5,
):
    """Get events from a static Outlook calendar dataset."""
    events = _load_events()

    try:
        from_date = datetime.strptime(From_Date, "%Y-%m-%d %H:%M:%S")
        to_date = datetime.strptime(To_Date, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error: {e}"

    if from_date > to_date:
        return "Error: from_date is greater than to_date"

    if from_date.weekday() == 5 or from_date.weekday() == 6:
        return "Error: from_date is weekend"

    base_date = From_Date.split(" ")[0]
    scheduled_events = []
    for event in events.get(base_date, []):
        if is_overlap(from_date, to_date, event["start_time"], event["end_time"]):
            scheduled_events.append(event)

    return scheduled_events[:Number_Of_Events]


def _schedule_zoom_impl(
    Start_Time: str,
    End_Time: str,
    Attendees: list[str],
    Subject: str,
    Description: str,
) -> str:
    try:
        start_time = datetime.strptime(Start_Time, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(End_Time, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error: {e}"

    if start_time > end_time:
        return "Error: start_time is greater than End_Time"

    if start_time.weekday() == 5 or start_time.weekday() == 6:
        return "Error: start_time is weekend"

    if end_time.weekday() == 5 or end_time.weekday() == 6:
        return "Error: End_Time is weekend"

    if start_time.hour < 9 or start_time.hour > 18:
        return "Error: start_time is not in business hours"

    for attendee in Attendees:
        if not is_valid_email(attendee):
            return f"Error: {attendee} is not a valid email address"

    return (
        f"Meeting scheduled successfully at {start_time} for {Subject} with {Attendees} "
        f"and description {Description}"
    )


@tool("AFE_ScheduleZoomWrapper")
async def AFE_ScheduleZoomWrapper(
    Start_Time: str,
    End_Time: str,
    Attendees: list[str],
    Subject: str,
    Description: str,
):
    """Schedule a Zoom meeting with basic validation."""
    return _schedule_zoom_impl(
        Start_Time=Start_Time,
        End_Time=End_Time,
        Attendees=Attendees,
        Subject=Subject,
        Description=Description,
    )


@tool("AFE_ScheduleZoomWrapper")
async def AFE_ScheduleZoomWrapperWrongArgs(
    Begin: str,
    End: str,
    Attendees: list[str],
    Subject: str,
    Description: str,
):
    """Schedule a Zoom meeting with a mismatched schema."""
    return _schedule_zoom_impl(
        Start_Time=Begin,
        End_Time=End,
        Attendees=Attendees,
        Subject=Subject,
        Description=Description,
    )
