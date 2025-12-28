"""
Layer 1 sources and event log helpers for the observation engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm import LLMClient, get_config
from .schema import Actor, Layer1Event
from .scenario import Scenario
from .prompting import LAYER1_SYSTEM_PROMPT, parse_event_line, parse_layer1_event
from .formatting import build_layer1_prompt
from .session import Layer1Batch
from .storage import deserialize_event


class Layer1Source:
    """Abstract base for Layer 1 sources."""

    def produce(
        self,
        scenario: Scenario,
        on_event: Optional[Callable[[Layer1Event], None]] = None,
    ) -> Layer1Batch:
        raise NotImplementedError


class LLMLayer1Source(Layer1Source):
    """Layer 1 generator backed by the LLM streaming interface."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self.config = get_config()

    def produce(
        self,
        scenario: Scenario,
        on_event: Optional[Callable[[Layer1Event], None]] = None,
    ) -> Layer1Batch:
        actors: Dict[str, Actor] = {}
        events: list[Layer1Event] = []
        base_time = datetime.now()

        user_prompt = build_layer1_prompt(scenario)

        def parse_line(line: str) -> Optional[Layer1Event]:
            return parse_event_line(line, base_time, actors, parse_layer1_event)

        for event in self.client.stream_and_parse(
            LAYER1_SYSTEM_PROMPT,
            user_prompt,
            parse_line,
            max_tokens=self.config.layer1_max_tokens,
            on_parsed=on_event,
        ):
            events.append(event)

        end_time = events[-1].timestamp if events else base_time
        return Layer1Batch(events=events, started_at=base_time, ended_at=end_time)


# =============================================================================
# L1 EVENT LOG
# =============================================================================


@dataclass
class L1EventLog:
    """A saved L1 event log with metadata for reconstruction."""

    log_id: str
    scenario_name: str
    scenario_description: str
    timestamp_created: datetime
    duration_seconds: int
    num_children: int
    num_adults: int
    llm_model: Optional[str] = None
    l1_events: List[Layer1Event] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""

        from .storage import serialize_event

        return {
            "log_id": self.log_id,
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "timestamp_created": self.timestamp_created.isoformat(),
            "duration_seconds": self.duration_seconds,
            "num_children": self.num_children,
            "num_adults": self.num_adults,
            "llm_model": self.llm_model,
            "l1_events": [serialize_event(evt) for evt in self.l1_events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_time: datetime) -> "L1EventLog":
        """Deserialize from dictionary."""

        actors: Dict[str, Actor] = {}
        l1_events: List[Layer1Event] = []

        for evt_data in data.get("l1_events", []):
            event_type = evt_data.get("_type")

            if event_type == "SpeechEvent":
                evt = parse_layer1_event("SPEECH", evt_data, base_time, actors)
            elif event_type == "AmbientAudioEvent":
                evt = parse_layer1_event("AMBIENT", evt_data, base_time, actors)
            elif event_type == "ProximityEvent":
                evt = parse_layer1_event("PROXIMITY", evt_data, base_time, actors)
            elif event_type == "GazeEvent":
                evt = parse_layer1_event("GAZE", evt_data, base_time, actors)
            elif event_type == "PostureEvent":
                evt = parse_layer1_event("POSTURE", evt_data, base_time, actors)
            elif event_type == "ObjectEvent":
                evt = parse_layer1_event("OBJECT", evt_data, base_time, actors)
            else:
                evt = None

            if evt:
                l1_events.append(evt)

        return cls(
            log_id=data["log_id"],
            scenario_name=data["scenario_name"],
            scenario_description=data["scenario_description"],
            timestamp_created=datetime.fromisoformat(data["timestamp_created"]),
            duration_seconds=data["duration_seconds"],
            num_children=data["num_children"],
            num_adults=data["num_adults"],
            llm_model=data.get("llm_model"),
            l1_events=l1_events,
        )


def load_l1_log_from_session(session_dir: str | Path) -> L1EventLog:
    """Construct an L1EventLog from an ObservationEngine session directory."""

    session_path = Path(session_dir)
    scenario_path = session_path / "scenario.json"
    layer1_path = session_path / "layer1_raw.json"
    metadata_path = session_path / "metadata.json"

    if not scenario_path.exists() or not layer1_path.exists():
        raise FileNotFoundError(
            f"Session directory missing scenario or layer1 artifacts: {session_path}"
        )

    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_data = json.load(f)

    with open(layer1_path, "r", encoding="utf-8") as f:
        layer1_data = json.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    started = layer1_data.get("started_at")
    ended = layer1_data.get("ended_at")

    try:
        started_dt = datetime.fromisoformat(started) if started else None
    except Exception:
        started_dt = None

    try:
        ended_dt = datetime.fromisoformat(ended) if ended else None
    except Exception:
        ended_dt = None

    if started_dt and ended_dt:
        duration_seconds = int((ended_dt - started_dt).total_seconds())
    else:
        duration_seconds = 0

    timestamp_created = started_dt or datetime.now()

    l1_events: List[Layer1Event] = []
    for evt in layer1_data.get("events", []):
        try:
            obj = deserialize_event(evt)
        except Exception:
            continue
        l1_events.append(obj)

    log_id = metadata.get("session_id") or session_path.name
    llm_model = (metadata.get("models", {}) or {}).get("layer1_llm_model") if metadata else None

    return L1EventLog(
        log_id=log_id,
        scenario_name=scenario_data.get("name", "unknown"),
        scenario_description=scenario_data.get("description", ""),
        timestamp_created=timestamp_created,
        duration_seconds=duration_seconds,
        num_children=scenario_data.get("num_children", 0),
        num_adults=scenario_data.get("num_adults", 0),
        llm_model=llm_model,
        l1_events=l1_events,
    )
