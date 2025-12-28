"""
Layer 2 composition and sanitization helpers.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Callable, Dict, Optional

from .llm import LLMClient, get_config
from .schema import Actor, Layer1Event, Layer2Event
from .privacy import validate_layer2_no_pii
from .prompting import LAYER2_SYSTEM_PROMPT, parse_event_line, parse_layer2_event
from .formatting import build_layer2_prompt
from .session import Layer2Batch


class Layer2Composer:
    """Generates Layer 2 events given Layer 1 input."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self.config = get_config()

    def compose(
        self,
        layer1_events: list[Layer1Event],
        on_event: Optional[Callable[[Layer2Event], None]] = None,
    ) -> Layer2Batch:
        base_time = layer1_events[0].timestamp if layer1_events else datetime.now()
        actors: Dict[str, Actor] = {}

        print(f"Composing Layer 2 from {len(layer1_events)} Layer 1 events..."  )

        user_prompt = build_layer2_prompt(layer1_events, base_time)

        events: list[Layer2Event] = []

        def parse_line(line: str) -> Optional[Layer2Event]:
            return parse_event_line(line, base_time, actors, parse_layer2_event)

        l2_events = self.client.stream_and_parse(
            LAYER2_SYSTEM_PROMPT,
            user_prompt,
            parse_line,
            max_tokens=self.config.layer2_max_tokens,
            on_parsed=on_event,
        )
        for event in l2_events:
            events.append(event)

        return Layer2Batch(events=events, generated_at=datetime.now(), sanitized=False)


class Layer2Sanitizer:
    """
    Validates and sanitizes Layer 2 events before they leave the secure boundary.
    """

    def sanitize(self, batch: Layer2Batch) -> Layer2Batch:
        sanitized_events: list[Layer2Event] = []

        for event in batch.events:
            if validate_layer2_no_pii(event):
                sanitized_events.append(event)
            else:
                event_details = (
                    f"{type(event).__name__}: {getattr(event, 'description', 'N/A')}"
                )
                #raise ValueError(
                print(
                    f"Layer2Sanitizer detected PII content in event {event.event_id}: {event_details}\n"
                    "Ensure Layer 2 generation replaces transcripts with descriptors:\n"
                    "  - Avoid direct speech quotes or speech-related verbs (said, stated, uttered, etc.)\n"
                    "  - Use behavioral metadata instead: word_count, vocal_quality, communication_target, prosody\n"
                    "  - Example: Instead of 'Child said \"no\"', write 'Child vocalized protest with elevated volume'"
                )

        sanitized_batch = Layer2Batch(
            events=[copy.deepcopy(evt) for evt in sanitized_events],
            generated_at=batch.generated_at,
            sanitized=True,
        )
        return sanitized_batch
