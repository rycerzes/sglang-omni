# SPDX-License-Identifier: Apache-2.0
"""Real-Time interactive voice API package."""

from sglang_omni.serve.realtime.api import register_realtime
from sglang_omni.serve.realtime.protocol import SessionConfig, TurnDetectionConfig
from sglang_omni.serve.realtime.session import RealtimeSession

__all__ = [
    "RealtimeSession",
    "SessionConfig",
    "TurnDetectionConfig",
    "register_realtime",
]
