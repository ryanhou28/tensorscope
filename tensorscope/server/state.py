"""Server state management for Tensorscope."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import asyncio

from fastapi import WebSocket

from ..core.scenario import Scenario
from ..core.tensor import TrackedTensor, TensorSummary, compute_summary


class ServerState:
    """Manages server-side state for scenarios and WebSocket connections.

    This singleton class holds:
    - Loaded scenarios
    - Current tensor cache (from last execution)
    - Active WebSocket connections and their subscriptions
    """

    _instance: Optional["ServerState"] = None

    def __new__(cls) -> ServerState:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._scenarios: Dict[str, Scenario] = {}
        self._tensor_cache: Dict[str, TrackedTensor] = {}
        self._summary_cache: Dict[str, TensorSummary] = {}
        self._active_scenario_id: Optional[str] = None
        self._current_params: Dict[str, Any] = {}

        # WebSocket management
        self._connections: Set[WebSocket] = set()
        self._subscriptions: Dict[WebSocket, Set[str]] = {}  # ws -> set of tensor_ids

    def register_scenario(self, scenario: Scenario) -> None:
        """Register a scenario for use by the API.

        Args:
            scenario: The scenario to register.
        """
        self._scenarios[scenario.id] = scenario

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get a registered scenario by ID.

        Args:
            scenario_id: The scenario ID.

        Returns:
            The scenario, or None if not found.
        """
        return self._scenarios.get(scenario_id)

    def list_scenarios(self) -> List[Scenario]:
        """Get all registered scenarios.

        Returns:
            List of all registered scenarios.
        """
        return list(self._scenarios.values())

    def run_scenario(
        self,
        scenario_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, TensorSummary]:
        """Run a scenario and cache the results.

        Args:
            scenario_id: The scenario ID to run.
            params: Optional parameter overrides.

        Returns:
            Dict mapping tensor keys to their summaries.

        Raises:
            ValueError: If scenario not found.
        """
        scenario = self.get_scenario(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario '{scenario_id}' not found")

        # Run the scenario
        tensors = scenario.run(params)

        # Update caches
        self._active_scenario_id = scenario_id
        self._current_params = scenario._last_params.copy()
        self._tensor_cache.clear()
        self._summary_cache.clear()

        for key, tensor in tensors.items():
            self._tensor_cache[tensor.id] = tensor
            self._tensor_cache[key] = tensor  # Also index by key
            summary = compute_summary(tensor)
            self._summary_cache[tensor.id] = summary
            self._summary_cache[key] = summary

        # Return summaries
        return {key: compute_summary(tensor) for key, tensor in tensors.items()}

    def get_tensor(self, tensor_id: str) -> Optional[TrackedTensor]:
        """Get a cached tensor by ID or key.

        Args:
            tensor_id: The tensor ID or key (e.g., "node.output").

        Returns:
            The tensor, or None if not found.
        """
        return self._tensor_cache.get(tensor_id)

    def get_tensor_summary(self, tensor_id: str) -> Optional[TensorSummary]:
        """Get a cached tensor summary by ID or key.

        Args:
            tensor_id: The tensor ID or key.

        Returns:
            The summary, or None if not found.
        """
        return self._summary_cache.get(tensor_id)

    def get_all_summaries(self) -> Dict[str, TensorSummary]:
        """Get all cached tensor summaries.

        Returns:
            Dict mapping tensor IDs to summaries.
        """
        # Return only summaries indexed by tensor ID (not key duplicates)
        seen_ids: Set[str] = set()
        result: Dict[str, TensorSummary] = {}
        for key, summary in self._summary_cache.items():
            if summary.id not in seen_ids:
                result[summary.id] = summary
                seen_ids.add(summary.id)
        return result

    @property
    def active_scenario_id(self) -> Optional[str]:
        """Get the currently active scenario ID."""
        return self._active_scenario_id

    @property
    def current_params(self) -> Dict[str, Any]:
        """Get the current parameter values."""
        return self._current_params.copy()

    # WebSocket management

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection.
        """
        await websocket.accept()
        self._connections.add(websocket)
        self._subscriptions[websocket] = set()

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection.

        Args:
            websocket: The WebSocket connection.
        """
        self._connections.discard(websocket)
        self._subscriptions.pop(websocket, None)

    def subscribe(self, websocket: WebSocket, tensor_id: str) -> None:
        """Subscribe a connection to tensor updates.

        Args:
            websocket: The WebSocket connection.
            tensor_id: The tensor ID to subscribe to.
        """
        if websocket in self._subscriptions:
            self._subscriptions[websocket].add(tensor_id)

    def unsubscribe(self, websocket: WebSocket, tensor_id: str) -> None:
        """Unsubscribe a connection from tensor updates.

        Args:
            websocket: The WebSocket connection.
            tensor_id: The tensor ID to unsubscribe from.
        """
        if websocket in self._subscriptions:
            self._subscriptions[websocket].discard(tensor_id)

    def get_subscriptions(self, websocket: WebSocket) -> Set[str]:
        """Get all tensor subscriptions for a connection.

        Args:
            websocket: The WebSocket connection.

        Returns:
            Set of subscribed tensor IDs.
        """
        return self._subscriptions.get(websocket, set()).copy()

    async def broadcast_tensor_update(
        self,
        tensor_id: str,
        summary: TensorSummary,
    ) -> None:
        """Broadcast a tensor update to subscribed connections.

        Args:
            tensor_id: The tensor ID that was updated.
            summary: The updated tensor summary.
        """
        message = {
            "type": "tensor_update",
            "tensor_id": tensor_id,
            "summary": summary.to_dict(),
        }

        disconnected: List[WebSocket] = []
        for ws in self._connections:
            if tensor_id in self._subscriptions.get(ws, set()):
                try:
                    await ws.send_json(message)
                except Exception:
                    disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_all_updates(self) -> None:
        """Broadcast all cached tensor updates to all subscribers."""
        for tensor_id, summary in self._summary_cache.items():
            await self.broadcast_tensor_update(tensor_id, summary)


# Global singleton instance
state = ServerState()
