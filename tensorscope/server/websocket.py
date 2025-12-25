"""WebSocket handler for Tensorscope real-time updates."""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from .state import state
from .schemas import TensorSummaryResponse


async def handle_websocket(websocket: WebSocket) -> None:
    """Handle a WebSocket connection.

    Protocol:
        Client -> Server:
            { "type": "subscribe", "tensor_id": "..." }
            { "type": "unsubscribe", "tensor_id": "..." }
            { "type": "update_param", "scenario_id": "...", "param": "...", "value": ... }

        Server -> Client:
            { "type": "tensor_update", "tensor_id": "...", "summary": {...} }
            { "type": "error", "message": "..." }

    Args:
        websocket: The WebSocket connection.
    """
    await state.connect(websocket)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await send_error(websocket, "Invalid JSON")
                continue

            msg_type = message.get("type")

            if msg_type == "subscribe":
                await handle_subscribe(websocket, message)
            elif msg_type == "unsubscribe":
                await handle_unsubscribe(websocket, message)
            elif msg_type == "update_param":
                await handle_update_param(websocket, message)
            else:
                await send_error(websocket, f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        state.disconnect(websocket)
    except Exception as e:
        state.disconnect(websocket)
        raise


async def handle_subscribe(websocket: WebSocket, message: Dict[str, Any]) -> None:
    """Handle a subscribe message.

    Args:
        websocket: The WebSocket connection.
        message: The subscribe message.
    """
    tensor_id = message.get("tensor_id")
    if not tensor_id:
        await send_error(websocket, "Missing tensor_id in subscribe message")
        return

    state.subscribe(websocket, tensor_id)

    # Send current tensor state if available
    summary = state.get_tensor_summary(tensor_id)
    if summary is not None:
        await send_tensor_update(websocket, tensor_id, summary)


async def handle_unsubscribe(websocket: WebSocket, message: Dict[str, Any]) -> None:
    """Handle an unsubscribe message.

    Args:
        websocket: The WebSocket connection.
        message: The unsubscribe message.
    """
    tensor_id = message.get("tensor_id")
    if not tensor_id:
        await send_error(websocket, "Missing tensor_id in unsubscribe message")
        return

    state.unsubscribe(websocket, tensor_id)


async def handle_update_param(websocket: WebSocket, message: Dict[str, Any]) -> None:
    """Handle a parameter update message.

    This triggers a scenario re-run with updated parameters and sends
    the updated tensor summaries back to the requesting client.

    Args:
        websocket: The WebSocket connection.
        message: The update_param message.
    """
    scenario_id = message.get("scenario_id")
    param_name = message.get("param")
    param_value = message.get("value")

    if not scenario_id:
        await send_error(websocket, "Missing scenario_id in update_param message")
        return
    if not param_name:
        await send_error(websocket, "Missing param in update_param message")
        return
    if param_value is None:
        await send_error(websocket, "Missing value in update_param message")
        return

    # Get current parameters and update
    current_params = state.current_params
    current_params[param_name] = param_value

    try:
        # Re-run scenario with updated parameters
        summaries = state.run_scenario(scenario_id, current_params)

        # Send bulk update with all tensors to requesting client
        await send_tensors_update(websocket, summaries)

        # Also broadcast to other subscribed clients
        await state.broadcast_all_updates()

    except ValueError as e:
        await send_error(websocket, str(e))


async def send_tensor_update(
    websocket: WebSocket,
    tensor_id: str,
    summary: Any,
) -> None:
    """Send a tensor update message.

    Args:
        websocket: The WebSocket connection.
        tensor_id: The tensor ID.
        summary: The tensor summary.
    """
    response = TensorSummaryResponse(
        id=summary.id,
        name=summary.name,
        kind=summary.kind,
        tags=summary.tags,
        shape=list(summary.shape),
        dtype=summary.dtype,
        stats=summary.stats,
        recommended_views=summary.recommended_views,
    )

    await websocket.send_json({
        "type": "tensor_update",
        "tensor_id": tensor_id,
        "summary": response.model_dump(),
    })


async def send_tensors_update(
    websocket: WebSocket,
    summaries: Dict[str, Any],
) -> None:
    """Send a bulk tensor update message with all tensors.

    Args:
        websocket: The WebSocket connection.
        summaries: Dict of tensor key to TensorSummary.
    """
    tensors_dict = {}
    for key, summary in summaries.items():
        response = TensorSummaryResponse(
            id=summary.id,
            name=summary.name,
            kind=summary.kind,
            tags=summary.tags,
            shape=list(summary.shape),
            dtype=summary.dtype,
            stats=summary.stats,
            recommended_views=summary.recommended_views,
        )
        tensors_dict[key] = response.model_dump()

    await websocket.send_json({
        "type": "tensors_update",
        "tensors": tensors_dict,
    })


async def send_error(websocket: WebSocket, message: str) -> None:
    """Send an error message.

    Args:
        websocket: The WebSocket connection.
        message: The error message.
    """
    await websocket.send_json({
        "type": "error",
        "message": message,
    })
