"""Pydantic models for Tensorscope API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):
    """Schema for a scenario parameter."""

    name: str
    display_name: str
    type: str  # "continuous" or "discrete"
    default: Union[float, int, str]
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Any]] = None


class ProbeSchema(BaseModel):
    """Schema for a probe point."""

    key: str
    display_name: str
    description: str = ""


class GraphNodeSchema(BaseModel):
    """Schema for a node in the operator graph."""

    id: str
    name: str
    inputs: List[str]
    outputs: List[str]
    tags: List[str] = Field(default_factory=list)


class GraphEdgeSchema(BaseModel):
    """Schema for an edge in the operator graph."""

    from_node: str
    from_output: str
    to_node: str
    to_input: str


class GraphSchema(BaseModel):
    """Schema for the operator graph."""

    nodes: List[GraphNodeSchema]
    edges: List[GraphEdgeSchema]


class ScenarioInfo(BaseModel):
    """Summary info for listing scenarios."""

    id: str
    name: str
    description: str


class ScenarioDetail(BaseModel):
    """Full scenario details including parameters and graph."""

    id: str
    name: str
    description: str
    parameters: List[ParameterSchema]
    probes: List[ProbeSchema]
    graph: Optional[GraphSchema] = None


class TensorSummaryResponse(BaseModel):
    """Tensor summary for API responses."""

    id: str
    name: str
    kind: str
    tags: List[str]
    shape: List[int]
    dtype: str
    stats: Dict[str, Any]
    recommended_views: List[str]


class TensorDataResponse(BaseModel):
    """Full tensor data for API responses (small tensors only)."""

    id: str
    name: str
    shape: List[int]
    dtype: str
    data: List[Any]  # Nested list representing the tensor


class TensorSliceRequest(BaseModel):
    """Request for a tensor slice."""

    row_start: int = 0
    row_end: Optional[int] = None
    col_start: int = 0
    col_end: Optional[int] = None


class TensorSliceResponse(BaseModel):
    """Response containing a tensor slice."""

    id: str
    name: str
    full_shape: List[int]
    slice_shape: List[int]
    row_range: Tuple[int, int]
    col_range: Tuple[int, int]
    data: List[Any]


class ParameterUpdate(BaseModel):
    """Request to update a parameter value."""

    name: str
    value: Union[float, int, str]


class RunScenarioRequest(BaseModel):
    """Request to run a scenario with specific parameters."""

    parameters: Dict[str, Union[float, int, str]] = Field(default_factory=dict)


class RunScenarioResponse(BaseModel):
    """Response from running a scenario."""

    scenario_id: str
    parameters: Dict[str, Any]
    tensors: Dict[str, TensorSummaryResponse]


# WebSocket message types


class WSSubscribeMessage(BaseModel):
    """WebSocket message to subscribe to tensor updates."""

    type: str = "subscribe"
    tensor_id: str
    view: Optional[str] = None


class WSUnsubscribeMessage(BaseModel):
    """WebSocket message to unsubscribe from tensor updates."""

    type: str = "unsubscribe"
    tensor_id: str


class WSUpdateParamMessage(BaseModel):
    """WebSocket message to update a parameter."""

    type: str = "update_param"
    scenario_id: str
    param: str
    value: Union[float, int, str]


class WSTensorUpdateMessage(BaseModel):
    """WebSocket message sent when a tensor is updated."""

    type: str = "tensor_update"
    tensor_id: str
    summary: TensorSummaryResponse


class WSGraphUpdateMessage(BaseModel):
    """WebSocket message sent when the graph is updated."""

    type: str = "graph_update"
    nodes: List[GraphNodeSchema]
    edges: List[GraphEdgeSchema]


class WSErrorMessage(BaseModel):
    """WebSocket error message."""

    type: str = "error"
    message: str
