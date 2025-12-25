"""REST API routes for Tensorscope."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from .state import state
from .schemas import (
    ScenarioInfo,
    ScenarioDetail,
    ParameterSchema,
    ProbeSchema,
    GraphSchema,
    GraphNodeSchema,
    GraphEdgeSchema,
    TensorSummaryResponse,
    TensorDataResponse,
    TensorSliceResponse,
    RunScenarioRequest,
    RunScenarioResponse,
)

router = APIRouter(prefix="/api")


@router.get("/scenarios", response_model=List[ScenarioInfo])
async def list_scenarios() -> List[ScenarioInfo]:
    """List all available scenarios."""
    scenarios = state.list_scenarios()
    return [
        ScenarioInfo(
            id=s.id,
            name=s.name,
            description=s.description,
        )
        for s in scenarios
    ]


@router.get("/scenarios/{scenario_id}", response_model=ScenarioDetail)
async def get_scenario(scenario_id: str) -> ScenarioDetail:
    """Get detailed information about a scenario."""
    scenario = state.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")

    # Build parameter schemas
    parameters = [
        ParameterSchema(
            name=p.name,
            display_name=p.display_name,
            type=p.param_type.value,
            default=p.default,
            description=p.description,
            min=p.min_val,
            max=p.max_val,
            step=p.step,
            options=p.options,
        )
        for p in scenario.parameters.values()
    ]

    # Build probe schemas
    probes = [
        ProbeSchema(
            key=p.tensor_key,
            display_name=p.display_name,
            description=p.description,
        )
        for p in scenario.probes
    ]

    # Build graph schema
    graph_schema = None
    if scenario.graph is not None:
        graph_dict = scenario.graph.to_dict()
        nodes = [
            GraphNodeSchema(
                id=n["id"],
                name=n["name"],
                inputs=n["inputs"],
                outputs=n["outputs"],
                tags=n.get("tags", []),
            )
            for n in graph_dict["nodes"]
        ]
        edges = [
            GraphEdgeSchema(
                from_node=e["from_node"],
                from_output=e["from_output"],
                to_node=e["to_node"],
                to_input=e["to_input"],
            )
            for e in graph_dict["edges"]
        ]
        graph_schema = GraphSchema(nodes=nodes, edges=edges)

    return ScenarioDetail(
        id=scenario.id,
        name=scenario.name,
        description=scenario.description,
        parameters=parameters,
        probes=probes,
        graph=graph_schema,
    )


@router.post("/scenarios/{scenario_id}/run", response_model=RunScenarioResponse)
async def run_scenario(
    scenario_id: str,
    request: RunScenarioRequest,
) -> RunScenarioResponse:
    """Run a scenario with the given parameters."""
    scenario = state.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")

    try:
        summaries = state.run_scenario(scenario_id, request.parameters or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert summaries to response format
    tensor_responses: Dict[str, TensorSummaryResponse] = {}
    for key, summary in summaries.items():
        tensor_responses[key] = TensorSummaryResponse(
            id=summary.id,
            name=summary.name,
            kind=summary.kind,
            tags=summary.tags,
            shape=list(summary.shape),
            dtype=summary.dtype,
            stats=summary.stats,
            recommended_views=summary.recommended_views,
        )

    return RunScenarioResponse(
        scenario_id=scenario_id,
        parameters=state.current_params,
        tensors=tensor_responses,
    )


@router.get("/tensors/{tensor_id}/summary", response_model=TensorSummaryResponse)
async def get_tensor_summary(tensor_id: str) -> TensorSummaryResponse:
    """Get summary statistics for a tensor."""
    summary = state.get_tensor_summary(tensor_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Tensor '{tensor_id}' not found")

    return TensorSummaryResponse(
        id=summary.id,
        name=summary.name,
        kind=summary.kind,
        tags=summary.tags,
        shape=list(summary.shape),
        dtype=summary.dtype,
        stats=summary.stats,
        recommended_views=summary.recommended_views,
    )


@router.get("/tensors/{tensor_id}/data", response_model=TensorDataResponse)
async def get_tensor_data(
    tensor_id: str,
    max_size: int = Query(default=10000, description="Maximum number of elements"),
) -> TensorDataResponse:
    """Get full tensor data (for small tensors only)."""
    tensor = state.get_tensor(tensor_id)
    if tensor is None:
        raise HTTPException(status_code=404, detail=f"Tensor '{tensor_id}' not found")

    if tensor.data.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Tensor too large ({tensor.data.size} elements). Use /slice endpoint.",
        )

    return TensorDataResponse(
        id=tensor.id,
        name=tensor.name,
        shape=list(tensor.shape),
        dtype=tensor.dtype,
        data=tensor.data.tolist(),
    )


@router.get("/tensors/{tensor_id}/slice", response_model=TensorSliceResponse)
async def get_tensor_slice(
    tensor_id: str,
    row_start: int = Query(default=0, ge=0),
    row_end: Optional[int] = Query(default=None),
    col_start: int = Query(default=0, ge=0),
    col_end: Optional[int] = Query(default=None),
) -> TensorSliceResponse:
    """Get a slice of a tensor."""
    tensor = state.get_tensor(tensor_id)
    if tensor is None:
        raise HTTPException(status_code=404, detail=f"Tensor '{tensor_id}' not found")

    data = tensor.data

    # Handle different tensor dimensions
    if data.ndim == 1:
        # For 1D tensors, only use row_start/row_end as the slice
        actual_row_end = row_end if row_end is not None else len(data)
        sliced = data[row_start:actual_row_end]
        slice_shape = list(sliced.shape)
        row_range = (row_start, actual_row_end)
        col_range = (0, 0)
    elif data.ndim == 2:
        actual_row_end = row_end if row_end is not None else data.shape[0]
        actual_col_end = col_end if col_end is not None else data.shape[1]
        sliced = data[row_start:actual_row_end, col_start:actual_col_end]
        slice_shape = list(sliced.shape)
        row_range = (row_start, actual_row_end)
        col_range = (col_start, actual_col_end)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Slicing not supported for {data.ndim}D tensors",
        )

    return TensorSliceResponse(
        id=tensor.id,
        name=tensor.name,
        full_shape=list(tensor.shape),
        slice_shape=slice_shape,
        row_range=row_range,
        col_range=col_range,
        data=sliced.tolist(),
    )
