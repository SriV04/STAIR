"""Sequential edge-driven DA4ML evaluation for a fixed FoldPlan."""

from __future__ import annotations

from .cost_model import evaluate_node
from .records import (
    DA4MLNodeArtifact,
    EvaluationResult,
    NodeEvaluation,
    SymbolicTensorState,
    snapshot_node_evaluation,
)


def _topological_order(graph) -> list[int]:
    indegree = {node: 0 for node in graph.vertices}
    successors = {node: [] for node in graph.vertices}
    for source, destination in graph.edges:
        indegree[destination] += 1
        successors[source].append(destination)
    ready = [node for node in graph.vertices if indegree[node] == 0]
    ordered = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for destination in successors[node]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
    if len(ordered) != len(graph.vertices):
        raise ValueError("Sched-IR contains a dependency cycle")
    return ordered


class FoldedCostChainExecutor:
    def __init__(self, model, fold_plan, config):
        self.model = model
        self.fold_plan = fold_plan
        self.config = config or {}
        self.tensor_states: dict[str, SymbolicTensorState] = {}
        self.node_results: list[NodeEvaluation] = []
        self.runtime_artifacts: dict[str, DA4MLNodeArtifact] = {}

    def _layer_for(self, node_properties):
        if node_properties.get("keras_layer") is not None:
            return node_properties.get("keras_layer")
        if self.model is None:
            return None
        name = node_properties.get("nn_layer_name")
        if name is None:
            return None
        try:
            return self.model.get_layer(name)
        except ValueError:
            return None

    def _incoming_states(self, graph, node_id) -> list[SymbolicTensorState]:
        states = []
        for source, destination in graph.edges:
            if destination != node_id:
                continue
            edge = graph.pmap[(source, destination)]
            value_id = edge.get("value_id")
            if value_id is None or value_id not in self.tensor_states:
                raise ValueError(f"Node {node_id} input edge from {source} has no evaluated value")
            states.append(self.tensor_states[value_id])
        return states

    def _publish_outputs(self, graph, node_id, primitive) -> list[SymbolicTensorState]:
        outgoing = [(source, destination) for source, destination in graph.edges if source == node_id]
        output_count = max(len(primitive.output_shapes), 1)
        states = []
        for output_index in range(output_count):
            shape = primitive.output_shapes[min(output_index, len(primitive.output_shapes) - 1)]
            value_id = f"{node_id}:out:{output_index}"
            state = SymbolicTensorState(
                value_id=value_id,
                shape=shape,
                qints=primitive.output_qints,
                kifs=primitive.output_kifs,
                latency=primitive.output_latency,
                producer_node=node_id,
                symbolic_value=primitive.symbolic_outputs[min(output_index, len(primitive.symbolic_outputs) - 1)],
            )
            self.tensor_states[value_id] = state
            states.append(state)
        for source, destination in outgoing:
            edge = graph.pmap[(source, destination)]
            state = states[0]
            edge["value_id"] = state.value_id
            edge["evaluated_qints"] = state.qints
            edge["evaluated_kifs"] = state.kifs
            edge["evaluated_shape"] = state.shape
            edge["evaluated_latency"] = state.latency
        return states

    def run(self, graph) -> EvaluationResult:
        for index, node_id in enumerate(_topological_order(graph)):
            properties = graph.pmap[node_id]
            input_states = self._incoming_states(graph, node_id)
            primitive = evaluate_node(
                properties,
                input_states=input_states,
                keras_layer=self._layer_for(properties),
                config=self.config,
            )
            trace_id = f"trace:{index}"
            output_states = self._publish_outputs(graph, node_id, primitive)
            evaluation = NodeEvaluation(
                node_id=node_id,
                trace_id=trace_id,
                input_states=input_states,
                output_states=output_states,
                cost=primitive.cost,
                latency=(0.0, primitive.output_latency),
                n_ops=primitive.n_ops,
            )
            properties.update(snapshot_node_evaluation(evaluation))
            properties["evaluated_comb_shape"] = getattr(primitive.comb_logic, "shape", None)
            properties["evaluated_pipeline_stages"] = primitive.cost.get("pipeline_stages")
            self.node_results.append(evaluation)
            self.runtime_artifacts[trace_id] = DA4MLNodeArtifact(
                trace_id=trace_id,
                node_id=node_id,
                symbolic_inputs=primitive.symbolic_inputs,
                symbolic_outputs=primitive.symbolic_outputs,
                comb_logic=primitive.comb_logic,
                pipeline=primitive.pipeline,
            )
        graph.pmap["backend"] = "da4ml"
        graph.pmap["backend_evaluated"] = True
        return EvaluationResult(graph, self.node_results, self.tensor_states, self.runtime_artifacts)
