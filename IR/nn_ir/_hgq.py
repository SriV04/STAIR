from __future__ import annotations

from .hgq2.hgq_extractor import (
    avg_bw,
    bitwidth_from_kif,
    bw_array,
    extract_all_quantizers,
    extract_kif,
    extract_layer_values,
    extract_quantizer_modes,
    extract_quantizer_variables,
    find_activation_quantizer,
    find_output_quantizer,
    kif_to_qint,
    max_bw,
    min_bw,
    quantizer_summary,
    safe_array,
    safe_get_config,
    weight_stats,
)


def sparsity(kernel, tol=1e-12):
    return weight_stats(kernel, include_histogram=False).get("sparsity")
