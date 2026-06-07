import sys
import types


def _install_main_import_stubs():
    sys.modules.setdefault("keras", types.SimpleNamespace(models=types.SimpleNamespace(load_model=None)))
    sys.modules.setdefault("hgq", types.SimpleNamespace())
    sys.modules.setdefault("heterograph", types.SimpleNamespace(HGraph=object))
    sys.modules.setdefault(
        "heterograph.webview",
        types.SimpleNamespace(WebView=object),
    )
    sys.modules.setdefault("da4ml", types.SimpleNamespace())
    sys.modules.setdefault(
        "da4ml.converter",
        types.SimpleNamespace(trace_model=None),
    )
    sys.modules.setdefault(
        "da4ml.trace",
        types.SimpleNamespace(comb_trace=None),
    )


def test_save_graph_png_writes_png_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    _install_main_import_stubs()

    from main import save_graph_png

    class FakeGraph:
        vertices = [0, 1]
        edges = [(0, 1)]
        pmap = {
            0: {"name": "source"},
            1: {"name": "sink"},
            (0, 1): {"name": "flow"},
        }
        vstyle = {
            "label": lambda graph, vertex: graph.pmap[vertex]["name"],
            "fillcolor": lambda graph, vertex: "#64B5F6",
            "fontcolor": lambda graph, vertex: "black",
        }
        estyle = {
            "label": lambda graph, edge: graph.pmap[edge]["name"],
            "color": lambda graph, edge: "#666666",
        }

    output_path = save_graph_png(FakeGraph(), "Example Graph", tmp_path)

    assert output_path.name == "example_graph.png"
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
