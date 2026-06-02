import json
import time
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="CLEVER", page_icon="⚛️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .panel {
        border: 1px solid #e6e9ef;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: #ffffff;
    }
    .metric-card {
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 0.6rem 0.7rem;
        background: #fafbfc;
        height: 84px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
    }
    .answer-box {
        border: 1px solid #dbe4f0;
        border-left: 4px solid #2563eb;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        background: #fbfdff;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .tiny-note {
        color: #6b7280;
        font-size: 0.8rem;
        line-height: 1.5;
    }
    .status-pill {
        display: block;
        width: 100%;
        box-sizing: border-box;
        padding: 0.5rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        border: 1px solid #dbe4f0;
        font-weight: 600;
        margin-top: 0.35rem;
        line-height: 1.35;
        word-break: break-word;
        white-space: normal;
    }
    .ok-pill {
        color: #166534;
        background: #f0fdf4;
        border-color: #bbf7d0;
    }
    .warn-pill {
        color: #991b1b;
        background: #fef2f2;
        border-color: #fecaca;
    }
    div[data-testid="stTextArea"] > label,
    div[data-testid="stSelectbox"] > label,
    div[data-testid="stNumberInput"] > label {
        font-size: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Constants
# ============================================================
SUPPORTED_EXTENSIONS = [
    ".f90", ".f", ".for", ".f95", ".txt", ".py", ".c", ".cpp", ".h", ".hpp", ".json", ".md"
]

NODE_COLOR_MAP = {
    "System": "#F59E0B",
    "File": "#0F7B45",
    "Equation": "#2E4A9E",
    "Global Variable": "#F59E0B",
    "Local Variable": "#F59E0B",
    "Constant": "#F59E0B",
    "Unknown": "#BDBDBD",
}

LOW_CONTRAST_NODE_COLOR_MAP = {
    "System": "#C9B28A",
    "File": "#6C9A7D",
    "Equation": "#667CB3",
    "Global Variable": "#D7A85A",
    "Local Variable": "#D7A85A",
    "Constant": "#D7A85A",
    "Unknown": "#C7CBD1",
}

ALLOWED_NODE_TYPES = {
    "System",
    "File",
    "Equation",
    "Global Variable",
    "Local Variable",
    "Constant",
}

ALLOWED_RELATIONS = {
    "has",
    "encodes",
    "hasVariable",
    "hasConstant",
    "contains",
    "belongs_to",
    "related_to",
}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemma-4-26b-a4b-it"

DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_CHUNK_SIZE = 2500
DEFAULT_SYSTEM_NAME = "Sci-KG"


# ============================================================
# Secrets / env
# ============================================================
def get_gemini_api_key() -> str:
    key = st.secrets.get("GEMINI_API_KEY", "").strip()
    return key


# ============================================================
# Data classes
# ============================================================
@dataclass
class GraphData:
    nodes: pd.DataFrame
    edges: pd.DataFrame


def empty_graph() -> GraphData:
    return GraphData(
        nodes=pd.DataFrame(columns=["id", "label", "type"]),
        edges=pd.DataFrame(columns=["source", "target", "relation"]),
    )


# ============================================================
# Session state
# ============================================================
if "graph_data" not in st.session_state:
    st.session_state.graph_data = empty_graph()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []
if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = ""
if "last_health_message" not in st.session_state:
    st.session_state.last_health_message = "Not checked yet"
if "last_health_ok" not in st.session_state:
    st.session_state.last_health_ok = False


# ============================================================
# Helpers
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def cached_check_gemini(model: str, api_key: str, timeout: int = 20) -> tuple[bool, str]:
    if not api_key:
        return False, "Missing GEMINI_API_KEY in Streamlit secrets."

    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [{"text": "Reply only with OK"}]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )

        if response.ok:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return False, f"No candidates returned: {data}"
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
            return True, text or "OK"

        try:
            err = response.json()
            return False, f"{response.status_code} {err}"
        except Exception:
            return False, f"{response.status_code} {response.text}"

    except Exception as exc:
        return False, str(exc)


def allowed_extension(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def safe_json_load(text: str) -> dict:
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            return json.loads(candidate)
        raise ValueError("Could not parse JSON from LLM response.")


def canonical_text(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def canonical_node_key(label: str, node_type: str) -> tuple[str, str]:
    return (canonical_text(label), canonical_text(node_type))


def validate_fragment(fragment: dict) -> dict:
    nodes = fragment.get("nodes", [])
    edges = fragment.get("edges", [])

    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    valid_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        label = str(node.get("label", node_id)).strip()
        node_type = str(node.get("type", "Unknown")).strip()
        if not node_id:
            continue
        if node_type not in ALLOWED_NODE_TYPES:
            node_type = "Unknown"
        valid_nodes.append({"id": node_id, "label": label or node_id, "type": node_type})

    valid_node_ids = {node["id"] for node in valid_nodes}
    valid_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        relation = str(edge.get("relation", "related_to")).strip()
        if not source or not target:
            continue
        if relation not in ALLOWED_RELATIONS:
            relation = "related_to"
        if source in valid_node_ids and target in valid_node_ids:
            valid_edges.append({"source": source, "target": target, "relation": relation})

    return {"nodes": valid_nodes, "edges": valid_edges}


def deduplicate_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> GraphData:
    if nodes_df.empty:
        return empty_graph()

    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()

    for col in ["id", "label", "type"]:
        if col not in nodes_df.columns:
            nodes_df[col] = ""

    nodes_df["id"] = nodes_df["id"].astype(str)
    nodes_df["label"] = nodes_df["label"].astype(str)
    nodes_df["type"] = nodes_df["type"].astype(str)

    extra_cols = [c for c in nodes_df.columns if c not in {"id", "label", "type"}]

    alias_map: dict[str, str] = {}
    canonical_rep: dict[tuple[str, str], str] = {}
    canonical_rows: list[dict] = []

    for _, row in nodes_df.iterrows():
        node_id = row["id"]
        label = row["label"]
        node_type = row["type"]
        key = canonical_node_key(label, node_type)

        if key not in canonical_rep:
            canonical_rep[key] = node_id
            new_row = {"id": node_id, "label": label, "type": node_type}
            for col in extra_cols:
                new_row[col] = row[col]
            canonical_rows.append(new_row)

        alias_map[node_id] = canonical_rep[key]

    if edges_df.empty:
        edges_df = pd.DataFrame(columns=["source", "target", "relation"])
    else:
        for col in ["source", "target", "relation"]:
            if col not in edges_df.columns:
                edges_df[col] = ""
        edges_df["source"] = edges_df["source"].astype(str).map(lambda x: alias_map.get(x, x))
        edges_df["target"] = edges_df["target"].astype(str).map(lambda x: alias_map.get(x, x))
        edges_df["relation"] = edges_df["relation"].astype(str)
        edges_df = edges_df[edges_df["source"] != edges_df["target"]]
        edges_df = edges_df.drop_duplicates()

    canonical_nodes_df = pd.DataFrame(canonical_rows).drop_duplicates(subset=["id"])
    return GraphData(nodes=canonical_nodes_df, edges=edges_df)


def normalize_fragment(fragment: dict) -> GraphData:
    fragment = validate_fragment(fragment)
    nodes = pd.DataFrame(fragment.get("nodes", []))
    edges = pd.DataFrame(fragment.get("edges", []))

    if nodes.empty:
        nodes = pd.DataFrame(columns=["id", "label", "type"])
    if edges.empty:
        edges = pd.DataFrame(columns=["source", "target", "relation"])

    return deduplicate_graph(nodes, edges)


def merge_graphs(graphs: list[GraphData]) -> GraphData:
    if not graphs:
        return empty_graph()
    all_nodes = pd.concat([g.nodes for g in graphs], ignore_index=True)
    all_edges = pd.concat([g.edges for g in graphs], ignore_index=True)
    return deduplicate_graph(all_nodes, all_edges)


def graph_context_text(graph: GraphData, limit_nodes: int = 220, limit_edges: int = 350) -> str:
    nodes = graph.nodes.head(limit_nodes).to_dict(orient="records")
    edges = graph.edges.head(limit_edges).to_dict(orient="records")
    return json.dumps({"nodes": nodes, "edges": edges}, indent=2)


def sample_placeholder_graph() -> GraphData:
    nodes = pd.DataFrame([
        {"id": "sys_scikg", "label": "Sci-KG", "type": "System", "x": -0.15, "y": 0.95, "z": -0.25},
        {"id": "file_diff", "label": "diff_fusion_solver.F90", "type": "File", "x": -3.10, "y": -1.85, "z": -0.25},
        {"id": "file_geo", "label": "geopotential.F90", "type": "File", "x": -0.15, "y": 0.40, "z": -0.25},
        {"id": "file_ice", "label": "ice_ocean.F90", "type": "File", "x": 1.00, "y": 2.10, "z": -0.15},
        {"id": "v_Fdiff", "label": "F_diff", "type": "Local Variable", "x": -3.75, "y": -1.20, "z": 0.15},
        {"id": "eq_Fdiff", "label": "F_diff = −K_h (∂q/∂z)", "type": "Equation", "x": -3.00, "y": -1.35, "z": 0.00},
        {"id": "v_dqdz", "label": "∂q/∂z", "type": "Local Variable", "x": -2.35, "y": -0.95, "z": 0.15},
        {"id": "v_q", "label": "q", "type": "Local Variable", "x": -1.85, "y": -0.40, "z": 0.05},
        {"id": "eq_dqdy", "label": "∂q/∂y = dh · dμ_ocn / ρ_atm", "type": "Equation", "x": -2.45, "y": 0.30, "z": -0.20},
        {"id": "v_dqdy", "label": "∂q/∂y", "type": "Local Variable", "x": -1.95, "y": 0.00, "z": -0.10},
        {"id": "eq_virtem", "label": "virtem = t · (1.0 + zvir · q)", "type": "Equation", "x": -1.00, "y": -0.70, "z": -0.25},
        {"id": "eq_dqdt", "label": "∂q/∂t = (q₃ − q_minus) / δt", "type": "Equation", "x": 0.35, "y": 0.10, "z": -0.20},
        {"id": "v_dqdt", "label": "∂q/∂t", "type": "Local Variable", "x": -0.55, "y": -0.05, "z": 0.00},
        {"id": "eq_Tv1", "label": "Tᵥ = T(1 + 0.61q)", "type": "Equation", "x": -1.20, "y": 0.20, "z": 0.00},
        {"id": "v_Tv", "label": "Tᵥ", "type": "Local Variable", "x": -1.40, "y": 0.75, "z": 0.12},
        {"id": "eq_Tv2", "label": "Tᵥ = T · TVFC", "type": "Equation", "x": -0.55, "y": 1.20, "z": 0.00},
        {"id": "v_TVFC", "label": "TVFC", "type": "Local Variable", "x": -1.00, "y": 1.95, "z": -0.10},
        {"id": "v_T", "label": "T", "type": "Local Variable", "x": -0.35, "y": 1.00, "z": 0.15},
        {"id": "eq_flwout", "label": "flwout_ocn = −σ_SB T_sf⁴", "type": "Equation", "x": 0.15, "y": 1.75, "z": 0.00},
        {"id": "v_flwout", "label": "flwout_ocn", "type": "Local Variable", "x": -0.10, "y": 0.75, "z": -0.15},
        {"id": "v_sigmaSB", "label": "σ_SB", "type": "Constant", "x": 0.55, "y": 1.45, "z": -0.10},
        {"id": "v_Tsf", "label": "T_sf", "type": "Local Variable", "x": 0.10, "y": 2.55, "z": 0.12},
        {"id": "eq_Tsf", "label": "T_sf = sst + T_fresh", "type": "Equation", "x": 0.55, "y": 2.55, "z": 0.00},
        {"id": "v_Tfresh", "label": "T_fresh", "type": "Local Variable", "x": 0.05, "y": 3.20, "z": 0.05},
        {"id": "v_sst", "label": "sst", "type": "Local Variable", "x": 0.90, "y": 3.40, "z": 0.10},
        {"id": "eq_dsstdt", "label": "∂sst/∂t", "type": "Equation", "x": 1.30, "y": 3.00, "z": 0.18},
        {"id": "eq_frzmlt", "label": "frzmlt = T_f − sst", "type": "Equation", "x": 0.55, "y": 4.15, "z": 0.00},
        {"id": "v_frzmlt", "label": "frzmlt", "type": "Local Variable", "x": -0.10, "y": 4.65, "z": -0.10},
        {"id": "v_Tf", "label": "T_f", "type": "Local Variable", "x": 1.15, "y": 4.15, "z": 0.05},
    ])

    edges = pd.DataFrame([
        {"source": "sys_scikg", "target": "file_diff", "relation": "has"},
        {"source": "sys_scikg", "target": "file_geo", "relation": "has"},
        {"source": "sys_scikg", "target": "file_ice", "relation": "has"},
        {"source": "file_diff", "target": "eq_Fdiff", "relation": "encodes"},
        {"source": "file_geo", "target": "eq_Tv1", "relation": "encodes"},
        {"source": "file_geo", "target": "eq_Tv2", "relation": "encodes"},
        {"source": "file_ice", "target": "eq_flwout", "relation": "encodes"},
        {"source": "file_ice", "target": "eq_Tsf", "relation": "encodes"},
        {"source": "file_ice", "target": "eq_frzmlt", "relation": "encodes"},
        {"source": "v_Fdiff", "target": "eq_Fdiff", "relation": "hasVariable"},
        {"source": "eq_Fdiff", "target": "v_dqdz", "relation": "hasVariable"},
        {"source": "v_dqdz", "target": "v_q", "relation": "related_to"},
        {"source": "v_q", "target": "eq_Tv1", "relation": "hasVariable"},
        {"source": "eq_Tv1", "target": "v_Tv", "relation": "hasVariable"},
        {"source": "v_Tv", "target": "eq_Tv2", "relation": "hasVariable"},
        {"source": "eq_Tv2", "target": "v_T", "relation": "hasVariable"},
        {"source": "v_T", "target": "eq_flwout", "relation": "related_to"},
        {"source": "eq_flwout", "target": "v_Tsf", "relation": "hasVariable"},
        {"source": "v_Tsf", "target": "eq_Tsf", "relation": "hasVariable"},
        {"source": "eq_Tsf", "target": "v_sst", "relation": "hasVariable"},
        {"source": "v_sst", "target": "eq_dsstdt", "relation": "related_to"},
        {"source": "v_sst", "target": "eq_frzmlt", "relation": "related_to"},
        {"source": "eq_dqdy", "target": "v_dqdy", "relation": "hasVariable"},
        {"source": "v_dqdy", "target": "v_q", "relation": "related_to"},
        {"source": "eq_virtem", "target": "v_q", "relation": "hasVariable"},
        {"source": "eq_dqdt", "target": "v_dqdt", "relation": "hasVariable"},
        {"source": "v_dqdt", "target": "v_q", "relation": "related_to"},
        {"source": "eq_Tv2", "target": "v_TVFC", "relation": "hasVariable"},
        {"source": "eq_flwout", "target": "v_sigmaSB", "relation": "hasConstant"},
        {"source": "eq_Tsf", "target": "v_Tfresh", "relation": "hasVariable"},
        {"source": "eq_frzmlt", "target": "v_frzmlt", "relation": "hasVariable"},
        {"source": "eq_frzmlt", "target": "v_Tf", "relation": "hasVariable"},
        {"source": "eq_flwout", "target": "v_flwout", "relation": "hasVariable"},
    ])

    return deduplicate_graph(nodes, edges)


def get_placeholder_highlight_sets() -> tuple[set[str], set[tuple[str, str]]]:
    highlight_nodes = {
        "v_Fdiff", "eq_Fdiff", "v_dqdz", "v_q", "eq_Tv1", "v_Tv",
        "eq_Tv2", "v_T", "eq_flwout", "v_Tsf", "eq_Tsf",
        "v_sst", "eq_dsstdt", "eq_frzmlt"
    }
    red_edge_pairs = [
        ("v_Fdiff", "eq_Fdiff"),
        ("eq_Fdiff", "v_dqdz"),
        ("v_dqdz", "v_q"),
        ("v_q", "eq_Tv1"),
        ("eq_Tv1", "v_Tv"),
        ("v_Tv", "eq_Tv2"),
        ("eq_Tv2", "v_T"),
        ("v_T", "eq_flwout"),
        ("eq_flwout", "v_Tsf"),
        ("v_Tsf", "eq_Tsf"),
        ("eq_Tsf", "v_sst"),
        ("v_sst", "eq_dsstdt"),
        ("v_sst", "eq_frzmlt"),
    ]
    highlight_edges = {tuple(sorted(x)) for x in red_edge_pairs}
    return highlight_nodes, highlight_edges


def build_nx_graph(graph: GraphData, low_contrast: bool = False) -> nx.Graph:
    graph_nx = nx.Graph()
    color_map = LOW_CONTRAST_NODE_COLOR_MAP if low_contrast else NODE_COLOR_MAP
    for _, row in graph.nodes.iterrows():
        node_type = str(row["type"]) if pd.notna(row["type"]) else "Unknown"
        attrs = {
            "label": str(row["label"]),
            "node_type": node_type,
            "color": color_map.get(node_type, color_map["Unknown"]),
        }
        if "x" in row.index and "y" in row.index and pd.notna(row["x"]) and pd.notna(row["y"]):
            attrs["pos"] = (
                float(row["x"]),
                float(row["y"]),
                float(row["z"]) if "z" in row.index and pd.notna(row["z"]) else 0.0,
            )
        graph_nx.add_node(str(row["id"]), **attrs)
    for _, row in graph.edges.iterrows():
        graph_nx.add_edge(
            str(row["source"]),
            str(row["target"]),
            relation=str(row["relation"]),
        )
    return graph_nx


def subgraph_by_hops(graph_nx: nx.Graph, seed: str, hops: int = 1) -> nx.Graph:
    if seed not in graph_nx:
        return graph_nx.copy()
    visited = {seed}
    frontier = {seed}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph_nx.neighbors(node))
        frontier = next_frontier - visited
        visited.update(frontier)
    return graph_nx.subgraph(visited).copy()


def get_graph_positions(graph_nx: nx.Graph) -> dict:
    have_manual_pos = all("pos" in data for _, data in graph_nx.nodes(data=True))
    if have_manual_pos:
        return {node: data["pos"] for node, data in graph_nx.nodes(data=True)}
    return nx.spring_layout(graph_nx, dim=3, seed=12, k=1.35, iterations=200)


def render_3d_graph(
    graph_nx: nx.Graph,
    highlight_nodes: Optional[set[str]] = None,
    highlight_edges: Optional[set[tuple[str, str]]] = None,
    placeholder_mode: bool = False,
):
    if not PLOTLY_AVAILABLE or go is None:
        return None

    highlight_nodes = highlight_nodes or set()
    highlight_edges = highlight_edges or set()

    if len(graph_nx.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(height=760, margin=dict(l=0, r=0, t=10, b=0))
        return fig

    pos = get_graph_positions(graph_nx)

    ash_edge_x, ash_edge_y, ash_edge_z = [], [], []
    red_edge_x, red_edge_y, red_edge_z = [], [], []

    for source, target in graph_nx.edges():
        x0, y0, z0 = pos[source]
        x1, y1, z1 = pos[target]
        edge_key = tuple(sorted((source, target)))

        if edge_key in highlight_edges:
            red_edge_x += [x0, x1, None]
            red_edge_y += [y0, y1, None]
            red_edge_z += [z0, z1, None]
        else:
            ash_edge_x += [x0, x1, None]
            ash_edge_y += [y0, y1, None]
            ash_edge_z += [z0, z1, None]

    ash_edge_trace = go.Scatter3d(
        x=ash_edge_x,
        y=ash_edge_y,
        z=ash_edge_z,
        mode="lines",
        line=dict(color="rgba(40,40,40,0.45)", width=4),
        hoverinfo="none",
        showlegend=False,
    )

    red_edge_trace = go.Scatter3d(
        x=red_edge_x,
        y=red_edge_y,
        z=red_edge_z,
        mode="lines",
        line=dict(color="rgba(206,40,40,0.95)", width=7),
        hoverinfo="none",
        showlegend=False,
    )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    zs = [p[2] for p in pos.values()]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)

    reg_x, reg_y, reg_z = [], [], []
    reg_text, reg_colors, reg_sizes = [], [], []
    hi_x, hi_y, hi_z = [], [], []
    hi_text, hi_colors, hi_sizes = [], [], []
    label_x, label_y, label_z = [], [], []
    label_text = []

    for node, data in graph_nx.nodes(data=True):
        x, y, z = pos[node]
        label = data.get("label", node)
        node_type = data.get("node_type", "Unknown")
        hover_text = f"{label}<br>{node_type}<br>{node}"
        node_color = data.get("color", NODE_COLOR_MAP["Unknown"])
        node_size = 14 if node_type == "System" else (12 if node_type == "Equation" else 8)

        if node in highlight_nodes:
            hi_x.append(x)
            hi_y.append(y)
            hi_z.append(z)
            hi_text.append(hover_text)
            hi_colors.append(node_color)
            hi_sizes.append(node_size)
        else:
            reg_x.append(x)
            reg_y.append(y)
            reg_z.append(z)
            reg_text.append(hover_text)
            reg_colors.append(node_color)
            reg_sizes.append(node_size)

        dx = x - cx
        dy = y - cy
        dz = z - cz
        norm = (dx**2 + dy**2 + dz**2) ** 0.5
        if norm == 0:
            norm = 1.0

        offset = 0.22 if node_type == "System" else (0.18 if node_type == "Equation" else 0.12)
        label_x.append(x + offset * dx / norm)
        label_y.append(y + offset * dy / norm)
        label_z.append(z + offset * dz / norm)
        label_text.append(label)

    regular_trace = go.Scatter3d(
        x=reg_x,
        y=reg_y,
        z=reg_z,
        mode="markers",
        text=reg_text,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=reg_sizes,
            color=reg_colors,
            opacity=0.98,
            line=dict(color="rgba(60,60,60,0.30)", width=1.0),
        ),
        showlegend=False,
    )

    highlight_trace = go.Scatter3d(
        x=hi_x,
        y=hi_y,
        z=hi_z,
        mode="markers",
        text=hi_text,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=hi_sizes,
            color=hi_colors,
            opacity=1.0,
            line=dict(color="#C81E1E", width=2.3),
        ),
        showlegend=False,
    )

    label_trace = go.Scatter3d(
        x=label_x,
        y=label_y,
        z=label_z,
        mode="text",
        text=label_text,
        textfont=dict(size=12, color="#3F4752"),
        hoverinfo="none",
        showlegend=False,
    )

    fig = go.Figure(data=[ash_edge_trace, red_edge_trace, regular_trace, highlight_trace, label_trace])
    fig.update_layout(
        height=560,
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.55, y=1.35, z=1.10)),
        ),
    )
    return fig


def render_2d_graph(graph_nx: nx.Graph):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_axis_off()
    if len(graph_nx.nodes()) == 0:
        return fig

    manual = all("pos" in data for _, data in graph_nx.nodes(data=True))
    if manual:
        pos = {node: (data["pos"][0], data["pos"][1]) for node, data in graph_nx.nodes(data=True)}
    else:
        pos = nx.spring_layout(graph_nx, seed=42)

    node_colors = [data.get("color", NODE_COLOR_MAP["Unknown"]) for _, data in graph_nx.nodes(data=True)]
    labels = {node: data.get("label", node) for node, data in graph_nx.nodes(data=True)}

    nx.draw_networkx_edges(
        graph_nx,
        pos,
        ax=ax,
        alpha=0.35,
        width=1.6,
        edge_color="#2f2f2f",
    )
    nx.draw_networkx_nodes(graph_nx, pos, ax=ax, node_color=node_colors, node_size=380)
    nx.draw_networkx_labels(graph_nx, pos, labels=labels, ax=ax, font_size=8)
    fig.tight_layout()
    return fig


# ============================================================
# Gemini client
# ============================================================
class LLMClient:
    def __init__(self, model: str, api_key: str, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.model and self.api_key)

    def chat(self, system_prompt: str, user_prompt: str, response_format: Optional[dict] = None) -> str:
        if not self.is_configured():
            return "Gemini is not configured yet. Add GEMINI_API_KEY to Streamlit secrets."

        url = f"{GEMINI_BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ]
        }

        if response_format is not None and response_format.get("type") == "json_object":
            payload["generationConfig"] = {
                "responseMimeType": "application/json"
            }

        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )

                if not response.ok:
                    try:
                        err = response.json()
                    except Exception:
                        err = response.text
                    raise RuntimeError(f"Gemini error {response.status_code}: {err}")

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"No candidates returned by Gemini. Response: {data}")

                parts = candidates[0].get("content", {}).get("parts", [])
                text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
                if text:
                    return text

                raise ValueError(f"No text found in Gemini response: {data}")

            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini request failed after retries: {last_error}") from exc


# ============================================================
# Sci-KG extraction
# ============================================================
def llm_extract_file_graph(llm_client: LLMClient, system_name: str, file_name: str, file_text: str) -> dict:
    system_prompt = (
        "You extract a Scientific Knowledge Graph from scientific source files. "
        "Return only JSON. No markdown. No explanation."
    )

    user_prompt = f'''Build a Scientific Knowledge Graph fragment for CLEVER from this file.

System name: {system_name}
File name: {file_name}

Allowed node types:
- System
- File
- Equation
- Global Variable
- Local Variable
- Constant

Allowed edge relations:
- has
- encodes
- hasVariable
- hasConstant
- contains
- belongs_to
- related_to

Return strict JSON with this schema:
{{
  "nodes": [
    {{"id": "unique_id", "label": "display label", "type": "one allowed type"}}
  ],
  "edges": [
    {{"source": "node_id", "target": "node_id", "relation": "allowed relation"}}
  ]
}}

Rules:
1. Include exactly one System node using the given system name.
2. Include exactly one File node using the given file name.
3. Link System -> File with relation "has".
4. Extract equations when they are explicit or strongly implied by assignments or formulas.
5. Distinguish Global Variable vs Local Variable when reasonably inferable. If uncertain, use Local Variable.
6. Deduplicate semantically equivalent nodes within this fragment.
7. Do not invent unsupported facts beyond the file.

File content:
{file_text}
'''

    last_error = None
    for _ in range(2):
        try:
            response = llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"},
            )
            return safe_json_load(response)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Extraction failed for {file_name}: {last_error}")


# ============================================================
# Header
# ============================================================
st.title("CLEVER")
st.caption("Cross-module Latent Equation Variable Extraction and Recovery")

summary_graph = st.session_state.graph_data
summary_edges = len(summary_graph.edges)
summary_files = 0
summary_vars = 0
summary_eqs = 0

if not summary_graph.nodes.empty:
    summary_files = int(summary_graph.nodes[summary_graph.nodes["type"] == "File"].shape[0])
    summary_vars = int(
        summary_graph.nodes[
            summary_graph.nodes["type"].isin(["Global Variable", "Local Variable", "Constant"])
        ].shape[0]
    )
    summary_eqs = int(summary_graph.nodes[summary_graph.nodes["type"] == "Equation"].shape[0])

ok, msg = cached_check_gemini(
    GEMINI_MODEL,
    get_gemini_api_key(),
    20,
)
st.session_state.last_health_ok = ok
st.session_state.last_health_message = msg


# ============================================================
# Fixed defaults
# ============================================================
system_name = DEFAULT_SYSTEM_NAME


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("CLEVER Settings")

    if ok:
        st.markdown(
            '<div class="status-pill ok-pill">Gemma 4 online</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill warn-pill">Gemma 4 offline</div>',
            unsafe_allow_html=True,
        )

    # Fixed internal limits — not visible in the interface
    file_limit = 20
    focus_hops = 10
    max_nodes = 300

    st.subheader("Files")
    uploaded_files = st.file_uploader(
        "Upload scientific files",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    build_clicked = st.button("Build Sci-KG", use_container_width=True)

    st.subheader("Sci-KG Stats")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Files", summary_files)
        st.metric("Variables", summary_vars)

    with col2:
        st.metric("Equations", summary_eqs)
        st.metric("Edges", summary_edges)

    if st.button("Reset graph", use_container_width=True):
        st.session_state.graph_data = empty_graph()
        st.session_state.processed_files = []
        st.session_state.pipeline_logs = []
        st.session_state.chat_answer = ""
        st.rerun()

llm_client = LLMClient(
    model=GEMINI_MODEL,
    api_key=get_gemini_api_key(),
)


# ============================================================
# Main layout
# ============================================================
left_col, right_col = st.columns([0.82, 2.18], gap="small")

with left_col:
    graph = st.session_state.graph_data

    st.subheader("Chat")
    focus_options = [""] + sorted(graph.nodes["id"].astype(str).tolist()) if not graph.nodes.empty else [""]
    focus_node = st.selectbox("Focus graph around node", options=focus_options)

    graph_question = st.text_area(
        "Question",
        placeholder="Ask about equations, variables, dependencies, files, or scientific meaning.",
        height=140,
        label_visibility="collapsed",
    )

    if st.button("Ask CLEVER", use_container_width=True):
        if not llm_client.is_configured():
            st.error("Gemma 4 is not configured. Add GEMINI_API_KEY to Streamlit secrets.")
        elif graph_question.strip() == "":
            st.error("Enter a question first.")
        else:
            context = graph_context_text(graph)
            system_prompt = (
                "You are CLEVER, an agent over a Scientific Knowledge Graph (Sci-KG). "
                "Use the graph context first. If the graph is insufficient, you may add cautious scientific reasoning and clearly mark it as added knowledge."
            )
            user_prompt = (
                "Graph context:\n"
                f"{context}\n\n"
                f"User question: {graph_question}"
            )
            try:
                st.session_state.chat_answer = llm_client.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                st.session_state.chat_answer = f"Error: {exc}"

    if st.session_state.chat_answer:
        st.markdown(f'<div class="answer-box">{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

with right_col:
    st.subheader("Scientific Knowledge Graph")

    graph = st.session_state.graph_data
    show_placeholder = graph.nodes.empty

    if show_placeholder:
        display_graph = sample_placeholder_graph()
        graph_nx = build_nx_graph(display_graph, low_contrast=True)
        highlight_nodes, highlight_edges = get_placeholder_highlight_sets()
    else:
        display_graph = graph
        graph_nx = build_nx_graph(display_graph, low_contrast=False)
        highlight_nodes, highlight_edges = set(), set()

    if focus_node and not show_placeholder:
        view_graph = subgraph_by_hops(graph_nx, focus_node, hops=focus_hops)
    else:
        sample_nodes = list(graph_nx.nodes())[:max_nodes]
        view_graph = graph_nx.subgraph(sample_nodes).copy()

    if len(view_graph.nodes()) > max_nodes:
        trimmed_nodes = list(view_graph.nodes())[:max_nodes]
        view_graph = view_graph.subgraph(trimmed_nodes).copy()

    if show_placeholder:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown("🟤 **System**")
        with c2:
            st.markdown("🔵 **Equation**")
        with c3:
            st.markdown("🟠 **Variable**")
        with c4:
            st.markdown("🟢 **File**")
        with c5:
            st.markdown("⚫ **Edge**")
        with c6:
            st.markdown("🔴 **Path**")

    fig_3d = render_3d_graph(
        view_graph,
        highlight_nodes=highlight_nodes,
        highlight_edges=highlight_edges,
        placeholder_mode=show_placeholder,
    )

    if fig_3d is not None:
        st.plotly_chart(fig_3d, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Plotly is not installed, so the app is showing a 2D fallback graph.")
        fig_2d = render_2d_graph(view_graph)
        st.pyplot(fig_2d, use_container_width=True)


# ============================================================
# Build graph after UI inputs are bound
# ============================================================
if build_clicked:
    if not uploaded_files:
        st.error("Upload at least one scientific file.")
    elif not llm_client.is_configured():
        st.error("Gemma 4 is not configured. Add GEMINI_API_KEY to Streamlit secrets.")
    else:
        selected_files = [f for f in uploaded_files if allowed_extension(f.name)][:file_limit]
        if not selected_files:
            st.error("No supported files were selected.")
        else:
            fragments = []
            logs = []
            progress = st.progress(0)
            status_box = st.empty()
            start_all = time.time()

            for idx, uploaded in enumerate(selected_files, start=1):
                file_start = time.time()
                try:
                    raw_text = uploaded.read().decode("utf-8", errors="ignore")
                    chunks = chunk_text(raw_text, max_chars=DEFAULT_CHUNK_SIZE)
                    merged_parts = []

                    for chunk_index, chunk in enumerate(chunks, start=1):
                        status_box.info(f"Processing {uploaded.name} · chunk {chunk_index}/{len(chunks)}")
                        fragment_json = llm_extract_file_graph(
                            llm_client=llm_client,
                            system_name=system_name,
                            file_name=uploaded.name,
                            file_text=chunk,
                        )
                        merged_parts.append(normalize_fragment(fragment_json))

                    file_graph = merge_graphs(merged_parts)
                    fragments.append(file_graph)
                    logs.append({
                        "file": uploaded.name,
                        "chunks": len(chunks),
                        "nodes": len(file_graph.nodes),
                        "edges": len(file_graph.edges),
                        "seconds": round(time.time() - file_start, 2),
                        "status": "ok",
                    })
                except Exception as exc:
                    logs.append({
                        "file": uploaded.name,
                        "chunks": 0,
                        "nodes": 0,
                        "edges": 0,
                        "seconds": round(time.time() - file_start, 2),
                        "status": f"error: {exc}",
                    })

                progress.progress(idx / max(1, len(selected_files)))

            merged = merge_graphs([st.session_state.graph_data] + fragments)
            st.session_state.graph_data = merged
            st.session_state.processed_files = [log["file"] for log in logs if str(log["status"]).startswith("ok")]
            st.session_state.pipeline_logs = logs
            status_box.success(f"Scientific Knowledge Graph built in {round(time.time() - start_all, 2)} seconds.")
            st.rerun()
