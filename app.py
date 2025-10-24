import os

import dash
import json
import pandas as pd
import plotly.express as px
from dash import dcc, html

BACKENDS = [
    "milvus_ivf_flat",
    "milvus_hnsw_pq",
    "faiss_flat",
    "milvus_ivf_sq8",
    "qdrant_hnsw",
    "faiss_ivf_pq",
    "milvus_hnsw",
    "milvus_hnsw_sq",
    "weviate_hnsw",
    "milvus_ivf_pq",
]

TOPK_VALUES = [10, 20, 50, 100]

SEARCH_COUNT = 20000
ADD_COUNT = 50
DEL_COUNT = 30



def load_results():
    results = {}
    u = []

    for k in TOPK_VALUES:
        results[k] = res_k = {}
        backend_res_dirs = os.listdir(os.path.join("results", f"topk_{k}"))

        for backend in BACKENDS:
            res_dir = os.path.join(
                "results", f"topk_{k}", backend + ("_results" if k == 10 else "")
            )
            if not os.path.isdir(res_dir):
                continue

            res = res_k[backend] = {}
            # parse results
            res_files = os.listdir(res_dir)
            for fres in res_files:
                if not fres.endswith(".json"):
                    continue

                with open(os.path.join(res_dir, fres), "r") as io:
                    r = json.load(io)

                if "build" in fres:
                    res["build"] = r
                    u.append(
                        {
                            "backend": backend,
                            "k": k,
                            "step": "build",
                            "wall_time_sec": r["wall_time_sec"],
                            "memory": r["py_heap_peak_mb"],
                        }
                    )
                elif "search" in fres:
                    res["search"] = r
                    u.append(
                        {
                            "backend": backend,
                            "k": k,
                            "step": "search",
                            "wall_time_sec": 1000 * r["wall_time_sec"] / SEARCH_COUNT,  # search time in milleseconds
                            "memory": r["py_heap_peak_mb"],
                        }
                    )
                elif "update-add" in fres:
                    res["update-add"] = r
                    u.append(
                        {
                            "backend": backend,
                            "k": k,
                            "step": "update-add",
                            "wall_time_sec": 1000 * r["wall_time_sec"] / ADD_COUNT,  # update time in milleseconds
                            "memory": r["py_heap_peak_mb"],
                        }
                    )
                elif "update-del" in fres:
                    res["update-del"] = r
                    u.append(
                        {
                            "backend": backend,
                            "k": k,
                            "step": "update-del",
                            "wall_time_sec": 1000 * r["wall_time_sec"] / DEL_COUNT,  # update time in milleseconds
                            "memory": r["py_heap_peak_mb"],
                        }
                    )

    df = pd.DataFrame(u)
    df.sort_values(by=["backend", "k", "step"], inplace=True)

    return results, df


results, df = load_results()

# Pre-build figures
fig_build_time = px.bar(
    df[(df["step"] == "build") & (df["k"] == 10)],
    x="backend",
    y="wall_time_sec",
)
fig_build_memory = px.bar(
    df[(df["step"] == "build") & (df["k"] == 10)],
    x="backend",
    y="memory",
)

fig_search_time = px.line(
    df[(df["step"] == "search")],
    x="k",
    y="wall_time_sec",
    color="backend",
)
fig_search_memory = px.line(
    df[(df["step"] == "search")],
    x="k",
    y="memory",
    color="backend",
)

# Dash app setup
app = dash.Dash("Vector DB Benchmark")

app.layout = html.Div(
    [
        html.H1("Vector database benchmark"),
        html.H2("Index build"),
        html.Div(
            [
                dcc.Graph(
                    id="fig-build-time",
                    figure=fig_build_time,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
                dcc.Graph(
                    id="fig-build-memory",
                    figure=fig_build_memory,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
            ]
        ),
        html.H2("Search performance"),
        html.Div(
            [
                dcc.Graph(
                    id="fig-build-time",
                    figure=fig_search_time,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
                dcc.Graph(
                    id="fig-build-memory",
                    figure=fig_search_memory,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
