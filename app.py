import os

import dash
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Input, Output, dcc, html

BACKENDS = [
    "faiss_flat",
    "faiss_ivf_pq",
    "milvus_hnsw",
    "milvus_hnsw_sq",
    "milvus_hnsw_pq",
    "milvus_ivf_flat",
    "milvus_ivf_sq8",
    "milvus_ivf_pq",
    "weviate_hnsw",
    "qdrant_hnsw",
]

BACKENDS_INAT24 = [
    "faiss",
    "milvus",
    "qdrant",
    "weaviate",
]

INDEXES = ["flat", "ivf", "hnsw"]

STEPS = ["build", "search", "update-add", "update-delete"]

QUANTIZERS = ["pq", "sq", "sq8"]

TOPK_VALUES = [10, 20, 50, 100]

SEARCH_COUNT = 20000
ADD_COUNT = 50
DEL_COUNT = 30

def extract_backend_name(s: str) -> str:
    for backend in BACKENDS_INAT24:
        if backend.lower() in s.lower():
            return backend
    return ""

def extract_index_name(s: str) -> str:
    for index in INDEXES:
        if index.lower() in s.lower():
            return index
    return ""


def extract_quantizer(s: str) -> str:
    for q in QUANTIZERS:
        if q.lower() in s.lower():
            return q
    return ""

def extract_step_name(s: str) -> str:
    for step_name in STEPS:
        if step_name.lower() in s.lower():
            return step_name
    return ""

def load_result_files(res_dir: str) -> list:
    """Parse step-wise results in that result directory"""
    u = []

    res_files = os.listdir(res_dir)
    for res_file in res_files:
        if not res_file.endswith(".json"):
            continue

        # Parse JSON result file
        with open(os.path.join(res_dir, res_file), "r") as io:
            r = json.load(io)

        u.append(r)

    df = pd.DataFrame(u)
    return df


def load_results():
    results = []

    # Figure out available datasets, embedding models, and top-k values
    dataset_names = os.listdir("results")
    for dname in dataset_names:
        embedding_models = os.listdir(os.path.join("results", dname))
        for embedding_model in embedding_models:
            topk_folders = os.listdir(os.path.join("results", dname, embedding_model))
            topk_values = sorted([
                int(folder.split("_")[-1]) for folder in topk_folders
            ])

            for k in topk_values:
                backend_res_dirs = os.listdir(os.path.join("results", dname, embedding_model, f"topk_{k}"))
                for backend_res_dir in backend_res_dirs:
                    if not os.path.isdir(os.path.join("results", dname, embedding_model, f"topk_{k}", backend_res_dir)):
                        continue
                    backend_name = extract_backend_name(backend_res_dir)
                    df = load_result_files(os.path.join("results", dname, embedding_model, f"topk_{k}", backend_res_dir))
                    df["dataset"] = dname
                    df["embedding_model"] = embedding_model
                    df["topk"] = k
                    results.append(df)
    
    df = pd.concat(results, ignore_index=True)

    # Cleanup a few things
    df["backend"] = df["step"].apply(extract_backend_name)
    df["index_type"] = df["step"].apply(extract_index_name)
    df["quantizer"] = df["step"].apply(extract_quantizer)
    df["step_name"] = df["step"].apply(extract_step_name)
    df["backend_long"] = df["backend"] + "_" + df["index_type"] + "_" + df["quantizer"]
    df["backend_long"] = df["backend_long"].apply(lambda s: s.rstrip("_"))

    # Compute average times for search and updates
    df.loc[df["step_name"] == "search", "wall_time_sec"] = df.loc[df["step_name"] == "search", "wall_time_sec"] / SEARCH_COUNT
    df.loc[df["step_name"] == "update-add", "wall_time_sec"] = df.loc[df["step_name"] == "update-add", "wall_time_sec"] / ADD_COUNT
    df.loc[df["step_name"] == "update-delete", "wall_time_sec"] = df.loc[df["step_name"] == "update-delete", "wall_time_sec"] / DEL_COUNT

    # Sort values
    df.sort_values(by=["dataset", "embedding_model", "backend", "quantizer", "topk", "step_name"], inplace=True)

    return df


df = load_results()

# # Pre-build figures
# fig_build_time = px.bar(
#     df[(df["step_name"] == "build") & (df["k"] == 10)],
#     x="backend",
#     y="wall_time_sec",
#     color="backend",
#     labels={"wall_time_sec": "Build time (s)", "k": "top-k neighbors"},
# )
# fig_build_memory = px.bar(
#     df[(df["step"] == "build") & (df["k"] == 10)],
#     x="backend",
#     y="memory",
#     color="backend",
#     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
# )

# # Search figures
# fig_search_time = px.line(
#     df[(df["step"] == "search")],
#     x="k",
#     y="wall_time_sec",
#     color="backend",
#     labels={"wall_time_sec": "Average search time (ms)", "k": "top-k neighbors"},
# ).update_traces(mode="lines+markers")
# fig_search_memory = px.line(
#     df[(df["step"] == "search")],
#     x="k",
#     y="memory",
#     color="backend",
#     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
# ).update_traces(mode="lines+markers")

# # Update-add
# fig_update_add_time = px.bar(
#     df[(df["step"] == "update-add") & (df["k"] == 10)],
#     x="backend",
#     y="wall_time_sec",
#     color="backend",
#     labels={"wall_time_sec": "Average update time (ms)", "k": "top-k neighbors"},
# )
# fig_update_add_memory = px.bar(
#     df[(df["step"] == "update-add") & (df["k"] == 10)],
#     x="backend",
#     y="memory",
#     color="backend",
#     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
# )

# # Update-del
# fig_update_del_time = px.bar(
#     df[(df["step"] == "update-del") & (df["k"] == 10)],
#     x="backend",
#     y="wall_time_sec",
#     color="backend",
#     labels={"wall_time_sec": "Average update time (ms)", "k": "top-k neighbors"},
# )
# fig_update_del_memory = px.bar(
#     df[(df["step"] == "update-del") & (df["k"] == 10)],
#     x="backend",
#     y="memory",
#     color="backend",
#     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
# )

# Dash app setup
app = dash.Dash("Vector DB Benchmark")

app.layout = html.Div(
    [
        html.H1("Vector database benchmark"),
        html.Div(
            [
                html.Label("Select dataset:"),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=sorted(df["dataset"].unique()),
                    value="inat25-5m",
                ),
            ],
            # style={"width": "15%", "marginLeft": "1%"},
        ),
        html.Div(
            [
                html.Label("Select embedding model:"),
                dcc.Dropdown(
                    id="embedding-dropdown",
                    options=["clip", "vit-b-16"],
                    value="",
                ),
            ],
        ),
        # # Dataset information
        # html.H2("Dataset"),
        # html.Ul(
        #     [
        #         # Show dataset URL
        #         html.Li(
        #             "https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small"
        #         ),
        #         # Dataset stats: number of vectors, vector dimension, vector type, distance metric
        #         html.Li("Number of vectors: 20,000"),
        #         html.Li("Vector dimension: 768"),
        #     ]
        # ),
        # Benchmark results
        html.H2("Index build"),
        html.Div(
            [
                dcc.Graph(
                    id="fig-build-time",
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
                dcc.Graph(
                    id="fig-build-memory",
                    # figure=fig_build_memory,
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
        html.Div(children=f"Average search time (in s) across {SEARCH_COUNT} queries"),
        html.Div(
            [
                dcc.Graph(
                    id="fig-search-time",
                    # figure=fig_search_time,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
                dcc.Graph(
                    id="fig-search-memory",
                    # figure=fig_search_memory,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
            ]
        ),
        html.H2("Update-add performance"),
        html.Div(
            children=f"Average update time (in ms) to add {ADD_COUNT} data points"
        ),
        html.Div(
            [
                dcc.Graph(
                    id="fig-update_add-time",
                    # figure=fig_update_add_time,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
                dcc.Graph(
                    id="fig-update_add-memory",
                    # figure=fig_update_add_memory,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
            ]
        ),
        html.H2("Update-delete performance"),
        html.Div(
            children=f"Average update time (in ms) to delete {DEL_COUNT} data points"
        ),
        html.Div(
            [
                dcc.Graph(
                    id="fig-update_del-time",
                    # figure=fig_update_del_time,
                    style={
                        # Setting max width, max height and margin bottom.
                        "max-width": "800px",
                        "display": "inline-block",
                        "width": "49%",
                    },
                ),
                dcc.Graph(
                    id="fig-update_del-memory",
                    # figure=fig_update_del_memory,
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

@app.callback(
    Output("fig-build-time", "figure"),
    Output("fig-build-memory", "figure"),
    Output("fig-search-time", "figure"),
    Output("fig-search-memory", "figure"),
    Output("fig-update_add-time", "figure"),
    Output("fig-update_add-memory", "figure"),
    Output("fig-update_del-time", "figure"),
    Output("fig-update_del-memory", "figure"),
    Input("dataset-dropdown", "value"),
    Input("embedding-dropdown", "value"),
)
def update_figures(selected_dataset, selected_embedding):

    _df = df[(df["dataset"] == selected_dataset) & (df["embedding_model"] == selected_embedding)]

    # Bar chart of build time
    _df_build = _df[_df["step_name"] == "build"]
    tr_build_time = go.Bar(x=_df_build["backend_long"], y=_df_build["wall_time_sec"])
    fig_build_time = {
        "data": [tr_build_time],
        "layout": go.Layout(
            title="Index build time (s)",
            xaxis={"title": "VectorDB backend"},
            yaxis={
                "title": "Build time (s)",
            },
            hovermode="closest",
        ),
    }
    tr_build_mem = go.Bar(x=_df_build["backend_long"], y=_df_build["py_heap_peak_mb"])
    fig_build_mem = {
        "data": [tr_build_mem],
        "layout": go.Layout(
            title="Index build memory (Mb)",
            xaxis={"title": "VectorDB backend"},
            yaxis={
                "title": "Max memory consumption (Mb)",
                "type": "log",
            },
            hovermode="closest",
        ),
    }

    # Search performance
    _df_search = _df[_df["step_name"] == "search"]
    tr_search_time = go.Bar(x=_df_search["backend_long"], y=_df_search["wall_time_sec"])
    fig_search_time = {
        "data": [tr_search_time],
        "layout": go.Layout(
            title="Search time",
            xaxis={"title": "VectorDB backend"},
            yaxis={
                "title": "Average time / search (s)",
            },
            hovermode="closest",
        ),
    }
    tr_search_mem = go.Bar(x=_df_search["backend_long"], y=_df_search["rss_avg_mb"])
    fig_search_mem = {
        "data": [tr_search_mem],
        "layout": go.Layout(
            title="Search memory (Mb)",
            xaxis={"title": "VectorDB backend"},
            yaxis={
                "title": "Search memory",
            },
            hovermode="closest",
        ),
    }


    return fig_build_time, fig_build_mem, fig_search_time, fig_search_mem, {}, {}, {}, {}

    # fig_build_time = px.bar(
    #     df[(df["step_name"] == "build") & (df["k"] == 10)],
    #     x="backend",
    #     y="wall_time_sec",
    #     color="backend",
    #     labels={"wall_time_sec": "Build time (s)", "k": "top-k neighbors"},
    # )
    # fig_build_memory = px.bar(
    #     df[(df["step"] == "build") & (df["k"] == 10)],
    #     x="backend",
    #     y="memory",
    #     color="backend",
    #     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
    # )

    # # Search figures
    # fig_search_time = px.line(
    #     df[(df["step"] == "search")],
    #     x="k",
    #     y="wall_time_sec",
    #     color="backend",
    #     labels={"wall_time_sec": "Average search time (ms)", "k": "top-k neighbors"},
    # ).update_traces(mode="lines+markers")
    # fig_search_memory = px.line(
    #     df[(df["step"] == "search")],
    #     x="k",
    #     y="memory",
    #     color="backend",
    #     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
    # ).update_traces(mode="lines+markers")

    # # Update-add
    # fig_update_add_time = px.bar(
    #     df[(df["step"] == "update-add") & (df["k"] == 10)],
    #     x="backend",
    #     y="wall_time_sec",
    #     color="backend",
    #     labels={"wall_time_sec": "Average update time (ms)", "k": "top-k neighbors"},
    # )
    # fig_update_add_memory = px.bar(
    #     df[(df["step"] == "update-add") & (df["k"] == 10)],
    #     x="backend",
    #     y="memory",
    #     color="backend",
    #     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
    # )

    # # Update-del
    # fig_update_del_time = px.bar(
    #     df[(df["step"] == "update-del") & (df["k"] == 10)],
    #     x="backend",
    #     y="wall_time_sec",
    #     color="backend",
    #     labels={"wall_time_sec": "Average update time (ms)", "k": "top-k neighbors"},
    # )
    # fig_update_del_memory = px.bar(
    #     df[(df["step"] == "update-del") & (df["k"] == 10)],
    #     x="backend",
    #     y="memory",
    #     color="backend",
    #     labels={"memory": "Memory (Mb)", "k": "top-k neighbors"},
    # )

if __name__ == "__main__":
    app.run(debug=True)


# Dead code
"""
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
                            "wall_time_sec": 1000
                            * r["wall_time_sec"]
                            / SEARCH_COUNT,  # search time in milleseconds
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
                            "wall_time_sec": 1000
                            * r["wall_time_sec"]
                            / ADD_COUNT,  # update time in milleseconds
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
                            "wall_time_sec": 1000
                            * r["wall_time_sec"]
                            / DEL_COUNT,  # update time in milleseconds
                            "memory": r["py_heap_peak_mb"],
                        }
                    )

    df = pd.DataFrame(u)
    df.sort_values(by=["backend", "k", "step"], inplace=True)

    return results, df
"""