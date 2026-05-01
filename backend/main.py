"""
KGE Visualization Backend
FastAPI server that parses the visualization_data.csv and serves
graph data, stats, and filtered queries to the frontend.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io, os

app = FastAPI(title="KGE Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = {"df": None}


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required = {"head", "relation", "tail", "score", "head_x", "head_y", "tail_x", "tail_y"}
        missing  = required - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Normalize scores to [0,1] — RotatE scores are negative
        df["score_raw"] = df["score"]
        df["score"]     = -df["score"]
        s_min, s_max    = df["score"].min(), df["score"].max()
        df["score"]     = ((df["score"] - s_min) / (s_max - s_min)).round(6)

        store["df"] = df

        relations = df["relation"].value_counts().to_dict()
        entities  = set(df["head"].tolist() + df["tail"].tolist())

        return {
            "ok":         True,
            "rows":       len(df),
            "entities":   len(entities),
            "relations":  len(relations),
            "rel_counts": relations,
            "columns":    df.columns.tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    df = store["df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    entities  = set(df["head"].tolist() + df["tail"].tolist())
    relations = df["relation"].value_counts().to_dict()

    degree = {}
    for _, row in df.iterrows():
        degree[row["head"]] = degree.get(row["head"], 0) + 1
        degree[row["tail"]] = degree.get(row["tail"], 0) + 1

    top_entities = sorted(degree.items(), key=lambda x: -x[1])[:10]

    return {
        "total_triples":   len(df),
        "total_entities":  len(entities),
        "total_relations": len(relations),
        "rel_counts":      relations,
        "score_min":       round(float(df["score"].min()), 4),
        "score_max":       round(float(df["score"].max()), 4),
        "score_mean":      round(float(df["score"].mean()), 4),
        "score_raw_mean":      round(float(df["score_raw"].mean()), 4),
        "top_entities":    [{"entity": e, "degree": d} for e, d in top_entities],
    }


# ── Graph ─────────────────────────────────────────────────────────────────────
@app.get("/api/graph")
def get_graph(
    relation:  str   = "all",
    min_score: float = 0.0,
    search:    str   = "",
    limit:     int   = 300,
):
    df = store["df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    filtered = df.copy()

    if relation != "all":
        filtered = filtered[filtered["relation"] == relation]

    if min_score > 0:
        filtered = filtered[filtered["score"] >= min_score]

    if search:
        mask = (
            filtered["head"].str.contains(search, case=False, na=False) |
            filtered["tail"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    # ── FIX 1: Sample proportionally from each relation so all show up ────────
    # Instead of sort-by-score + head(limit), sample evenly across relations
    if relation == "all" and len(filtered) > limit:
        rel_groups  = filtered.groupby("relation")
        n_rels      = len(rel_groups)
        per_rel     = max(1, limit // n_rels)
        sampled     = pd.concat([
            grp.sort_values("score", ascending=False).head(per_rel)
            for _, grp in rel_groups
        ])
        filtered = sampled
    else:
        filtered = filtered.sort_values("score", ascending=False).head(limit)

    # Build nodes
    node_map = {}
    for _, row in filtered.iterrows():
        for name, x, y in [(row["head"], row["head_x"], row["head_y"]),
                           (row["tail"], row["tail_x"], row["tail_y"])]:
            if name not in node_map:
                node_map[name] = {"id": name, "x": float(x), "y": float(y), "degree": 0}
            node_map[name]["degree"] += 1

    edges = [
        {
            "source":   row["head"],
            "target":   row["tail"],
            "relation": row["relation"],
            "score":    round(float(row["score"]), 4),
        }
        for _, row in filtered.iterrows()
    ]

    return {
        "nodes": list(node_map.values()),
        "edges": edges,
        "total_filtered": len(filtered),
    }


# ── Relations list ────────────────────────────────────────────────────────────
@app.get("/api/relations")
def get_relations():
    df = store["df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    return {"relations": df["relation"].unique().tolist()}


# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("\n  KGE Visualizer running at: http://localhost:8000\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)