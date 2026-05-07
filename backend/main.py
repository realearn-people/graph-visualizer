"""
KGE Visualization Backend
FastAPI server — serves graph data and ABA tree structure
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io, os

app = FastAPI(title="KGE Visualizer API")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store ───────────────────────────────────────────────────────────
store = {
    "df":      None,   # visualization_data.csv  (Graph View)
    "df_tree": None,   # aba_tree_structure.csv  (Tree View)
}

CLAIMS = [
    'good_staff',    'bad_staff',
    'good_price',    'bad_price',
    'good_check-in', 'bad_check-in',
    'good_check-out','bad_check-out',
]


# ══════════════════════════════════════════════════════════════════════════════
# Upload — visualization_data.csv  (Graph View)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required = {"head", "relation", "tail", "score",
                    "head_x", "head_y", "tail_x", "tail_y"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Clean nulls and duplicates
        #df = df.dropna(subset=["head", "tail", "score", "head_x", "head_y", "tail_x", "tail_y"])
        #df = df.drop_duplicates()

        # Normalize scores to [0,1] per relation group
        df["score_raw"] = df["score"]
        df["score"]     = -df["score"]

        def normalize_group(g):
            s_min, s_max = g.min(), g.max()
            if s_max == s_min:
                return g * 0 + 1.0
            return ((g - s_min) / (s_max - s_min)).round(6)

        df["score"] = df.groupby("relation")["score"].transform(normalize_group)

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


# ══════════════════════════════════════════════════════════════════════════════
# Upload — aba_tree_structure.csv  (Tree View)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/upload/tree")
async def upload_tree_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required = {"claim", "body", "attacker", "body_relation",
                    "attack_relation", "body_score", "attack_score"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Replace NaN with None-friendly strings for JSON
        df = df.where(pd.notnull(df), None)

        store["df_tree"] = df

        return {
            "ok":       True,
            "rows":     len(df),
            "claims":   df["claim"].nunique(),
            "bodies":   df["body"].nunique(),
            "attackers":df["attacker"].nunique(),
            "claim_list": sorted(df["claim"].unique().tolist()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Stats
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/stats")
def get_stats():
    df = store["df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No graph data loaded")

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
        "score_raw_mean":  round(float(df["score_raw"].mean()), 4),
        "top_entities":    [{"entity": e, "degree": d} for e, d in top_entities],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Graph — for Graph View (visualization_data.csv)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/graph")
def get_graph(
    relation:  str   = "all",
    min_score: float = 0.0,
    search:    str   = "",
    limit:     int   = -1,
):
    df = store["df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No graph data loaded")

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

    # Limit — skip if -1 (unlimited)
    if limit > 0 and len(filtered) > limit:
        if relation == "all":
            rel_groups = filtered.groupby("relation")
            n_rels     = len(rel_groups)
            per_rel    = max(1, limit // n_rels)
            filtered   = pd.concat([
                grp.sort_values("score", ascending=False).head(per_rel)
                for _, grp in rel_groups
            ])
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
        "nodes":          list(node_map.values()),
        "edges":          edges,
        "total_filtered": len(filtered),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tree — for Tree View (aba_tree_structure.csv)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/tree")
def get_tree(
    claim:         str   = "all",
    min_body_score:float  = 0.0,
    attack_relation:str  = "all",
    domain:        str   = "all",
):
    df = store["df_tree"]
    if df is None:
        raise HTTPException(status_code=404, detail="No tree data loaded. Please upload aba_tree_structure.csv")

    filtered = df.copy()

    # Filter by specific claim
    if claim != "all":
        filtered = filtered[filtered["claim"] == claim]

    # Filter by min body score
    if min_body_score > 0:
        filtered = filtered[filtered["body_score"].notna()]
        filtered = filtered[filtered["body_score"] >= min_body_score]

    # Filter by attack relation type
    if attack_relation != "all":
        filtered = filtered[filtered["attack_relation"] == attack_relation]

    # Filter by domain
    if domain != "all" and "domain" in filtered.columns:
        filtered = filtered[filtered["domain"] == domain]

    # Build tree structure per claim
    trees = {}
    for claim_name, group in filtered.groupby("claim"):
        bodies = {}
        for _, row in group.iterrows():
            body = row["body"]
            if body not in bodies:
                bodies[body] = {
                    "name":         body,
                    "body_score":   row["body_score"],
                    "relation":     row["body_relation"],
                    "attackers":    [],
                }
            if row["attacker"] is not None and pd.notna(row["attacker"]):
                bodies[body]["attackers"].append({
                    "name":     row["attacker"],
                    "relation": row["attack_relation"],
                    "score":    row["attack_score"],
                })

        trees[claim_name] = {
            "claim":  claim_name,
            "bodies": list(bodies.values()),
            "stats":  {
                "total_bodies":    len(bodies),
                "total_attackers": sum(len(b["attackers"]) for b in bodies.values()),
                "contrary_count":  int(group[group["attack_relation"] == "CONTRARY_TO"].shape[0]),
                "not_contrary_count": int(group[group["attack_relation"] == "NOT_CONTRARY"].shape[0]),
                "support_count":   int(group[group["attack_relation"] == "SUPPORT"].shape[0]),
            }
        }

    # Get available filter options
    domains = []
    if "domain" in df.columns:
        domains = sorted(df["domain"].dropna().unique().tolist())

    return {
        "trees":          trees,
        "available_claims":  sorted(df["claim"].unique().tolist()),
        "available_domains": domains,
        "total_rows":     len(filtered),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Relations list
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/relations")
def get_relations():
    df = store["df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    return {"relations": df["relation"].unique().tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# Serve frontend
# ══════════════════════════════════════════════════════════════════════════════
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("\n  KGE Visualizer running at: http://localhost:8000\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)