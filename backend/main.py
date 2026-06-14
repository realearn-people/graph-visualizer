"""
KGE Visualization Backend
FastAPI server — serves graph data and ABA tree structure
Claude-Assisted
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
    "df":          None,   # visualization_data.csv — cleaned
    "df_raw":      None,   # visualization_data.csv — raw
    "df_tree":     None,   # aba_tree_structure.csv — cleaned
    "df_tree_raw": None,   # aba_tree_structure.csv — raw
}

CLAIMS = [
    'good_staff',    'bad_staff',
    'good_price',    'bad_price',
    'good_check-in', 'bad_check-in',
    'good_check-out','bad_check-out',
]


# ── Graph stats helper ────────────────────────────────────────────────────────
def compute_stats(d: pd.DataFrame) -> dict:
    """
    Computes triples, entities, relations, avg_score, rel_counts, top_entities
    for a graph dataframe (visualization_data.csv).
    Used for both raw and clean versions.
    """
    entities = set(d["head"].tolist() + d["tail"].tolist())

    degree = {}
    for _, row in d.iterrows():
        degree[row["head"]] = degree.get(row["head"], 0) + 1
        degree[row["tail"]] = degree.get(row["tail"], 0) + 1
    top_entities = sorted(degree.items(), key=lambda x: -x[1])[:10]

    avg_score = None
    if "score" in d.columns and d["score"].notna().any():
        avg_score = round(float(d["score"].mean()), 4)

    return {
        "triples":      len(d),
        "entities":     len(entities),
        "relations":    int(d["relation"].nunique()),
        "avg_score":    avg_score,
        "rel_counts":   d["relation"].value_counts().to_dict(),
        "top_entities": [{"entity": e, "degree": deg} for e, deg in top_entities],
    }


# ── Tree stats helper ─────────────────────────────────────────────────────────
def compute_tree_stats(d: pd.DataFrame) -> dict:
    """
    Computes claims, bodies, attackers, total_rows, and relation breakdown
    for a tree dataframe (aba_tree_structure.csv).
    Used for both raw and clean versions.
    """
    contrary_count     = int(d[d["attack_relation"] == "CONTRARY_TO"].shape[0])
    not_contrary_count = int(d[d["attack_relation"] == "NOT_CONTRARY"].shape[0])
    support_count      = int(d[d["body_relation"] == "SUPPORT"].shape[0])

    # attacker count excludes null attackers (body-only rows)
    attacker_series = d["attacker"].dropna()
    attacker_count  = int(attacker_series.nunique()) if len(attacker_series) > 0 else 0

    return {
        "claims":              int(d["claim"].nunique()),
        "bodies":              int(d["body"].nunique()),
        "attackers":           attacker_count,
        "total_rows":          len(d),
        "contrary_count":      contrary_count,
        "not_contrary_count":  not_contrary_count,
        "support_count":       support_count,
    }


# ── Score normalizer ──────────────────────────────────────────────────────────
def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stores original score as score_raw, negates then normalizes score to [0,1]
    per relation group. RotatE scores: lower distance = more plausible,
    so we negate first before normalizing.
    """
    df = df.copy()
    df["score_raw"] = df["score"]
    df["score"]     = -df["score"]

    def normalize_group(g):
        s_min, s_max = g.min(), g.max()
        if s_max == s_min:
            return g * 0 + 1.0
        return ((g - s_min) / (s_max - s_min)).round(6)

    df["score"] = df.groupby("relation")["score"].transform(normalize_group)
    return df


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

        # ── Raw copy — normalize scores but keep all rows ─────────────────
        df_raw = normalize_scores(df)

        # ── Clean copy — drop nulls + duplicates then normalize ───────────
        df_clean = df.dropna(subset=["head", "tail", "score",
                                     "head_x", "head_y", "tail_x", "tail_y"])
        df_clean = df_clean.drop_duplicates(
            subset=["head", "relation", "tail"]
        ).reset_index(drop=True)
        df_clean = normalize_scores(df_clean)

        store["df_raw"] = df_raw
        store["df"]     = df_clean

        raw_stats        = compute_stats(df_raw)
        clean_stats      = compute_stats(df_clean)
        duplicates       = len(df_raw) - len(df_clean)

        return {
            "ok":         True,
            "duplicates": duplicates,
            "raw":        raw_stats,
            "clean":      clean_stats,
            # backward compat
            "rows":       clean_stats["triples"],
            "entities":   clean_stats["entities"],
            "relations":  clean_stats["relations"],
            "rel_counts": clean_stats["rel_counts"],
            "columns":    df_clean.columns.tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Upload — aba_tree_structure.csv  (Tree View)
# Mirrors graph upload: one file → backend splits into raw + clean
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

        # Replace NaN with None for JSON serialization
        df = df.where(pd.notnull(df), None)

        # ── Raw copy — keep all rows ──────────────────────────────────────
        df_raw = df.copy()

        # ── Clean copy — deduplicate ──────────────────────────────────────
        # Rows with attackers: keep highest attack_score per unique combo
        df_with_atk = df[df["attacker"].notna()].copy()
        df_with_atk["attack_score_num"] = pd.to_numeric(
            df_with_atk["attack_score"], errors="coerce"
        )
        df_with_atk = (
            df_with_atk
            .sort_values("attack_score_num", ascending=False)
            .drop_duplicates(subset=["claim", "body", "attacker", "attack_relation"])
            .drop(columns=["attack_score_num"])
            .reset_index(drop=True)
        )

        # Rows without attackers (body-only): keep one per (claim, body)
        df_body_only = df[df["attacker"].isna()].copy()
        df_body_only = (
            df_body_only
            .drop_duplicates(subset=["claim", "body"])
            .reset_index(drop=True)
        )

        df_clean = pd.concat(
            [df_with_atk, df_body_only], ignore_index=True
        )

        store["df_tree_raw"] = df_raw
        store["df_tree"]     = df_clean

        raw_stats   = compute_tree_stats(df_raw)
        clean_stats = compute_tree_stats(df_clean)
        duplicates  = raw_stats["total_rows"] - clean_stats["total_rows"]

        return {
            "ok":         True,
            "duplicates": duplicates,
            "raw":        raw_stats,
            "clean":      clean_stats,
            # backward compat
            "rows":       clean_stats["total_rows"],
            "claims":     clean_stats["claims"],
            "bodies":     clean_stats["bodies"],
            "attackers":  clean_stats["attackers"],
            "claim_list": sorted(df_clean["claim"].dropna().unique().tolist()),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Stats — graph raw vs clean
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/stats")
def get_stats():
    df     = store["df"]
    df_raw = store["df_raw"]

    if df is None:
        raise HTTPException(status_code=404, detail="No graph data loaded")

    clean_stats = compute_stats(df)
    raw_stats   = compute_stats(df_raw) if df_raw is not None else None
    duplicates  = (len(df_raw) - len(df)) if df_raw is not None else 0

    return {
        "clean":      clean_stats,
        "raw":        raw_stats,
        "duplicates": duplicates,
        # backward compat
        "total_triples":   clean_stats["triples"],
        "total_entities":  clean_stats["entities"],
        "total_relations": clean_stats["relations"],
        "score_raw_mean":  round(float(df["score_raw"].mean()), 4),
        "rel_counts":      clean_stats["rel_counts"],
        "top_entities":    clean_stats["top_entities"],
        "score_min":       round(float(df["score"].min()), 4),
        "score_max":       round(float(df["score"].max()), 4),
        "score_mean":      round(float(df["score"].mean()), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tree stats — raw vs clean
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/stats/tree")
def get_tree_stats():
    df_tree     = store["df_tree"]
    df_tree_raw = store["df_tree_raw"]

    if df_tree is None:
        raise HTTPException(status_code=404, detail="No tree data loaded")

    clean_stats = compute_tree_stats(df_tree)
    raw_stats   = compute_tree_stats(df_tree_raw) if df_tree_raw is not None else None
    duplicates  = (raw_stats["total_rows"] - clean_stats["total_rows"]) if raw_stats else 0

    return {
        "clean":      clean_stats,
        "raw":        raw_stats,
        "duplicates": duplicates,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Graph — for Graph View
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/graph")
def get_graph(
    relation:  str   = "all",
    min_score: float = 0.0,
    search:    str   = "",
    limit:     int   = -1,
    show_raw:  bool  = False,
):
    df = store["df_raw"] if show_raw else store["df"]
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
# Tree — always serves clean data for drawing
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/tree")
def get_tree(
    claim:           str   = "all",
    min_body_score:  float = 0.0,
    attack_relation: str   = "all",
    domain:          str   = "all",
):
    df = store["df_tree"]   # always clean — raw is stats only
    if df is None:
        raise HTTPException(
            status_code=404,
            detail="No tree data loaded. Please upload aba_tree_structure.csv"
        )

    filtered = df.copy()

    if claim != "all":
        filtered = filtered[filtered["claim"] == claim]
    if min_body_score > 0:
        filtered = filtered[filtered["body_score"].notna()]
        filtered = filtered[filtered["body_score"] >= min_body_score]
    if attack_relation != "all":
        filtered = filtered[filtered["attack_relation"] == attack_relation]
    if domain != "all" and "domain" in filtered.columns:
        filtered = filtered[filtered["domain"] == domain]

    trees = {}
    for claim_name, group in filtered.groupby("claim"):
        bodies = {}
        for _, row in group.iterrows():
            body = row["body"]
            if body not in bodies:
                bodies[body] = {
                    "name":       body,
                    "body_score": row["body_score"],
                    "relation":   row["body_relation"],
                    "attackers":  [],
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
                "total_bodies":       len(bodies),
                "total_attackers":    sum(len(b["attackers"]) for b in bodies.values()),
                "contrary_count":     int(group[group["attack_relation"] == "CONTRARY_TO"].shape[0]),
                "not_contrary_count": int(group[group["attack_relation"] == "NOT_CONTRARY"].shape[0]),
                "support_count":      int(group[group["attack_relation"] == "SUPPORT"].shape[0]),
            }
        }

    domains = []
    if "domain" in df.columns:
        domains = sorted(df["domain"].dropna().unique().tolist())

    return {
        "trees":             trees,
        "available_claims":  sorted(df["claim"].dropna().unique().tolist()),
        "available_domains": domains,
        "total_rows":        len(filtered),
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