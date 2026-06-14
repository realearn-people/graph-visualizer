# graph-viz

Web-based visualizer for KGE (Knowledge Graph Embedding) results from the ABA mining project. Upload output CSVs from any KGE training script and explore the entity graph and ABA argument tree in the browser.

## Architecture

- **Backend**: FastAPI (Python) — `backend/main.py`
  - Runs on port 8000
  - In-memory data store (no database); resets on restart
  - Data is loaded by the user uploading CSVs in the browser
- **Frontend**: Static HTML/CSS/JS — `frontend/`
  - Served by FastAPI as a static file mount at `/`

## Views

| View | Input file | Description |
|---|---|---|
| **Graph View** | `visualization_data.csv` | 2D entity scatter with colored relation edges; filter by relation, score threshold, or search |
| **Tree View** | `aba_tree_structure.csv` | ABA argument tree (claim → body → attackers); filter by claim, domain, attack relation |

Both files are produced by the KGE training scripts in the aba-mining project (`sup_rotate.py`, `sup_complex.py`, etc.).

## Setup

### Dependencies

```bash
pip install fastapi uvicorn pandas python-multipart
```

### Run locally

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in the browser, then upload the CSV files.

### Systemd service (on the server)

The service file is at `/etc/systemd/system/graph-viz.service`:

```bash
sudo systemctl start graph-viz
sudo systemctl status graph-viz
sudo systemctl restart graph-viz   # after code changes
```

The service's `WorkingDirectory` must point to `backend/` so that the relative `../frontend` path resolves correctly.

## API Reference

### Upload endpoints

| Method | Path | Body | Description |
|---|---|---|---|
| POST | `/api/upload` | `visualization_data.csv` | Load graph data; returns raw + clean stats |
| POST | `/api/upload/tree` | `aba_tree_structure.csv` | Load tree data; returns raw + clean stats |

On upload, the backend stores both a **raw** copy (all rows, normalized scores) and a **clean** copy (null rows and duplicate triples removed).

### Query endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/stats` | Graph statistics (triples, entities, relations, score range, top entities) |
| GET | `/api/stats/tree` | Tree statistics (claims, bodies, attackers, relation breakdown) |
| GET | `/api/graph` | Filtered graph: returns nodes + edges for rendering |
| GET | `/api/tree` | Filtered ABA tree: returns nested claim → body → attacker structure |
| GET | `/api/relations` | List of relation types in the loaded graph |

### `/api/graph` query parameters

| Param | Default | Description |
|---|---|---|
| `relation` | `"all"` | Filter to one relation type |
| `min_score` | `0.0` | Minimum normalized score [0, 1] |
| `search` | `""` | Substring match on head or tail entity name |
| `limit` | `-1` | Max triples to return; distributed evenly across relations when `relation=all` |
| `show_raw` | `false` | Use pre-deduplication data |

### `/api/tree` query parameters

| Param | Default | Description |
|---|---|---|
| `claim` | `"all"` | Filter to one claim label |
| `min_body_score` | `0.0` | Minimum body score threshold |
| `attack_relation` | `"all"` | Filter attackers by relation type |
| `domain` | `"all"` | Filter by hotel domain (staff, price, check-in, check-out) |

## Score Normalization

RotatE produces distance-based scores where **lower score = more plausible**. To make the UI consistent, the backend:

1. Negates the score: `score = -score_raw`
2. Min-max normalizes per relation group to [0, 1]

So `score = 1.0` means the most plausible triple in that relation group, and `score = 0.0` means the least. The original RotatE distance is preserved as `score_raw`.

## Tests

```bash
cd backend
python -m pytest testcases.py -v
```

The test suite covers all helper functions (compute_stats, compute_tree_stats, normalize_scores), all API endpoints, and store isolation between tests (85 test methods).
