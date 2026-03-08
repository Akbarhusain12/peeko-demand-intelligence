"""
app.py
------
Flask API for the Peeko Baby Product Demand Prediction system (XGBoost).

Install:
    pip install flask flask-cors pandas numpy xgboost scikit-learn

Endpoints:
    GET  /                              - serves the dashboard (index.html)
    GET  /api/health                    - liveness check + model metrics
    GET  /api/predictions               - full demand forecast
    GET  /api/predictions?limit=N       - top N forecast rows
    GET  /api/restock                   - restock action board
    GET  /api/restock?limit=N           - top N critical SKUs
    POST /api/restock/custom-inventory  - restock with real inventory data
    GET  /api/model-metrics             - detailed XGBoost evaluation metrics

Run:
    python app.py
    open http://127.0.0.1:5000
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from model_data import run_full_pipeline, generate_restock_action_board
import pandas as pd

app = Flask(__name__)
CORS(app)

# ── Update this path to your CSV folder before running ───────────────────────
DATA_DIR = "D:/MCA/Project/Peeko/Data"

# ---------------------------------------------------------------------------
# Session cache — pipeline runs once on first request
# ---------------------------------------------------------------------------

import threading

_cache = {}
_cache_lock = threading.Lock() # The bouncer for our cache

def get_pipeline_results():
    """Run the full pipeline once and cache results. Uses a Lock to prevent race conditions."""
    with _cache_lock: # Forces concurrent API requests to wait in line
        if "results" not in _cache:
            print("\n>>> CACHE MISS: Booting the ML Pipeline (This will only happen once) <<<\n")
            _cache["results"] = run_full_pipeline(DATA_DIR)
        else:
            print("\n>>> CACHE HIT: Serving from memory <<<\n")
            
    return _cache["results"]

def serialize_df(df: pd.DataFrame) -> list:
    # ... [keep your existing serialize_df code exactly as it is] ...
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Liveness check. Returns XGBoost model metrics when pipeline is ready."""
    try:
        results = get_pipeline_results()
        return jsonify({
            "status":  "ok",
            "model":   "XGBoost",
            "metrics": results.get("eval_metrics", {}),
        })
    except Exception:
        return jsonify({"status": "ok", "model": "XGBoost"})


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """
    Return the full XGBoost demand forecast.
    Optional query param: ?limit=N
    """
    results = get_pipeline_results()
    df = results["final_predictions"]

    limit = request.args.get("limit", default=None, type=int)
    if limit:
        df = df.head(limit)

    return jsonify({
        "stats":       results["stats"],
        "predictions": serialize_df(df),
    })


@app.route("/api/restock", methods=["GET"])
def get_restock():
    """
    Return the XGBoost-powered restock action board.
    Optional query param: ?limit=N
    """
    results = get_pipeline_results()
    df = results["critical_restock"]

    limit = request.args.get("limit", default=None, type=int)
    if limit:
        df = df.head(limit)

    return jsonify({
        "stats":        results["stats"],
        "restock_list": serialize_df(df),
    })


@app.route("/api/restock/custom-inventory", methods=["POST"])
def restock_with_custom_inventory():
    """
    Accept real inventory data and return a restock plan.

    Request body (JSON):
    {
        "inventory": [
            {"product_id": 123, "current_stock": 5},
            ...
        ]
    }
    """
    body = request.get_json()
    if not body or "inventory" not in body:
        return jsonify({"error": "Request body must include an 'inventory' list"}), 400

    inventory_df = pd.DataFrame(body["inventory"])
    if not {"product_id", "current_stock"}.issubset(inventory_df.columns):
        return jsonify({"error": "Each item must have 'product_id' and 'current_stock'"}), 400

    results  = get_pipeline_results()
    critical = generate_restock_action_board(
        final_predictions=results["final_predictions"],
        baby_product_ids=None,
        inventory_df=inventory_df,
    )

    return jsonify({
        "total_critical_skus": int(len(critical)),
        "restock_list":        serialize_df(critical),
    })


@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    """Return detailed XGBoost training and evaluation metrics."""
    results = get_pipeline_results()
    return jsonify({
        "model":   "XGBoost",
        "metrics": results.get("eval_metrics", {}),
    })


@app.route("/", methods=["GET"])
def index():
    """Serve the frontend dashboard."""
    return send_from_directory(".", "index.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)