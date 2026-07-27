"""
ICT Spot Trading Bot for TrueMarkets (TrueX)

Executes ICT-based spot trades on TrueX via the truemarkets Python SDK.
Connects to TradingView webhook alerts from the ICT PineScript strategy
and places limit/IOC orders on BTC/USDC, ETH/USDC, SOL/USDC.

Setup:
  1. pip install truemarkets flask
  2. Create .env with TM_KEY_FILE=/path/to/api-key.json and TM_ENV=prod
  3. Configure TradingView alerts to POST to this bot's webhook endpoint
  4. python truemarkets_ict_bot.py

TradingView Alert Message Format (JSON):
  {
    "action": "buy" | "sell",
    "asset": "BTC" | "ETH" | "SOL",
    "price": "current_price",
    "score": 5,
    "sl": "stop_loss_price",
    "tp": "take_profit_price",
    "structure": "BOS" | "CHoCH",
    "zone": "discount" | "premium",
    "kill_zone": "london" | "new_york" | "asia" | "none"
  }
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock

from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ict_bot.log"),
    ],
)
log = logging.getLogger("ict_bot")

app = Flask(__name__)
trade_lock = Lock()


@dataclass
class TradeConfig:
    max_position_pct: float = 5.0
    min_score: int = 5
    quote_asset: str = "USDC"
    allowed_assets: tuple = ("BTC", "ETH", "SOL")
    max_open_positions: int = 3
    cooldown_seconds: int = 300


CONFIG = TradeConfig()
last_trade_time: dict[str, float] = {}
open_positions: dict[str, str] = {}


def get_client():
    from truemarkets import Client
    return Client()


def get_available_balance(client, asset: str) -> Decimal:
    balances = client.gateway.get_balances()
    for b in balances:
        if b.asset == asset:
            return Decimal(str(b.available))
    return Decimal("0")


def compute_position_size(client, base_asset: str, price: Decimal) -> Decimal:
    balance = get_available_balance(client, CONFIG.quote_asset)
    allocation = balance * Decimal(str(CONFIG.max_position_pct / 100))
    qty = allocation / price
    return qty.quantize(Decimal("0.00001"))


def place_limit_order(client, base_asset: str, side: str, qty: str, price: str) -> dict:
    from truemarkets._generated.gateway.models import (
        CreateOrderRequest,
        ExecuteOrderRequest,
        OrderSide,
        OrderType,
        SigningMethod,
    )

    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    order = client.gateway.create_order(
        body=CreateOrderRequest(
            base_asset=base_asset,
            quote_asset=CONFIG.quote_asset,
            side=order_side,
            type=OrderType.LIMIT,
            qty=qty,
            price=price,
        )
    )

    if order.payloads:
        stamps = [client.sign_turnkey(p.payload) for p in order.payloads]
        client.gateway.execute_order(
            id=order.order_id,
            body=ExecuteOrderRequest(
                signatures=stamps,
                auth_type=SigningMethod.API_KEY,
            ),
        )

    log.info(
        "Order placed: %s %s %s @ %s %s (order_id=%s)",
        side.upper(), qty, base_asset, price, CONFIG.quote_asset, order.order_id,
    )
    return {"order_id": order.order_id, "status": "submitted"}


def validate_signal(data: dict) -> tuple[bool, str]:
    action = data.get("action", "").lower()
    if action not in ("buy", "sell"):
        return False, f"Invalid action: {action}"

    asset = data.get("asset", "").upper()
    if asset not in CONFIG.allowed_assets:
        return False, f"Asset not allowed: {asset}"

    score = int(data.get("score", 0))
    if score < CONFIG.min_score:
        return False, f"Score too low: {score} < {CONFIG.min_score}"

    now = time.time()
    if asset in last_trade_time:
        elapsed = now - last_trade_time[asset]
        if elapsed < CONFIG.cooldown_seconds:
            return False, f"Cooldown active for {asset}: {int(CONFIG.cooldown_seconds - elapsed)}s remaining"

    if len(open_positions) >= CONFIG.max_open_positions and asset not in open_positions:
        return False, f"Max open positions reached ({CONFIG.max_open_positions})"

    return True, "OK"


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    log.info("Webhook received: %s", json.dumps(data))

    valid, reason = validate_signal(data)
    if not valid:
        log.warning("Signal rejected: %s", reason)
        return jsonify({"status": "rejected", "reason": reason}), 200

    action = data["action"].lower()
    asset = data["asset"].upper()
    price = data.get("price", "0")
    sl = data.get("sl")
    tp = data.get("tp")
    structure = data.get("structure", "unknown")
    zone = data.get("zone", "unknown")
    kill_zone = data.get("kill_zone", "unknown")

    log.info(
        "ICT Signal: %s %s | structure=%s zone=%s kz=%s score=%s",
        action, asset, structure, zone, kill_zone, data.get("score"),
    )

    with trade_lock:
        try:
            with get_client() as client:
                price_dec = Decimal(str(price))
                qty = compute_position_size(client, asset, price_dec)

                if qty <= 0:
                    return jsonify({"status": "skipped", "reason": "Insufficient balance"}), 200

                result = place_limit_order(client, asset, action, str(qty), price)

                last_trade_time[asset] = time.time()
                open_positions[asset] = result["order_id"]

                return jsonify({
                    "status": "executed",
                    "order_id": result["order_id"],
                    "asset": asset,
                    "side": action,
                    "qty": str(qty),
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                }), 200

        except Exception as e:
            log.error("Trade execution failed: %s", e, exc_info=True)
            return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/positions", methods=["GET"])
def positions():
    return jsonify({"open_positions": open_positions, "count": len(open_positions)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "open_positions": len(open_positions)})


if __name__ == "__main__":
    port = int(os.environ.get("BOT_PORT", 5000))
    log.info("ICT Bot starting on port %d", port)
    log.info("Config: %s", CONFIG)
    app.run(host="0.0.0.0", port=port)
