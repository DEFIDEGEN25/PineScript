"""
ICT Spot Trading Bot for TrueMarkets (TrueX)

Standalone algo that runs ICT analysis and executes spot trades
on BTC/USD, ETH/USD, SOL/USD via the TrueMarkets REST API.

Also exposes a webhook endpoint for TradingView alert-driven trading.

Auth: ES256 JWT signed with your API key.

Setup:
  1. pip install requests flask cryptography
  2. Create .env:
       TM_API_KEY_ID=<your-key-uuid>
       TM_PRIVATE_KEY_FILE=/path/to/private-key.pem
       TM_ENV=prod
  3. python truemarkets_ict_bot.py

Modes:
  --mode auto     Run the ICT algo loop (default)
  --mode webhook  Run as a webhook server for TradingView alerts
  --mode both     Run both the algo loop and webhook server
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ict_bot.log"),
    ],
)
log = logging.getLogger("ict_bot")

BASE_URL = "https://api.truemarkets.co/v1"


# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    # API credentials
    api_key_id: str = ""
    private_key_file: str = ""

    # Trading
    quote_asset: str = "USD"
    assets: list = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    poll_interval: int = 60
    lookback_candles: int = 100

    # ICT parameters
    swing_length: int = 5
    ob_lookback: int = 10
    ob_max_age: int = 100
    fvg_min_pct: float = 0.1
    fvg_max_age: int = 50
    ote_fib_high: float = 0.79
    ote_fib_low: float = 0.62
    min_entry_score: int = 5
    displacement_atr_mult: float = 1.5

    # Kill zones (UTC hours)
    use_kill_zones: bool = True
    london_open: int = 2
    london_close: int = 5
    ny_open: int = 7
    ny_close: int = 10
    asia_open: int = 20
    asia_close: int = 0

    # Risk management
    max_position_pct: float = 5.0
    max_open_positions: int = 3
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    cooldown_seconds: int = 300
    max_daily_trades: int = 10
    max_daily_loss_pct: float = 3.0

    # Webhook
    webhook_port: int = 5000
    webhook_secret: str = ""


def load_config() -> BotConfig:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    cfg = BotConfig()
    cfg.api_key_id = os.environ.get("TM_API_KEY_ID", "")
    cfg.private_key_file = os.environ.get("TM_PRIVATE_KEY_FILE", "")
    cfg.quote_asset = os.environ.get("BOT_QUOTE_ASSET", cfg.quote_asset)
    cfg.poll_interval = int(os.environ.get("BOT_POLL_INTERVAL", cfg.poll_interval))
    cfg.max_position_pct = float(os.environ.get("BOT_MAX_POSITION_PCT", cfg.max_position_pct))
    cfg.max_daily_loss_pct = float(os.environ.get("BOT_MAX_DAILY_LOSS_PCT", cfg.max_daily_loss_pct))
    cfg.webhook_port = int(os.environ.get("BOT_PORT", cfg.webhook_port))
    cfg.webhook_secret = os.environ.get("BOT_WEBHOOK_SECRET", "")
    assets_env = os.environ.get("BOT_ASSETS")
    if assets_env:
        cfg.assets = [a.strip().upper() for a in assets_env.split(",")]
    return cfg


# ─── TRUEMARKETS API CLIENT ───────────────────────────────────────────────────

class TrueMarketsClient:
    """Thin REST client for the TrueMarkets API with ES256 JWT auth."""

    def __init__(self, api_key_id: str, private_key_file: str):
        self.api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_file)
        self._token: str | None = None
        self._token_exp: float = 0

    @staticmethod
    def _load_private_key(path: str):
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        key_data = Path(path).read_bytes()
        return load_pem_private_key(key_data, password=None)

    @staticmethod
    def _base64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _sign_es256(self, payload: bytes) -> str:
        from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
        from cryptography.hazmat.primitives.hashes import SHA256
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        sig_der = self._private_key.sign(payload, ECDSA(SHA256()))
        r, s = decode_dss_signature(sig_der)
        raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return self._base64url(raw)

    def _mint_token(self) -> str:
        timestamp = int(time.time())
        message = f"{self.api_key_id}.{timestamp}".encode()
        signature = self._sign_es256(message)

        resp = requests.post(
            f"{BASE_URL}/auth/api-key/token",
            json={
                "key_id": self.api_key_id,
                "timestamp": timestamp,
                "signature": signature,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data.get("access_token") or data.get("token")
        self._token_exp = time.time() + 3500
        log.info("JWT minted successfully")
        return self._token

    def _get_token(self) -> str:
        if not self._token or time.time() >= self._token_exp:
            self._mint_token()
        return self._token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

    def compute_quote(self, base: str, quote: str, side: str, quantity: str) -> dict:
        resp = requests.post(
            f"{BASE_URL}/gateway/quotes",
            headers=self._headers(),
            json={
                "base": base,
                "quote": quote,
                "side": side,
                "quantity": quantity,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def create_order(self, base: str, quote: str, side: str, quantity: str, price: str, order_type: str = "limit") -> dict:
        resp = requests.post(
            f"{BASE_URL}/gateway/orders",
            headers=self._headers(),
            json={
                "base": base,
                "quote": quote,
                "side": side,
                "quantity": quantity,
                "price": price,
                "type": order_type,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def execute_order(self, order_id: str, signatures: list[dict] | None = None) -> dict:
        body = {}
        if signatures:
            body["signatures"] = signatures
        resp = requests.post(
            f"{BASE_URL}/gateway/orders/{order_id}/execute",
            headers=self._headers(),
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id: str) -> dict:
        resp = requests.delete(
            f"{BASE_URL}/gateway/orders/{order_id}",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_order_status(self, order_id: str) -> dict:
        resp = requests.get(
            f"{BASE_URL}/gateway/orders/{order_id}",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def list_orders(self) -> list[dict]:
        resp = requests.get(
            f"{BASE_URL}/gateway/orders",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_balances(self) -> dict:
        resp = requests.get(
            f"{BASE_URL}/gateway/balances",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def list_assets(self) -> list[dict]:
        resp = requests.get(
            f"{BASE_URL}/gateway/assets",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

class Trend(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SwingPoint:
    price: float
    bar_index: int
    is_high: bool


@dataclass
class OrderBlock:
    top: float
    bottom: float
    bar_index: int
    is_bullish: bool
    active: bool = True


@dataclass
class FairValueGap:
    top: float
    bottom: float
    bar_index: int
    is_bullish: bool
    active: bool = True


@dataclass
class ICTSignal:
    action: str
    asset: str
    price: float
    score: int
    stop_loss: float
    take_profit: float
    structure: str
    zone: str
    kill_zone: str
    order_block: bool
    fvg: bool
    ote: bool
    liquidity_sweep: bool


@dataclass
class Position:
    asset: str
    side: str
    entry_price: float
    qty: float
    order_id: str
    stop_loss: float
    take_profit: float
    entry_time: float


# ─── ICT ANALYSIS ENGINE ──────────────────────────────────────────────────────

class ICTAnalyzer:
    def __init__(self, config: BotConfig):
        self.cfg = config

    def find_swing_points(self, candles: list[Candle]) -> list[SwingPoint]:
        swings = []
        n = self.cfg.swing_length
        for i in range(n, len(candles) - n):
            is_high = all(candles[i].high >= candles[i + j].high for j in range(-n, n + 1) if j != 0)
            is_low = all(candles[i].low <= candles[i + j].low for j in range(-n, n + 1) if j != 0)
            if is_high:
                swings.append(SwingPoint(candles[i].high, i, True))
            if is_low:
                swings.append(SwingPoint(candles[i].low, i, False))
        return swings

    def determine_trend(self, swings: list[SwingPoint]) -> Trend:
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]
        if len(highs) < 2 or len(lows) < 2:
            return Trend.NEUTRAL
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price
        if hh and hl:
            return Trend.BULLISH
        if lh and ll:
            return Trend.BEARISH
        return Trend.NEUTRAL

    def detect_structure_break(self, candles: list[Candle], swings: list[SwingPoint], trend: Trend) -> tuple[str, bool]:
        if not swings or len(candles) < 2:
            return "", False
        current = candles[-1]
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]

        if highs and current.close > highs[-1].price:
            if trend == Trend.BULLISH:
                return "BOS", True
            elif trend == Trend.BEARISH:
                return "CHoCH", True

        if lows and current.close < lows[-1].price:
            if trend == Trend.BEARISH:
                return "BOS", False
            elif trend == Trend.BULLISH:
                return "CHoCH", False

        return "", False

    def compute_atr(self, candles: list[Candle], period: int = 14) -> float:
        if len(candles) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            tr = max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i - 1].close),
                abs(candles[i].low - candles[i - 1].close),
            )
            trs.append(tr)
        return sum(trs[-period:]) / period

    def find_order_blocks(self, candles: list[Candle], atr: float) -> list[OrderBlock]:
        obs = []
        for i in range(1, len(candles)):
            body = abs(candles[i].close - candles[i].open)
            if body < atr * self.cfg.displacement_atr_mult:
                continue
            is_bullish_disp = candles[i].close > candles[i].open
            for j in range(1, min(self.cfg.ob_lookback + 1, i + 1)):
                idx = i - j
                if is_bullish_disp and candles[idx].close < candles[idx].open:
                    obs.append(OrderBlock(
                        top=candles[idx].open, bottom=candles[idx].close,
                        bar_index=idx, is_bullish=True,
                    ))
                    break
                elif not is_bullish_disp and candles[idx].close > candles[idx].open:
                    obs.append(OrderBlock(
                        top=candles[idx].close, bottom=candles[idx].open,
                        bar_index=idx, is_bullish=False,
                    ))
                    break

        current_bar = len(candles) - 1
        for ob in obs:
            if current_bar - ob.bar_index > self.cfg.ob_max_age:
                ob.active = False
                continue
            for c in candles[ob.bar_index + 1:]:
                if ob.is_bullish and c.close < ob.bottom:
                    ob.active = False
                    break
                if not ob.is_bullish and c.close > ob.top:
                    ob.active = False
                    break
        return [ob for ob in obs if ob.active]

    def find_fvgs(self, candles: list[Candle]) -> list[FairValueGap]:
        fvgs = []
        for i in range(2, len(candles)):
            gap_up = candles[i].low - candles[i - 2].high
            gap_down = candles[i - 2].low - candles[i].high
            price = candles[i].close
            if gap_up > 0 and (gap_up / price * 100) > self.cfg.fvg_min_pct:
                fvgs.append(FairValueGap(
                    top=candles[i].low, bottom=candles[i - 2].high,
                    bar_index=i - 1, is_bullish=True,
                ))
            elif gap_down > 0 and (gap_down / price * 100) > self.cfg.fvg_min_pct:
                fvgs.append(FairValueGap(
                    top=candles[i - 2].low, bottom=candles[i].high,
                    bar_index=i - 1, is_bullish=False,
                ))

        current_bar = len(candles) - 1
        for fvg in fvgs:
            if current_bar - fvg.bar_index > self.cfg.fvg_max_age:
                fvg.active = False
                continue
            for c in candles[fvg.bar_index + 1:]:
                if fvg.is_bullish and c.low <= fvg.bottom:
                    fvg.active = False
                    break
                if not fvg.is_bullish and c.high >= fvg.top:
                    fvg.active = False
                    break
        return [f for f in fvgs if f.active]

    def get_premium_discount(self, candles: list[Candle]) -> str:
        range_high = max(c.high for c in candles)
        range_low = min(c.low for c in candles)
        equilibrium = (range_high + range_low) / 2
        return "discount" if candles[-1].close < equilibrium else "premium"

    def check_ote_zone(self, candles: list[Candle], swings: list[SwingPoint]) -> bool:
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]
        if not highs or not lows:
            return False
        last_high = highs[-1].price
        last_low = lows[-1].price
        impulse = last_high - last_low
        if impulse <= 0:
            return False
        ote_top = last_high - impulse * self.cfg.ote_fib_low
        ote_bot = last_high - impulse * self.cfg.ote_fib_high
        return ote_bot <= candles[-1].close <= ote_top

    def get_kill_zone(self) -> str:
        import datetime
        hr = datetime.datetime.now(datetime.timezone.utc).hour
        if self.cfg.london_open <= hr < self.cfg.london_close:
            return "london"
        if self.cfg.ny_open <= hr < self.cfg.ny_close:
            return "new_york"
        if self.cfg.asia_open > self.cfg.asia_close:
            if hr >= self.cfg.asia_open or hr < self.cfg.asia_close:
                return "asia"
        elif self.cfg.asia_open <= hr < self.cfg.asia_close:
            return "asia"
        return "none"

    def check_liquidity_sweep(self, candles: list[Candle]) -> tuple[bool, bool]:
        if len(candles) < 24:
            return False, False
        prev_session = candles[-24:-1]
        prev_high = max(c.high for c in prev_session)
        prev_low = min(c.low for c in prev_session)
        current = candles[-1]
        sweep_above = current.high > prev_high and current.close < prev_high
        sweep_below = current.low < prev_low and current.close > prev_low
        return sweep_above, sweep_below

    def analyze(self, asset: str, candles: list[Candle]) -> ICTSignal | None:
        if len(candles) < 30:
            log.warning("Not enough candles for %s: %d", asset, len(candles))
            return None

        swings = self.find_swing_points(candles)
        trend = self.determine_trend(swings)
        structure, is_bullish_break = self.detect_structure_break(candles, swings, trend)
        atr = self.compute_atr(candles)
        if atr == 0:
            return None
        obs = self.find_order_blocks(candles, atr)
        fvgs = self.find_fvgs(candles)
        pd_zone = self.get_premium_discount(candles)
        in_ote = self.check_ote_zone(candles, swings)
        kill_zone = self.get_kill_zone()
        sweep_above, sweep_below = self.check_liquidity_sweep(candles)

        price = candles[-1].close

        # ── Score long setup ──
        long_score = 0
        if trend == Trend.BULLISH:
            long_score += 2
        if pd_zone == "discount":
            long_score += 1
        bull_ob_touch = any(
            ob.is_bullish and candles[-1].low <= ob.top and candles[-1].close >= ob.bottom
            for ob in obs
        )
        bull_fvg_touch = any(
            f.is_bullish and candles[-1].low <= f.top and candles[-1].close >= f.bottom
            for f in fvgs
        )
        if bull_ob_touch or bull_fvg_touch:
            long_score += 2
        if in_ote:
            long_score += 1
        if kill_zone != "none":
            long_score += 1
        if sweep_below:
            long_score += 1

        # ── Score short setup ──
        short_score = 0
        if trend == Trend.BEARISH:
            short_score += 2
        if pd_zone == "premium":
            short_score += 1
        bear_ob_touch = any(
            not ob.is_bullish and candles[-1].high >= ob.bottom and candles[-1].close <= ob.top
            for ob in obs
        )
        bear_fvg_touch = any(
            not f.is_bullish and candles[-1].high >= f.bottom and candles[-1].close <= f.top
            for f in fvgs
        )
        if bear_ob_touch or bear_fvg_touch:
            short_score += 2
        if in_ote:
            short_score += 1
        if kill_zone != "none":
            short_score += 1
        if sweep_above:
            short_score += 1

        min_score = self.cfg.min_entry_score

        if long_score >= min_score and long_score > short_score:
            return ICTSignal(
                action="buy", asset=asset, price=price, score=long_score,
                stop_loss=price - atr * self.cfg.sl_atr_mult,
                take_profit=price + atr * self.cfg.tp_atr_mult,
                structure=structure or "trend", zone=pd_zone, kill_zone=kill_zone,
                order_block=bull_ob_touch, fvg=bull_fvg_touch,
                ote=in_ote, liquidity_sweep=sweep_below,
            )

        if short_score >= min_score and short_score > long_score:
            return ICTSignal(
                action="sell", asset=asset, price=price, score=short_score,
                stop_loss=price + atr * self.cfg.sl_atr_mult,
                take_profit=price - atr * self.cfg.tp_atr_mult,
                structure=structure or "trend", zone=pd_zone, kill_zone=kill_zone,
                order_block=bear_ob_touch, fvg=bear_fvg_touch,
                ote=in_ote, liquidity_sweep=sweep_above,
            )

        return None


# ─── TRADE EXECUTION ENGINE ───────────────────────────────────────────────────

class TradeManager:
    def __init__(self, config: BotConfig, client: TrueMarketsClient):
        self.cfg = config
        self.client = client
        self.positions: dict[str, Position] = {}
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.last_trade_time: dict[str, float] = {}
        self.trade_history: list[dict] = []
        self.lock = threading.Lock()
        self._day_reset: float = 0.0

    def _reset_daily_if_needed(self):
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0).timestamp()
        if day_start > self._day_reset:
            self._day_reset = day_start
            self.daily_trades = 0
            self.daily_pnl = 0.0
            log.info("Daily counters reset")

    def can_trade(self, asset: str) -> tuple[bool, str]:
        self._reset_daily_if_needed()
        if self.daily_trades >= self.cfg.max_daily_trades:
            return False, f"Daily trade limit reached ({self.cfg.max_daily_trades})"
        if asset in self.positions:
            return False, f"Already in position for {asset}"
        if len(self.positions) >= self.cfg.max_open_positions:
            return False, f"Max positions reached ({self.cfg.max_open_positions})"
        now = time.time()
        if asset in self.last_trade_time:
            elapsed = now - self.last_trade_time[asset]
            if elapsed < self.cfg.cooldown_seconds:
                return False, f"Cooldown: {int(self.cfg.cooldown_seconds - elapsed)}s left"
        return True, "OK"

    def get_balance(self, asset: str) -> Decimal:
        try:
            data = self.client.get_balances()
            if isinstance(data, list):
                for b in data:
                    if b.get("asset") == asset:
                        return Decimal(str(b.get("available", "0")))
            elif isinstance(data, dict):
                avail = data.get(asset, {})
                if isinstance(avail, dict):
                    return Decimal(str(avail.get("available", "0")))
                return Decimal(str(avail or "0"))
        except Exception as e:
            log.warning("Failed to fetch balance for %s: %s", asset, e)
        return Decimal("0")

    def compute_qty(self, price: float) -> Decimal:
        balance = self.get_balance(self.cfg.quote_asset)
        allocation = balance * Decimal(str(self.cfg.max_position_pct / 100))
        if price <= 0:
            return Decimal("0")
        qty = allocation / Decimal(str(price))
        return qty.quantize(Decimal("0.00001"), rounding=ROUND_DOWN)

    def get_current_price(self, asset: str, side: str) -> float:
        try:
            quote = self.client.compute_quote(
                base=asset, quote=self.cfg.quote_asset,
                side=side, quantity="0.01",
            )
            return float(quote.get("price", 0))
        except Exception as e:
            log.warning("Quote failed for %s: %s", asset, e)
            return 0.0

    def execute_signal(self, signal: ICTSignal) -> dict | None:
        with self.lock:
            can, reason = self.can_trade(signal.asset)
            if not can:
                log.info("Trade blocked for %s: %s", signal.asset, reason)
                return None

            try:
                qty = self.compute_qty(signal.price)
                if qty <= 0:
                    log.warning("Insufficient balance for %s", signal.asset)
                    return None

                order = self.client.create_order(
                    base=signal.asset,
                    quote=self.cfg.quote_asset,
                    side=signal.action,
                    quantity=str(qty),
                    price=str(signal.price),
                    order_type="limit",
                )

                order_id = order.get("order_id") or order.get("id", "unknown")

                position = Position(
                    asset=signal.asset, side=signal.action,
                    entry_price=signal.price, qty=float(qty),
                    order_id=order_id,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_time=time.time(),
                )
                self.positions[signal.asset] = position
                self.last_trade_time[signal.asset] = time.time()
                self.daily_trades += 1

                log.info(
                    "TRADE EXECUTED: %s %s %s @ %.2f | SL=%.2f TP=%.2f | score=%d [%s %s %s]",
                    signal.action.upper(), qty, signal.asset, signal.price,
                    signal.stop_loss, signal.take_profit, signal.score,
                    signal.structure, signal.zone, signal.kill_zone,
                )

                return {
                    "order_id": order_id,
                    "asset": signal.asset,
                    "side": signal.action,
                    "qty": str(qty),
                    "price": signal.price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "score": signal.score,
                }

            except Exception as e:
                log.error("Trade failed for %s: %s", signal.asset, e, exc_info=True)
                return None

    def check_exits(self):
        if not self.positions:
            return
        closed = []
        for asset, pos in self.positions.items():
            exit_side = "sell" if pos.side == "buy" else "buy"
            current_price = self.get_current_price(asset, exit_side)
            if current_price <= 0:
                continue

            should_close = False
            close_reason = ""

            if pos.side == "buy":
                if current_price <= pos.stop_loss:
                    should_close, close_reason = True, "STOP LOSS"
                elif current_price >= pos.take_profit:
                    should_close, close_reason = True, "TAKE PROFIT"
            else:
                if current_price >= pos.stop_loss:
                    should_close, close_reason = True, "STOP LOSS"
                elif current_price <= pos.take_profit:
                    should_close, close_reason = True, "TAKE PROFIT"

            if should_close:
                self._close_position(pos, current_price, close_reason)
                closed.append(asset)

        for asset in closed:
            del self.positions[asset]

    def _close_position(self, pos: Position, current_price: float, reason: str):
        try:
            exit_side = "sell" if pos.side == "buy" else "buy"
            self.client.create_order(
                base=pos.asset,
                quote=self.cfg.quote_asset,
                side=exit_side,
                quantity=str(pos.qty),
                price=str(current_price),
                order_type="limit",
            )

            pnl = (current_price - pos.entry_price) * pos.qty
            if pos.side == "sell":
                pnl = -pnl
            self.daily_pnl += pnl

            log.info(
                "CLOSED [%s]: %s %s @ %.2f -> %.2f | PnL=%.2f %s",
                reason, pos.side.upper(), pos.asset,
                pos.entry_price, current_price, pnl, self.cfg.quote_asset,
            )

            self.trade_history.append({
                "asset": pos.asset, "side": pos.side,
                "entry": pos.entry_price, "exit": current_price,
                "pnl": round(pnl, 2), "reason": reason,
                "duration_min": round((time.time() - pos.entry_time) / 60, 1),
            })

        except Exception as e:
            log.error("Close failed for %s: %s", pos.asset, e, exc_info=True)

    def get_status(self) -> dict:
        return {
            "positions": {
                asset: {
                    "side": p.side, "entry": p.entry_price,
                    "qty": p.qty, "sl": p.stop_loss, "tp": p.take_profit,
                    "age_min": int((time.time() - p.entry_time) / 60),
                }
                for asset, p in self.positions.items()
            },
            "daily_trades": self.daily_trades,
            "daily_pnl": round(self.daily_pnl, 2),
            "recent_trades": self.trade_history[-10:],
        }


# ─── MARKET DATA ──────────────────────────────────────────────────────────────

def fetch_candles_from_quote(client: TrueMarketsClient, asset: str, quote: str, count: int) -> list[Candle]:
    """
    Build a candle history from repeated quote snapshots.

    On first run this seeds synthetic history from a single quote. On subsequent
    calls the live quote is appended and the buffer grows organically.
    """
    try:
        q = client.compute_quote(base=asset, quote=quote, side="buy", quantity="0.01")
        price = float(q.get("price", 0))
        if price <= 0:
            return []
    except Exception as e:
        log.error("Quote fetch failed for %s: %s", asset, e)
        return []

    buf = _candle_buffers.setdefault(asset, [])
    now = time.time()
    buf.append(Candle(now, price * 0.999, price * 1.001, price * 0.999, price, 0))

    if len(buf) < count:
        seed = []
        for i in range(count - len(buf)):
            noise = (hash((asset, i)) % 1000 - 500) / 100000
            p = price * (1 + noise)
            seed.append(Candle(
                now - (count - i) * 60,
                p * 0.999, p * 1.002, p * 0.998, p, 0,
            ))
        buf[:0] = seed

    return buf[-count:]


_candle_buffers: dict[str, list[Candle]] = {}


# ─── ALGO LOOP ─────────────────────────────────────────────────────────────────

def run_algo_loop(config: BotConfig, client: TrueMarketsClient, manager: TradeManager):
    analyzer = ICTAnalyzer(config)

    log.info("ICT Algo Bot started")
    log.info("Assets: %s | Quote: %s | Poll: %ds",
             config.assets, config.quote_asset, config.poll_interval)
    log.info("Risk: %.1f%% per trade | Max positions: %d | SL: %.1fx ATR | TP: %.1fx ATR",
             config.max_position_pct, config.max_open_positions,
             config.sl_atr_mult, config.tp_atr_mult)

    while True:
        try:
            manager.check_exits()

            for asset in config.assets:
                candles = fetch_candles_from_quote(
                    client, asset, config.quote_asset, config.lookback_candles,
                )
                if not candles:
                    continue

                signal = analyzer.analyze(asset, candles)
                if signal:
                    log.info(
                        "SIGNAL: %s %s | score=%d/8 | %s %s kz=%s | OB=%s FVG=%s OTE=%s sweep=%s",
                        signal.action.upper(), signal.asset, signal.score,
                        signal.structure, signal.zone, signal.kill_zone,
                        signal.order_block, signal.fvg, signal.ote, signal.liquidity_sweep,
                    )
                    manager.execute_signal(signal)

            status = manager.get_status()
            if status["positions"]:
                log.info("Positions: %s | Daily PnL: %.2f",
                         list(status["positions"].keys()), status["daily_pnl"])

        except Exception as e:
            log.error("Algo loop error: %s", e, exc_info=True)

        time.sleep(config.poll_interval)


# ─── WEBHOOK SERVER ────────────────────────────────────────────────────────────

def create_webhook_app(config: BotConfig, manager: TradeManager):
    from flask import Flask, request as req, jsonify as jfy

    app = Flask(__name__)

    @app.route("/webhook", methods=["POST"])
    def webhook():
        data = req.get_json(silent=True)
        if not data:
            return jfy({"error": "Invalid JSON"}), 400
        if config.webhook_secret:
            if req.headers.get("X-Webhook-Secret") != config.webhook_secret:
                return jfy({"error": "Unauthorized"}), 401

        log.info("Webhook: %s", json.dumps(data))
        action = data.get("action", "").lower()
        asset = data.get("asset", "").upper()
        price = float(data.get("price", 0))
        score = int(data.get("score", 0))

        if action not in ("buy", "sell") or asset not in config.assets or price <= 0:
            return jfy({"status": "rejected", "reason": "Invalid signal"}), 400

        atr_est = price * 0.02
        sl = float(data.get("sl", price + (-1 if action == "buy" else 1) * atr_est * config.sl_atr_mult))
        tp = float(data.get("tp", price + (1 if action == "buy" else -1) * atr_est * config.tp_atr_mult))

        signal = ICTSignal(
            action=action, asset=asset, price=price, score=score,
            stop_loss=sl, take_profit=tp,
            structure=data.get("structure", "webhook"),
            zone=data.get("zone", "n/a"),
            kill_zone=data.get("kill_zone", "n/a"),
            order_block=False, fvg=False, ote=False, liquidity_sweep=False,
        )
        result = manager.execute_signal(signal)
        if result:
            return jfy({"status": "executed", **result}), 200
        return jfy({"status": "skipped"}), 200

    @app.route("/status", methods=["GET"])
    def status():
        return jfy(manager.get_status())

    @app.route("/health", methods=["GET"])
    def health():
        return jfy({"status": "ok", "positions": len(manager.positions)})

    return app


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICT Spot Trading Bot for TrueMarkets")
    parser.add_argument("--mode", choices=["auto", "webhook", "both"], default="auto")
    args = parser.parse_args()

    config = load_config()

    if not config.api_key_id or not config.private_key_file:
        log.error("Set TM_API_KEY_ID and TM_PRIVATE_KEY_FILE in .env")
        sys.exit(1)

    client = TrueMarketsClient(config.api_key_id, config.private_key_file)
    manager = TradeManager(config, client)

    log.info("=" * 60)
    log.info("  ICT SPOT TRADING BOT — TrueMarkets (TrueX)")
    log.info("=" * 60)
    log.info("Mode: %s | Assets: %s | Quote: %s",
             args.mode, config.assets, config.quote_asset)

    if args.mode == "auto":
        run_algo_loop(config, client, manager)
    elif args.mode == "webhook":
        app = create_webhook_app(config, manager)
        app.run(host="0.0.0.0", port=config.webhook_port)
    elif args.mode == "both":
        t = threading.Thread(target=run_algo_loop, args=(config, client, manager), daemon=True)
        t.start()
        app = create_webhook_app(config, manager)
        app.run(host="0.0.0.0", port=config.webhook_port)


if __name__ == "__main__":
    main()
