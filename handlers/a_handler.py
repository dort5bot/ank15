# handlers/a12_handler.py
# -*- coding: utf-8 -*-
"""
Unified Anlz Command Handler - Final
----------------------------------
Single entrypoint for Anlz commands. Merges REPORT_PROFILES and Anlz_COMMANDS
into a single COMMAND_PROFILES structure, provides clean parsing, smart
multi-symbol scanning, fallback strategies and unified report formatting.

Designed to integrate with existing AnalysisCore.process_command(...) which
is expected to accept: command, symbols, analysis_mode, required_metrics
and return a dict mapping symbol -> {metric: value, ...} for multi-symbol
mode or {metric: value,...} for single-symbol mode.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

from aiogram import Router, types

from analysis.analysis_core import AnalysisCore

logger = logging.getLogger(__name__)

router = Router(name="a12_command_router")


class CommandHandler:
    """Unified Anlz command handler.

    - COMMAND_PROFILES contains all command-level metadata: description,
      columns (report layout), metrics (required composites/macros), supports
      (single/multi/market), params (usage examples).
    - Parsing and mode detection consults supports to decide allowed modes.
    - Smart scanning tries AnalysisCore._get_trending_symbols else falls
      back to a local async scanner that uses Binance aggregator.
    """

    COMMAND_PROFILES: Dict[str, Dict[str, Any]] = {
        "/t": {
            "description": "Trend odaklı genel piyasa görünümü",
            "interpretation": "Trend yapısı ve momentum ilişkisi değerlendirilir.",
            "columns": ["trend", "vol", "regime", "risk", "core"],
            "metrics": [
                "trend_momentum_composite",
                "volatility_composite",
                "regime_composite",
                "risk_composite",
                "core_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [
                {"type": "int", "usage": "n adet varlık (örn: /t 20)"},
                {"type": "symbol", "usage": "tek sembol (örn: /t btc)"},
            ],
        },

        "/tv": {
            "description": "Volatilite odaklı piyasa görünümü",
            "interpretation": "Volatilite seviyesi ve fiyat oynaklığı analiz edilir.",
            "columns": ["vol", "vol_mom", "entropy", "complexity"],
            "metrics": [
                "volatility_composite",
                "volatility_momentum_composite",
                "entropy_fractal_composite",
                "complexity_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [
                {"type": "int", "usage": "n adet varlık (örn: /tv 20)"},
                {"type": "symbol", "usage": "tek sembol (örn: /tv btc)"},
            ],
        },

        "/tvm": {
            "description": "Volatilite momentumu analizi",
            "interpretation": "Volatilitenin ivmesi ve değişim hızı değerlendirilir.",
            "columns": ["vol_mom", "vol", "hurst", "entropy"],
            "metrics": [
                "volatility_momentum_composite",
                "volatility_composite",
                "entropy_fractal_composite",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },

        "/tr": {
            "description": "Regime (piyasa rejimi) analizi",
            "interpretation": "Piyasanın risk-on / risk-off / nötr durumları incelenir.",
            "columns": ["regime", "trend", "risk", "core"],
            "metrics": [
                "regime_composite",
                "trend_momentum_composite",
                "risk_composite",
                "core_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },

        "/trc": {
            "description": "Risk kompozit analizi",
            "interpretation": "Sistemik risk, kırılganlık ve risk yoğunluğu değerlendirilir.",
            "columns": ["risk", "l_risk", "of_stress", "micro"],
            "metrics": [
                "risk_composite",
                "liquidity_risk_composite",
                "order_flow_stress_composite",
                "microstructure_macro",
            ],
            # liquidity/risk metrics are most meaningful single-symbol; multi allowed
            "supports": ["single", "multi"],
            "params": [],
        },

        "/tl": {
            "description": "Likidite analizi",
            "interpretation": "Likidite derinliği ve emir defteri akıcılığı analiz edilir.",
            "columns": ["liquidity", "l_risk", "micro"],
            "metrics": [
                "liquidity_composite",
                "liquidity_risk_composite",
                "microstructure_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },

        "/tlr": {
            "description": "Likidite + risk birleşik analizi",
            "interpretation": "Likidite koşulları risk ortamıyla beraber ele alınır.",
            "columns": ["l_risk", "liquidity", "of_stress", "micro"],
            "metrics": [
                "liquidity_risk_composite",
                "liquidity_composite",
                "order_flow_stress_composite",
                "microstructure_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },

        "/te": {
            "description": "Entropi ve fraktal yapı analizi",
            "interpretation": "Piyasanın kaotik yapısı, düzensizlik seviyesi ve fraktal düzeni değerlendirilir.",
            "columns": ["entropy", "fdi", "hurst", "complexity"],
            "metrics": [
                "entropy_fractal_composite",
                "volatility_composite",
                "complexity_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },

        "/to": {
            "description": "Order flow stres analizi",
            "interpretation": "Emir akışındaki baskı, dengesizlik ve agresif emir davranışları incelenir.",
            "columns": ["of_stress", "liquidity", "taker", "micro"],
            "metrics": [
                "order_flow_stress_composite",
                "liquidity_composite",
                "microstructure_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },

        "/tf": {
            "description": "Akış dinamikleri analizi",
            "interpretation": "Likidite akışı, hacim yoğunluğu ve emir dengesi trendi analiz edilir.",
            "columns": ["flow", "sentiment", "funding", "macro_sent"],
            "metrics": [
                "flow_dynamics_composite",
                "sentiment_composite",
                "market_sentiment_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },

        "/ts": {
            "description": "Piyasa duyarlılığı (sentiment) analizi",
            "interpretation": "Pozitif/negatif duyarlılık, davranışsal etkiler ve trend uyumu değerlendirilir.",
            "columns": ["sentiment", "flow", "funding", "macro_sent"],
            "metrics": [
                "sentiment_composite",
                "flow_dynamics_composite",
                "market_sentiment_macro",
            ],
            "supports": ["single", "multi", "market"],
            "params": [],
        },
    }

    DEFAULTS = {
        "fallback_command": "/t",
        "error_response": "Komut bulunamadı. /t ile genel görünüm alabilirsiniz.",
        "precision": 4,
        "default_pairing": "USDT",
        "max_multi": 12,
    }

    def __init__(self) -> None:
        self.analysis_core = AnalysisCore()
        logger.info("✅ Anlz CommandHandler initialized - Unified COMMAND_PROFILES")

    # -----------------------------
    # Parsing & mode determination
    # -----------------------------
    def parse_command(self, text: str) -> Optional[dict]:
        parts = text.strip().split()
        if not parts:
            logger.debug("Empty command received")
            return None

        cmd = parts[0].lower()
        args = parts[1:]

        profile = self.COMMAND_PROFILES.get(cmd)
        if not profile:
            logger.debug(f"Ignoring non-Anlz command: {cmd}")
            return None

        mode_hint = self._determine_analysis_mode(args, profile)
        return {"cmd": cmd, "args": args, "profile": profile, "mode": mode_hint}

    def _determine_analysis_mode(self, args: List[str], profile: Dict[str, Any]) -> str:
        supports = profile.get("supports", ["single", "multi", "market"])
        if not args:
            return "market" if "market" in supports else "multi" if "multi" in supports else "single"

        first = args[0].upper()
        if first.isdigit():
            return "multi" if "multi" in supports else "single"
        if self._is_valid_symbol_format(first):
            return "single" if "single" in supports else "multi"
        # default fallback
        return "single"

    # -----------------------------
    # Symbol utilities
    # -----------------------------
    def _normalize_symbol(self, symbol_input: str) -> str:
        if not symbol_input or not isinstance(symbol_input, str):
            return "BTCUSDT"
        clean = symbol_input.upper().strip()
        if len(clean) <= 5 and clean.isalpha():
            return f"{clean}USDT"
        if clean.endswith(("USDT", "FDUSD", "BTC", "ETH", "BNB")):
            return clean
        return f"{clean}USDT"

    def _is_valid_symbol_format(self, symbol: str) -> bool:
        if not symbol or not isinstance(symbol, str):
            return False
        patterns = [
            r'^[A-Z]{3,6}USDT$',
            r'^[A-Z]{3,6}FDUSD$',
            r'^[A-Z]{3,6}BTC$',
            r'^[A-Z]{3,6}ETH$',
            r'^[A-Z]{3,6}BNB$',
        ]
        return any(re.match(p, symbol) for p in patterns)

    # -----------------------------
    # Smart symbol extraction
    # -----------------------------
    async def _extract_symbols(self, args: List[str], mode: str, cmd: str) -> List[str]:
        logger.debug(f"_extract_symbols: args={args}, mode={mode}, cmd={cmd}")
        if mode == "market":
            return []

        if mode == "single" and args:
            return [self._normalize_symbol(args[0])]

        if mode == "multi" and args:
            try:
                count = min(int(args[0]), self.DEFAULTS["max_multi"]) if args else self.DEFAULTS["max_multi"]
            except Exception:
                count = self.DEFAULTS["max_multi"]

            # Prefer AnalysisCore helper when available
            try:
                if hasattr(self.analysis_core, '_get_trending_symbols'):
                    scan_type = self._get_scan_type_for_command(cmd)
                    symbols = await self.analysis_core._get_trending_symbols(count, scan_type)
                    if symbols:
                        logger.info(f"🔍 Core smart scan found {len(symbols)} symbols for {cmd}")
                        return symbols
                # fallback to local scan
                symbols = await self._smart_symbol_scan(count, cmd)
                return symbols
            except Exception as e:
                logger.warning(f"Smart symbol scan failed: {e}")
                return self._get_fallback_symbols(count, cmd)

        return []

    async def _smart_symbol_scan(self, count: int, cmd: str) -> List[str]:
        try:
            aggregator = await self.analysis_core._get_aggregator()
            ticker_data = await aggregator.get_data('futures_ticker_24hr')
            if not isinstance(ticker_data, list):
                return self._get_fallback_symbols(count, cmd)

            usdt_pairs = [s for s in ticker_data if isinstance(s, dict) and s.get('symbol', '').endswith('USDT')]
            if not usdt_pairs:
                return self._get_fallback_symbols(count, cmd)

            scan_type = self._get_scan_type_for_command(cmd)
            scored = await self._score_symbols(usdt_pairs, scan_type)
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scored[:count]]
        except Exception as e:
            logger.error(f"Local smart scan failed: {e}")
            return self._get_fallback_symbols(count, cmd)

    async def _score_symbols(self, symbols: List[dict], scan_type: str) -> List[tuple]:
        scored: List[tuple] = []
        for s in symbols:
            try:
                sym = s.get('symbol', '')
                volume = float(s.get('quoteVolume', 0))
                price_change = float(s.get('priceChangePercent', 0))
                price_change_abs = abs(float(s.get('priceChange', 0)))

                if scan_type == 'general':
                    volume_score = min(volume / 50_000_000, 1.0) * 50
                    change_score = abs(price_change) * 0.3
                    momentum_score = min(price_change_abs / 1000, 1.0) * 20
                    total = volume_score + change_score + momentum_score
                elif scan_type == 'opportunity':
                    if price_change < -1.0:
                        volume_score = min(volume / 10_000_000, 1.0) * 70
                        drop_score = abs(price_change) * 2.0
                        total = volume_score + drop_score
                    else:
                        total = 0
                elif scan_type == 'volatility':
                    volatility_score = min(abs(price_change) * 5, 60)
                    volume_score = min(volume / 5_000_000, 1.0) * 40
                    total = volatility_score + volume_score
                else:
                    total = volume / 1_000_000

                if total > 0:
                    scored.append((sym, total))
            except Exception:
                continue
        return scored

    def _get_scan_type_for_command(self, command: str) -> str:
        return {
            "/t": "general",
            "/tp": "volatility",
            "/tb": "opportunity",
        }.get(command, "general")

    def _get_fallback_symbols(self, count: int, command: str) -> List[str]:
        fallback = {
            "/t": ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT'],
            "/tp": ['SOLUSDT', 'AVAXUSDT', 'DOGEUSDT', 'ADAUSDT', 'MATICUSDT', 'LTCUSDT'],
            "/tb": ['ADAUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'ATOMUSDT'],
        }
        return fallback.get(command, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])[:count]

    # -----------------------------
    # Main handler
    # -----------------------------
    async def handle(self, text: str) -> Optional[dict]:
        parsed = self.parse_command(text)
        if not parsed:
            return None

        cmd = parsed['cmd']
        args = parsed['args']
        profile = parsed['profile']
        mode = parsed['mode']  # one of: 'single','multi','market'

        try:
            symbols = await self._extract_symbols(args, mode, cmd)
            if mode == 'single' and not symbols:
                return {'error': 'Geçersiz sembol formatı'}

            logger.info(f"Processing {cmd} mode={mode} symbols={symbols}")

            # ask core to compute required metrics
            required = profile.get('metrics', [])
            raw = await self.analysis_core.process_command(
                command=cmd,
                symbols=symbols,
                analysis_mode=mode,
                required_metrics=required,
                command_mode=cmd.strip('/'),
            )

            # raw expected to be {SYMBOL: {metric: val}} for multi/market
            # or {metric: val} for single. Normalize to unified dict for formatting.
            if mode == 'multi' or mode == 'market':
                # Try to accept raw as-is if mapping
                if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
                    table_data = raw
                elif isinstance(raw, dict) and 'result' in raw and isinstance(raw['result'], dict):
                    # maybe core returned composite under result->score; fall back
                    table_data = {s: raw['result'] for s in symbols} if symbols else raw.get('symbols', {})
                else:
                    # fallback: empty mapping
                    table_data = {s: {} for s in symbols}

                return {
                    'command': cmd,
                    'mode': mode,
                    'symbols': symbols,
                    'data': table_data,
                    'columns': profile.get('columns', [])
                }

            # single
            if mode == 'single' and symbols:
                coin = symbols[0]
                single_data = {}
                if isinstance(raw, dict) and coin in raw:
                    single_data = raw[coin]
                elif isinstance(raw, dict) and all(isinstance(v, (int, float)) for v in raw.values()):
                    single_data = raw
                else:
                    single_data = raw.get(coin, {}) if isinstance(raw, dict) else {}

                return {
                    'command': cmd,
                    'mode': mode,
                    'symbols': [coin],
                    'data': {coin: single_data},
                    'columns': profile.get('columns', [])
                }

        except Exception as e:
            logger.exception(f"Anlz Command handling error: {e}")
            return {"error": f"İşlem hatası: {str(e)}"}


# -----------------------------
# Formatting helpers (used by router)
# -----------------------------

def format_multi_table(result: dict) -> str:
    columns: List[str] = result.get('columns', [])
    table_data: Dict[str, Dict[str, Any]] = result.get('data', {})

    header = "Coin | " + " | ".join(c.capitalize() for c in columns)
    header += "\n" + ("-" * max(len(header), 20)) + "\n"

    lines: List[str] = [header]
    for coin, values in table_data.items():
        row = [f"{coin:<9}"]
        for c in columns:
            v = values.get(c)
            if v is None:
                row.append(" - ")
            else:
                try:
                    row.append(f"{v:.2f}")
                except Exception:
                    row.append(str(v))
        lines.append(" | ".join(row))
    return "\n".join(lines)


def format_single_block(result: dict) -> str:
    columns: List[str] = result.get('columns', [])
    data: Dict[str, Dict[str, Any]] = result.get('data', {})
    coin = (result.get('symbols') or [None])[0]
    if not coin:
        return "❌ Veri yok"

    out = [f"<b>{coin} Analizi</b>"]
    vals = data.get(coin, {})
    for c in columns:
        v = vals.get(c)
        if v is None:
            continue
        try:
            out.append(f"• <b>{c.capitalize()}</b>: {v:.3f}")
        except Exception:
            out.append(f"• <b>{c.capitalize()}</b>: {v}")

    return "\n".join(out)


# -----------------------------
# Router glue
# -----------------------------
handler_instance = CommandHandler()

@router.message(lambda msg: msg.text and msg.text.split()[0].lower() in handler_instance.COMMAND_PROFILES)
async def handle_a12_command(message: types.Message):
    text = message.text.strip()
    result = await handler_instance.handle(text)
    if result is None:
        return

    if 'error' in result:
        await message.answer(f"⚠️ {result['error']}")
        return

    # decide formatting
    mode = result.get('mode')
    if mode in ('multi', 'market'):
        formatted = format_multi_table(result)
        await message.answer(formatted)
    else:
        formatted = format_single_block(result)
        await message.answer(formatted, parse_mode="HTML")


# -----------------------------
# Quick manual test when run as script
# -----------------------------
if __name__ == '__main__':
    import asyncio

    async def _test():
        h = CommandHandler()
        tests = [
            "/t BTCUSDT",
            "/t 5",
            "/tv BTC",
            "/tvm 3",
        ]
        for t in tests:
            print("---", t)
            r = await h.handle(t)
            print(r)

    asyncio.run(_test())
