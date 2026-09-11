import asyncio
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

try:
  from sklearn.linear_model import LogisticRegression
  from sklearn.preprocessing import OneHotEncoder

  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False

BASE_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
JSON_FILE = Path("crazytime_history.json")

# Fatias exatas da roleta do Crazy Time (Total = 54 fatias)
SECTOR_SLOTS = {
    "1": 21,
    "2": 13,
    "5": 7,
    "10": 4,
    "CoinFlip": 4,
    "Pachinko": 2,
    "CashHunt": 2,
    "CrazyBonus": 1,
}
SECTORS = list(SECTOR_SLOTS.keys())
N_SLOTS = float(sum(SECTOR_SLOTS.values()))

THEORETICAL_PROBS = {k: v / N_SLOTS for k, v in SECTOR_SLOTS.items()}
EXPECTED_GAPS = {k: N_SLOTS / v for k, v in SECTOR_SLOTS.items()}

# Janelas e decaimento (em rodadas, 0 = mais recente)
EWMA_HALF_LIFE = 280
HOT_WINDOW = 120
DEALER_MIN_SPINS = 80
SEQ_HALF_LIFE = 220
SEQ2_MIN_SAMPLES = 12
PAYOUT_HALF_LIFE = 200
TOPSLOT_HALF_LIFE = 200


def clear_console():
  os.system("cls" if os.name == "nt" else "clear")


class CrazyTimeAnalyzer:

  def __init__(self):
    self.history_dict = {}
    self._sklearn_model = None
    self._sklearn_encoder = None
    self._sklearn_trained_on = 0
    self.load_from_json()

  def load_from_json(self):
    """Carrega todo o histórico salvo no JSON sem limitação de tamanho."""
    if JSON_FILE.exists():
      try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
          data = json.load(f)
          for item in data:
            if "id" in item:
              self.history_dict[item["id"]] = item
      except Exception as e:
        print(f"⚠️ Erro ao carregar JSON local: {e}")

  def save_to_json(self):
    """Ordena por data e persiste TODAS as rodadas acumuladas sem limite maximo."""
    try:
      sorted_history = sorted(
          list(self.history_dict.values()),
          key=lambda x: x.get("settledAt", ""),
          reverse=True,
      )

      with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
      print(f"❌ Erro ao salvar JSON: {e}")

  def parse_item(self, item: dict) -> dict:
    data = item.get("data", {})
    result = data.get("result", {}).get("outcome", {})
    wheel_res = result.get("wheelResult", {})
    top_slot = result.get("topSlot", {})
    bonus_data = wheel_res.get("bonus", {})

    winners = item.get("winners", [])
    top_winner_name = winners[0].get("screenName", "-") if winners else "-"
    top_winner_val = winners[0].get("winnings", 0.0) if winners else 0.0

    return {
        "id": item.get("id"),
        "transmissionId": item.get("transmissionId"),
        "gameId": data.get("id"),
        "startedAt": data.get("startedAt"),
        "settledAt": data.get("settledAt"),
        "dealerName": data.get("dealer", {}).get("name", "-"),
        "wheelSector": wheel_res.get("wheelSector"),
        "resultType": wheel_res.get("type"),
        "topSlotSector": top_slot.get("wheelSector"),
        "topSlotMultiplier": top_slot.get("multiplier", 1),
        "isTopSlotMatched": result.get(
            "isTopSlotMatchedToWheelResult", False
        ),
        "totalWinners": item.get("totalWinners", 0),
        "totalAmount": item.get("totalAmount", 0),
        "numOfParticipants": data.get("numOfParticipants", 0),
        "topWinnerName": top_winner_name,
        "topWinnerVal": top_winner_val,
        "maxMultiplier": result.get("maxMultiplier", 1),
        "bonusType": bonus_data.get("type"),
        "bonusMultiplier": bonus_data.get("bonusMultiplier", {}).get("value"),
    }

  async def load_initial_history(
      self, client: httpx.AsyncClient, pages_count: int = 5
  ):
    print("📥 Sincronizando rodadas recentes da API...")
    initial_count = len(self.history_dict)

    for page in range(pages_count):
      params = {
          "page": page,
          "size": 500,
          "sort": "data.settledAt,desc",
          "duration": 6,
          "wheelResults": "Pachinko,CashHunt,CrazyBonus,CoinFlip,1,2,5,10",
          "isTopSlotMatched": "true,false",
          "tableId": "CrazyTime0000001",
      }
      try:
        response = await client.get(BASE_URL, params=params, timeout=10.0)
        items = response.json()
        if not items or not isinstance(items, list):
          break
        for item in items:
          parsed = self.parse_item(item)
          if parsed["id"]:
            self.history_dict[parsed["id"]] = parsed
      except Exception:
        break

    new_added = len(self.history_dict) - initial_count
    if new_added > 0 or initial_count == 0:
      self.save_to_json()

  def get_ordered_df(self) -> pd.DataFrame:
    sorted_items = sorted(
        list(self.history_dict.values()),
        key=lambda x: x.get("settledAt", ""),
        reverse=True,
    )
    return pd.DataFrame(sorted_items)

  @staticmethod
  def _valid_sectors(series: pd.Series) -> np.ndarray:
    values = series.astype(str).to_numpy()
    mask = np.isin(values, SECTORS)
    return values[mask]

  @staticmethod
  def _ewma_counts(sectors_newest_first: np.ndarray, half_life: float):
    n = len(sectors_newest_first)
    if n == 0:
      zeros = {s: 0.0 for s in SECTORS}
      return zeros, 0.0
    decay = np.log(2.0) / max(half_life, 1.0)
    weights = np.exp(-decay * np.arange(n, dtype=np.float64))
    counts = {s: 0.0 for s in SECTORS}
    for sector, weight in zip(sectors_newest_first, weights):
      counts[sector] += weight
    return counts, float(weights.sum())

  @staticmethod
  def _dirichlet_posterior(counts: dict, total_weight: float) -> dict:
    """Prior = fatias físicas da roda; posterior = (prior + evidência)."""
    denom = N_SLOTS + max(total_weight, 0.0)
    return {
        s: (SECTOR_SLOTS[s] + counts.get(s, 0.0)) / denom for s in SECTORS
    }

  @staticmethod
  def _normalize(probs: dict) -> dict:
    total = sum(probs.values()) or 1.0
    return {s: probs[s] / total for s in SECTORS}

  def _smoothed_from_weights(self, weights: dict, total_w: float, prior_scale: float = 0.45):
    """Mistura contagem empírica com a roda física (prior fraco = sequência vale mais)."""
    alphas = {s: SECTOR_SLOTS[s] * prior_scale for s in SECTORS}
    denom = sum(alphas.values()) + max(total_w, 0.0)
    return {
        s: (alphas[s] + weights.get(s, 0.0)) / denom for s in SECTORS
    }

  def _seq1_probs(self, sectors_newest_first: np.ndarray, last: str):
    n = len(sectors_newest_first)
    empty_w = {s: 0.0 for s in SECTORS}
    if n < 8 or last not in SECTOR_SLOTS:
      return self._normalize(dict(THEORETICAL_PROBS)), 0.0, empty_w

    decay = np.log(2.0) / SEQ_HALF_LIFE
    weights = dict(empty_w)
    total_w = 0.0
    raw_recent = dict(empty_w)
    raw_n = 0.0
    for i in range(1, n):
      if sectors_newest_first[i] != last:
        continue
      nxt = sectors_newest_first[i - 1]
      w = float(np.exp(-decay * (i - 1)))
      weights[nxt] += w
      total_w += w
      raw_recent[nxt] += 1.0
      raw_n += 1.0

    probs = self._smoothed_from_weights(weights, total_w, prior_scale=0.40)
    display = (
        {s: raw_recent[s] / raw_n for s in SECTORS}
        if raw_n
        else dict(THEORETICAL_PROBS)
    )
    return probs, raw_n, display

  def _seq2_probs(self, sectors_newest_first: np.ndarray):
    n = len(sectors_newest_first)
    if n < 3:
      return None, 0.0
    last = sectors_newest_first[0]
    prev = sectors_newest_first[1]
    decay = np.log(2.0) / max(SEQ_HALF_LIFE * 1.3, 1.0)
    weights = {s: 0.0 for s in SECTORS}
    total_w = 0.0
    hits = 0.0
    for i in range(2, n):
      if sectors_newest_first[i] != prev or sectors_newest_first[i - 1] != last:
        continue
      nxt = sectors_newest_first[i - 2]
      w = float(np.exp(-decay * (i - 2)))
      weights[nxt] += w
      total_w += w
      hits += 1.0
    if hits < SEQ2_MIN_SAMPLES:
      return None, hits
    return self._smoothed_from_weights(weights, total_w, prior_scale=0.55), hits

  def _conditional_next_from_mask(
      self, df: pd.DataFrame, mask: pd.Series, half_life: float, prior_scale: float
  ):
    """P(próximo setor | condição na rodada atual), com decaimento temporal."""
    work = df.copy()
    work["_ok"] = work["wheelSector"].isin(SECTORS)
    if not work["_ok"].any():
      return dict(THEORETICAL_PROBS), 0.0

    decay = np.log(2.0) / max(half_life, 1.0)
    weights = {s: 0.0 for s in SECTORS}
    total_w = 0.0
    hits = 0.0
    n = len(work)
    mask_vals = mask.fillna(False).to_numpy()
    sectors = work["wheelSector"].astype(str).to_numpy()
    for i in range(1, n):
      if not mask_vals[i]:
        continue
      nxt = sectors[i - 1]
      if nxt not in weights:
        continue
      w = float(np.exp(-decay * (i - 1)))
      weights[nxt] += w
      total_w += w
      hits += 1.0
    if hits < 8:
      return dict(THEORETICAL_PROBS), hits
    return self._smoothed_from_weights(weights, total_w, prior_scale), hits

  def _topslot_next_probs(self, df: pd.DataFrame, top_slot: str, matched: bool):
    if not top_slot or str(top_slot) == "nan":
      return dict(THEORETICAL_PROBS), 0.0, dict(THEORETICAL_PROBS)
    same_slot = df["topSlotSector"].astype(str) == str(top_slot)
    p_slot, n_slot = self._conditional_next_from_mask(
        df, same_slot, TOPSLOT_HALF_LIFE, prior_scale=0.50
    )
    match_mask = df["isTopSlotMatched"].fillna(False).astype(bool) == bool(matched)
    p_match, _ = self._conditional_next_from_mask(
        df, match_mask, TOPSLOT_HALF_LIFE, prior_scale=0.70
    )
    blended = {s: 0.72 * p_slot[s] + 0.28 * p_match[s] for s in SECTORS}
    return self._normalize(blended), n_slot, p_slot

  def _payout_next_probs(self, df: pd.DataFrame, last_amount: float):
    amounts = pd.to_numeric(df["totalAmount"], errors="coerce").fillna(0.0)
    if amounts.max() <= 0:
      return dict(THEORETICAL_PROBS), 0.0
    q33, q66 = amounts.quantile(0.33), amounts.quantile(0.66)
    if last_amount >= q66:
      bucket = amounts >= q66
      label = "alto"
    elif last_amount <= q33:
      bucket = amounts <= q33
      label = "baixo"
    else:
      bucket = (amounts > q33) & (amounts < q66)
      label = "médio"
    probs, n = self._conditional_next_from_mask(
        df, bucket, PAYOUT_HALF_LIFE, prior_scale=0.60
    )
    return probs, n, label

  def _payout_heat(self, df: pd.DataFrame) -> dict:
    """Quanto cada setor pagou recentemente vs a média (não é probabilidade)."""
    recent = df.head(min(len(df), 250)).copy()
    recent = recent[recent["wheelSector"].isin(SECTORS)]
    if recent.empty:
      return {s: 0.5 for s in SECTORS}
    recent["totalAmount"] = pd.to_numeric(recent["totalAmount"], errors="coerce").fillna(0.0)
    overall = float(recent["totalAmount"].mean() or 1.0)
    heat = {}
    for s in SECTORS:
      sub = recent[recent["wheelSector"] == s]["totalAmount"]
      avg = float(sub.mean()) if len(sub) else overall
      ratio = avg / overall if overall else 1.0
      heat[s] = float(np.clip(ratio / 3.0, 0.05, 1.0))
    return heat

  def _markov_pvalue(self, sectors_newest_first: np.ndarray) -> float:
    n = len(sectors_newest_first)
    if n < 80:
      return 1.0
    chrono = sectors_newest_first[::-1]
    idx = {s: i for i, s in enumerate(SECTORS)}
    table = np.zeros((len(SECTORS), len(SECTORS)), dtype=np.float64)
    for a, b in zip(chrono[:-1], chrono[1:]):
      table[idx[a], idx[b]] += 1.0
    table += 0.5
    try:
      _, pvalue, _, _ = chi2_contingency(table)
      return float(pvalue)
    except Exception:
      return 1.0

  def _dealer_posterior(self, df: pd.DataFrame, dealer: str) -> dict | None:
    if not dealer or dealer == "-":
      return None
    subset = df[df["dealerName"] == dealer]
    sectors = self._valid_sectors(subset["wheelSector"])
    if len(sectors) < DEALER_MIN_SPINS:
      return None
    counts, total = self._ewma_counts(sectors, half_life=180)
    return self._dirichlet_posterior(counts, total)

  def _hot_multipliers(self, sectors_newest_first: np.ndarray) -> dict:
    window = sectors_newest_first[:HOT_WINDOW]
    n = max(len(window), 1)
    uniq, cnts = np.unique(window, return_counts=True)
    observed = dict(zip(uniq.tolist(), cnts.tolist()))
    multipliers = {}
    for s in SECTORS:
      rate = observed.get(s, 0) / n
      ratio = rate / THEORETICAL_PROBS[s] if THEORETICAL_PROBS[s] else 1.0
      multipliers[s] = 0.80 + 0.20 * float(np.clip(ratio, 0.45, 1.70))
    return multipliers

  def _sklearn_next_probs(self, df: pd.DataFrame) -> dict | None:
    """Logística no estado da última rodada: setor, top slot, payout, dealer, hora."""
    if not HAS_SKLEARN or len(df) < 280:
      return None

    work = df.copy()
    work["settledAt"] = pd.to_datetime(work["settledAt"], errors="coerce", utc=True)
    work["totalAmount"] = pd.to_numeric(work["totalAmount"], errors="coerce").fillna(0.0)
    work["maxMultiplier"] = pd.to_numeric(work["maxMultiplier"], errors="coerce").fillna(1.0)
    work["topSlotMultiplier"] = pd.to_numeric(
        work["topSlotMultiplier"], errors="coerce"
    ).fillna(1.0)
    work = work.dropna(subset=["settledAt", "wheelSector"])
    work = work[work["wheelSector"].isin(SECTORS)]
    if len(work) < 280:
      return None

    chrono = work.iloc[::-1].reset_index(drop=True)
    y = chrono["wheelSector"].iloc[1:].to_numpy()
    state = chrono.iloc[:-1]

    cat = np.column_stack(
        [
            state["wheelSector"].astype(str).to_numpy(),
            state["topSlotSector"].fillna("-").astype(str).to_numpy(),
            state["dealerName"].fillna("-").astype(str).to_numpy(),
        ]
    )
    hours = state["settledAt"].dt.hour.to_numpy()
    numeric = np.column_stack(
        [
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.log1p(state["totalAmount"].to_numpy()),
            np.log1p(state["maxMultiplier"].to_numpy()),
            np.log1p(state["topSlotMultiplier"].to_numpy()),
            state["isTopSlotMatched"].fillna(False).astype(float).to_numpy(),
        ]
    )

    n_rows = len(y)
    if n_rows != self._sklearn_trained_on or self._sklearn_model is None:
      enc = OneHotEncoder(handle_unknown="ignore", min_frequency=8)
      cat_oh = enc.fit_transform(cat)
      cat_arr = cat_oh.toarray() if hasattr(cat_oh, "toarray") else np.asarray(cat_oh)
      x = np.hstack([cat_arr, numeric])
      model = LogisticRegression(
          solver="lbfgs",
          max_iter=400,
          C=0.55,
      )
      model.fit(x, y)
      self._sklearn_model = model
      self._sklearn_encoder = enc
      self._sklearn_trained_on = n_rows
    else:
      enc = self._sklearn_encoder
      model = self._sklearn_model

    latest = work.iloc[0]
    cat_now = np.array(
        [
            [
                str(latest["wheelSector"]),
                str(latest.get("topSlotSector") or "-"),
                str(latest.get("dealerName") or "-"),
            ]
        ]
    )
    hour = latest["settledAt"].hour
    numeric_now = np.array(
        [
            [
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                np.log1p(float(latest["totalAmount"] or 0)),
                np.log1p(float(latest["maxMultiplier"] or 1)),
                np.log1p(float(latest["topSlotMultiplier"] or 1)),
                1.0 if latest.get("isTopSlotMatched") else 0.0,
            ]
        ]
    )
    cat_oh = enc.transform(cat_now)
    cat_arr = cat_oh.toarray() if hasattr(cat_oh, "toarray") else np.asarray(cat_oh)
    x_now = np.hstack([cat_arr, numeric_now])

    classes = list(model.classes_)
    raw = model.predict_proba(x_now)[0]
    probs = {s: THEORETICAL_PROBS[s] for s in SECTORS}
    for cls, p in zip(classes, raw):
      if cls in probs:
        probs[cls] = float(p)
    return self._normalize(probs)

  @staticmethod
  def _unit(values: dict) -> dict:
    mx = max(values.values()) or 1.0
    return {k: (v / mx) for k, v in values.items()}

  def calculate_potency(self, df: pd.DataFrame):
    total_rounds = len(df)
    if total_rounds == 0:
      return None, {}

    sectors = self._valid_sectors(df["wheelSector"])
    if len(sectors) == 0:
      return None, {}

    latest = df.iloc[0]
    latest_sector = str(latest["wheelSector"])
    latest_dealer = str(latest.get("dealerName") or "-")
    latest_top = latest.get("topSlotSector")
    latest_match = bool(latest.get("isTopSlotMatched"))
    latest_amount = float(pd.to_numeric(latest.get("totalAmount"), errors="coerce") or 0)

    ewma_counts, ewma_n = self._ewma_counts(sectors, EWMA_HALF_LIFE)
    p_global = self._dirichlet_posterior(ewma_counts, ewma_n)
    p_dealer = self._dealer_posterior(df, latest_dealer)
    if p_dealer:
      p_base = {s: 0.70 * p_global[s] + 0.30 * p_dealer[s] for s in SECTORS}
    else:
      p_base = dict(p_global)
    hot = self._hot_multipliers(sectors)
    p_base = self._normalize({s: p_base[s] * hot[s] for s in SECTORS})

    p_seq1, seq1_n, seq1_raw = self._seq1_probs(sectors, latest_sector)
    p_seq2, seq2_n = self._seq2_probs(sectors)
    p_top, top_n, _ = self._topslot_next_probs(df, latest_top, latest_match)
    p_pay, pay_n, pay_bucket = self._payout_next_probs(df, latest_amount)
    pay_heat = self._payout_heat(df)
    p_sk = self._sklearn_next_probs(df)
    markov_p = self._markov_pvalue(sectors)

    # Sequência sempre entra; chi² só ajusta o peso (p baixo = mais sequência).
    seq_boost = 0.22 + 0.16 * float(np.clip(1.0 - markov_p, 0.0, 1.0))
    w_seq2 = 0.11 if p_seq2 is not None else 0.0
    w_sk = 0.10 if p_sk else 0.0
    w_base = 0.16
    w_top = 0.14
    w_pay = 0.10
    w_heat = 0.07
    w_delay = 0.10
    w_seq1 = seq_boost
    leftover = 1.0 - (w_seq1 + w_seq2 + w_sk + w_base + w_top + w_pay + w_heat + w_delay)
    w_seq1 += leftover

    sector_pos = {s: total_rounds for s in SECTORS}
    for i, s in enumerate(sectors):
      if sector_pos[s] == total_rounds:
        sector_pos[s] = i

    delay_unit = {}
    for s in SECTORS:
      gap = sector_pos[s]
      delay_ratio = gap / EXPECTED_GAPS[s]
      delay_unit[s] = float(np.clip(delay_ratio, 0.0, 2.2) / 2.2)

    u_base = self._unit(p_base)
    u_seq1 = self._unit(p_seq1)
    u_seq2 = self._unit(p_seq2) if p_seq2 else {s: 0.0 for s in SECTORS}
    u_top = self._unit(p_top)
    u_pay = self._unit(p_pay)
    u_sk = self._unit(p_sk) if p_sk else {s: 0.0 for s in SECTORS}

    mixed = {}
    scores = {}
    for s in SECTORS:
      mixed[s] = (
          w_base * p_base[s]
          + w_seq1 * p_seq1[s]
          + w_seq2 * (p_seq2[s] if p_seq2 else 0.0)
          + w_top * p_top[s]
          + w_pay * p_pay[s]
          + w_sk * (p_sk[s] if p_sk else 0.0)
      )
    mixed = self._normalize(mixed)

    def _lift(p_map, sector):
      theo = THEORETICAL_PROBS[sector]
      return float(np.clip((p_map[sector] / theo) if theo else 1.0, 0.35, 2.4))

    for s in SECTORS:
      # Rankeia o que o contexto empurra ACIMA da roda, senão o Top 3 vira sempre 1-2-5.
      scores[s] = (
          0.30 * _lift(p_seq1, s)
          + 0.12 * (_lift(p_seq2, s) if p_seq2 else 1.0)
          + 0.16 * _lift(p_top, s)
          + 0.10 * _lift(p_pay, s)
          + 0.08 * _lift(p_base, s)
          + 0.06 * (_lift(p_sk, s) if p_sk else 1.0)
          + 0.10 * (1.0 + delay_unit[s])
          + 0.08 * (0.5 + pay_heat[s])
      )

    metrics = {}
    max_score = max(scores.values()) or 1.0
    for sector in SECTORS:
      p = mixed[sector]
      theo = THEORETICAL_PROBS[sector]
      gap = sector_pos[sector]
      delay_ratio = gap / EXPECTED_GAPS[sector]
      sector_df = df[df["wheelSector"] == sector]
      avg_amount = sector_df["totalAmount"].mean() if len(sector_df) > 0 else 0
      avg_multiplier = (
          sector_df["maxMultiplier"].mean() if len(sector_df) > 0 else 1
      )
      match_rate = (
          float(sector_df["isTopSlotMatched"].mean() * 100)
          if len(sector_df) > 0
          else 0.0
      )
      potency_score = 100.0 * (scores[sector] / max_score)

      lift = mixed[sector] / theo if theo else 1.0
      metrics[sector] = {
          "Score": round(potency_score, 2),
          "P(prox)": f"{p * 100:.1f}%",
          "Lift": f"{lift:.2f}x",
          f"Pós-[{latest_sector}]": f"{seq1_raw.get(sector, 0.0) * 100:.1f}%",
          "Seq2": (
              f"{p_seq2[sector] * 100:.1f}%" if p_seq2 else "-"
          ),
          "TopSlot": f"{p_top.get(sector, 0.0) * 100:.1f}%",
          "PagoProx": f"{p_pay.get(sector, 0.0) * 100:.1f}%",
          "Atraso": gap,
          "Relativo": f"{delay_ratio:.1f}x",
          "Match%": f"{match_rate:.0f}%",
          "Média Pago": f"€{avg_amount:,.0f}",
          "Mult": f"{avg_multiplier:.1f}x",
      }

    meta = {
        "markov_p": markov_p,
        "seq1_n": seq1_n,
        "seq2_n": seq2_n,
        "top_n": top_n,
        "pay_n": pay_n,
        "pay_bucket": pay_bucket,
        "used_dealer": p_dealer is not None,
        "used_sklearn": p_sk is not None,
        "used_seq2": p_seq2 is not None,
        "dealer": latest_dealer,
        "top_slot": latest_top,
        "n": len(sectors),
        "w_seq1": w_seq1,
    }
    res_df = pd.DataFrame(metrics).T.sort_values(by="Score", ascending=False)
    return res_df, meta

  def analyze_patterns(self):
    df = self.get_ordered_df()
    if df.empty:
      return

    latest = df.iloc[0]
    latest_sector = latest["wheelSector"]

    last_10 = df["wheelSector"].head(10).tolist()
    last_10_str = " ➔ ".join([f"[{s}]" for s in reversed(last_10)])

    recent_100 = df.head(100)
    top_slot_matches = (
        recent_100["isTopSlotMatched"].sum() if not recent_100.empty else 0
    )

    potency_df, meta = self.calculate_potency(df)
    if potency_df is None or potency_df.empty:
      return
    top_3 = potency_df.index[:3].tolist()

    clear_console()

    print("=" * 85)
    print(f"📁 ELEMENTOS ACUMULADOS NO JSON: {len(self.history_dict)} rodadas")
    print(f"📜 ÚLTIMAS 10 RODADAS: {last_10_str}")
    print("=" * 85)
    print(
        f"🎯 ÚLTIMO RESULTADO: [{latest_sector}] | Dealer: {latest['dealerName']}"
        f" | Payout: €{latest['totalAmount']:,}"
    )
    print(
        f"🏆 Maior Prêmio: {latest['topWinnerName']} com"
        f" €{latest['topWinnerVal']:,.2f} | Ganhadores: {latest['totalWinners']}"
    )
    print(
        f"🎰 TOP SLOT: [{latest['topSlotSector']}]"
        f" ({latest['topSlotMultiplier']}x) | Match nos 100g:"
        f" {top_slot_matches}%"
    )

    markov_note = f"chi² p={meta['markov_p']:.3f} (peso seq={meta['w_seq1']*100:.0f}%)"
    extras = [f"n-gram1={meta['seq1_n']:.0f}"]
    if meta["used_seq2"]:
      extras.append(f"n-gram2={meta['seq2_n']:.0f}")
    extras.append(f"top-slot n={meta['top_n']:.0f}")
    extras.append(f"payout {meta['pay_bucket']} n={meta['pay_n']:.0f}")
    if meta["used_dealer"]:
      extras.append(f"dealer [{meta['dealer']}]")
    if meta["used_sklearn"]:
      extras.append("sklearn estado completo")
    extra_str = " | " + ", ".join(extras)
    print(f"📐 Modelo: seq+topslot+payout+atraso | {markov_note}{extra_str}")
    print("=" * 85)
    print(potency_df.to_string())

    print("\n🏆 TOP 3 SUGESTÕES (melhor contexto vs roda, não só o mais comum):")
    print(f"👉 1º: [{top_3[0]}]  |  2º: [{top_3[1]}]  |  3º: [{top_3[2]}]")

    if latest["isTopSlotMatched"]:
      print(
          f"\n🔥 [ALERTA TOP SLOT MATCH] O Top Slot ativou no resultado:"
          f" [{latest_sector}] com multiplicador {latest['topSlotMultiplier']}x!"
      )
    print("=" * 85 + "\n")
    print("⚡ Monitorando em tempo real... aguarde novas rodadas.")

  async def poll_realtime(self, client: httpx.AsyncClient):
    params = {
        "page": 0,
        "size": 10,
        "sort": "data.settledAt,desc",
        "duration": 6,
        "wheelResults": "Pachinko,CashHunt,CrazyBonus,CoinFlip,1,2,5,10",
        "isTopSlotMatched": "true,false",
        "tableId": "CrazyTime0000001",
    }

    while True:
      try:
        response = await client.get(BASE_URL, params=params, timeout=3.0)
        items = response.json()

        if isinstance(items, list):
          new_entries = 0
          for item in reversed(items):
            parsed = self.parse_item(item)
            if parsed["id"] and parsed["id"] not in self.history_dict:
              self.history_dict[parsed["id"]] = parsed
              new_entries += 1

          if new_entries > 0:
            self.save_to_json()
            self.analyze_patterns()

      except Exception:
        pass

      await asyncio.sleep(1.5)


async def main():
  clear_console()
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }
  async with httpx.AsyncClient(headers=headers) as client:
    analyzer = CrazyTimeAnalyzer()
    await analyzer.load_initial_history(client, pages_count=2)
    analyzer.analyze_patterns()
    await analyzer.poll_realtime(client)


if __name__ == "__main__":
  asyncio.run(main())
