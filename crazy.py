import asyncio
import json
import os
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
from scipy.stats import chisquare

BASE_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
JSON_FILE = Path("crazytime_history.json")

# Estrutura teórica do Crazy Time (Total = 54 fatias)
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

BONUS_SECTORS = {"CoinFlip", "Pachinko", "CashHunt", "CrazyBonus"}
SECTORS_LIST = list(SECTOR_SLOTS.keys())
THEORETICAL_PROBS = {k: v / 54.0 for k, v in SECTOR_SLOTS.items()}
EXPECTED_GAPS = {k: 54.0 / v for k, v in SECTOR_SLOTS.items()}

BASE_PAYOUTS = {
    "1": 1 + 1,
    "2": 2 + 1,
    "5": 5 + 1,
    "10": 10 + 1,
    "CoinFlip": 10,
    "Pachinko": 18,
    "CashHunt": 26,
    "CrazyBonus": 40,
}


def clear_console():
  os.system("cls" if os.name == "nt" else "clear")


class CrazyTimeUnifiedAnalyzer:

  def __init__(self):
    self.history_dict = {}
    self.load_from_json()

  def load_from_json(self):
    """Carrega o histórico completo salvo no arquivo JSON local."""
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
    """Persiste o histórico ordenado por data no JSON."""
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
    """Extrai e normaliza todos os campos do payload bruto da API."""
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
        "wheelSector": str(wheel_res.get("wheelSector", "N/A")),
        "resultType": wheel_res.get("type"),
        "topSlotSector": str(top_slot.get("wheelSector", "N/A")),
        "topSlotMultiplier": float(top_slot.get("multiplier", 1.0)),
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
      self, client: httpx.AsyncClient, pages_count: int = 3
  ):
    """Faz a carga inicial de dados paginados da API."""
    print("📥 Sincronizando histórico inicial de rodadas...")
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

    if len(self.history_dict) > initial_count or initial_count == 0:
      self.save_to_json()

  def get_ordered_df(self) -> pd.DataFrame:
    """Retorna DataFrame ordenado do mais recente para o mais antigo."""
    sorted_items = sorted(
        list(self.history_dict.values()),
        key=lambda x: x.get("settledAt", ""),
        reverse=True,
    )
    df = pd.DataFrame(sorted_items)
    if not df.empty and "settledAt" in df.columns:
      df["datetime"] = pd.to_datetime(
          df["settledAt"], format="ISO8601", errors="coerce"
      )
    return df

  def detect_market_phase(self, df: pd.DataFrame, window: int = 12):
    """Identifica rajadas de bônus ou sequências frias nas últimas N rodadas."""
    recent = df.head(window)
    if recent.empty:
      return "NEUTRO", 0

    bonus_count = sum(
        1 for sector in recent["wheelSector"] if sector in BONUS_SECTORS
    )
    if bonus_count >= 3:
      phase = "🔥 FASE QUENTE (Rajada de Bônus)"
    elif bonus_count <= 1:
      phase = "❄️ FASE FRIA (Sequência de Números)"
    else:
      phase = "⚖️ FASE EQUILIBRADA"

    return phase, bonus_count

  def analyze_sequence_transitions(
      self, df: pd.DataFrame, max_seq_length: int = 4
  ):
    """Encontra a maior sequência recente (até 4 termos) com padrão histórico."""
    sectors = df["wheelSector"].values
    if len(sectors) <= 1:
      return {}, 0, []

    for seq_len in range(max_seq_length, 0, -1):
      if len(sectors) <= seq_len:
        continue

      target_sequence = list(reversed(sectors[:seq_len]))
      next_after_sequence = []

      for i in range(len(sectors) - seq_len, 0, -1):
        sub_seq = list(reversed(sectors[i : i + seq_len]))
        if sub_seq == target_sequence:
          next_after_sequence.append(sectors[i - 1])

      if next_after_sequence:
        total = len(next_after_sequence)
        counts = pd.Series(next_after_sequence).value_counts()
        probs = (counts / total).to_dict()
        return probs, seq_len, target_sequence

    return {}, 0, []

  def calculate_potency(self, df: pd.DataFrame):
    """Calcula a matriz de afinidade, atraso relativo e potência do setor."""
    total_rounds = len(df)
    if total_rounds == 0:
      return pd.DataFrame(), 0, []

    transition_probs, matched_seq_len, current_seq = (
        self.analyze_sequence_transitions(df, max_seq_length=4)
    )
    phase_str, _ = self.detect_market_phase(df, window=12)

    metrics = {}
    for sector, expected_gap in EXPECTED_GAPS.items():
      sector_idx = df[df["wheelSector"] == sector].index
      gap = sector_idx[0] if len(sector_idx) > 0 else total_rounds
      delay_ratio = gap / expected_gap
      capped_delay_ratio = min(delay_ratio, 1.5)

      observed_trans_prob = transition_probs.get(sector, 0.0)
      theoretical_prob = THEORETICAL_PROBS[sector]
      affinity_ratio = (
          observed_trans_prob / theoretical_prob if theoretical_prob > 0 else 0
      )

      # Pontuação combinando afinidade de sequência (70%) e atraso (30%)
      potency_score = (affinity_ratio * 70) + (capped_delay_ratio * 30)

      if "QUENTE" in phase_str and sector in BONUS_SECTORS:
        potency_score *= 1.25
      elif "FRIA" in phase_str and sector in BONUS_SECTORS:
        potency_score *= 0.80

      seq_header = (
          f"Pós-{'➔'.join(current_seq)}" if current_seq else "Pós-[Seq]"
      )

      metrics[sector] = {
          "Score Potência": round(potency_score, 2),
          "Atraso (Gaps)": gap,
          "Atraso Rel.": f"{delay_ratio:.1f}x",
          seq_header: f"{(observed_trans_prob * 100):.1f}%",
          "Prob. Teórica": f"{(theoretical_prob * 100):.1f}%",
      }

    res_df = pd.DataFrame(metrics).T.sort_values(
        by="Score Potência", ascending=False
    )
    return res_df, matched_seq_len, current_seq

  def calculate_expected_value(
      self, top_slot_sector: str, top_slot_mult: float
  ) -> pd.DataFrame:
    """Calcula o Valor Esperado (EV) exato para a próxima rodada."""
    ev_results = {}

    for sector, prob in THEORETICAL_PROBS.items():
      base_payout = BASE_PAYOUTS[sector]

      if str(sector) == str(top_slot_sector) and top_slot_mult > 1:
        effective_payout = base_payout * top_slot_mult
      else:
        effective_payout = base_payout

      ev = (prob * effective_payout) - 1.0

      ev_results[sector] = {
          "Prob. Teórica": f"{prob * 100:.2f}%",
          "Payout Base": f"{base_payout}x",
          "Mult. Top Slot": (
              top_slot_mult if str(sector) == str(top_slot_sector) else 1.0
          ),
          "EV (Valor Esperado)": round(ev, 3),
          "Entrada EV+": "🚀 APOSTAR" if ev > 0 else "❌ AGUARDAR",
      }

    df_ev = pd.DataFrame(ev_results).T.sort_values(
        by="EV (Valor Esperado)", ascending=False
    )
    return df_ev

  def perform_chi_square_test(self, df: pd.DataFrame, window: int = 200):
    """Métrica de aderência estatística Qui-Quadrado."""
    recent = df.head(window)
    if len(recent) < window or "wheelSector" not in recent.columns:
      return None

    observed_counts = recent["wheelSector"].value_counts()
    observed = [
        observed_counts.get(sector, 0) for sector in SECTOR_SLOTS.keys()
    ]
    expected = [
        THEORETICAL_PROBS[sector] * len(recent) for sector in SECTOR_SLOTS.keys()
    ]

    _, p_value = chisquare(f_obs=observed, f_exp=expected)
    return p_value

  def analyze_patterns(self):
    """Renderiza a análise unificada no terminal."""
    df = self.get_ordered_df()
    if df.empty:
      return

    latest = df.iloc[0]
    latest_sector = str(latest.get("wheelSector", "N/A"))
    top_sector = str(latest.get("topSlotSector", "N/A"))
    top_mult = float(latest.get("topSlotMultiplier", 1.0))

    last_10 = df["wheelSector"].head(10).tolist()
    last_10_str = " ➔ ".join([f"[{s}]" for s in reversed(last_10)])

    recent_100 = df.head(100)
    top_slot_matches = (
        recent_100["isTopSlotMatched"].sum() if not recent_100.empty else 0
    )

    phase_str, bonus_count_12g = self.detect_market_phase(df, window=12)
    potency_df, matched_seq_len, current_seq = self.calculate_potency(df)
    ev_df = self.calculate_expected_value(top_sector, top_mult)
    p_val = self.perform_chi_square_test(df, window=200)

    clear_console()

    print("=" * 90)
    print(
        f"📂 ELEMENTOS ACUMULADOS NO JSON: {len(self.history_dict)} rodadas"
        " (Sincronizado)"
    )
    print(f"📜 ÚLTIMAS 10 RODADAS: {last_10_str}")
    print(
        f"📊 CICLO DA MESA: {phase_str} ({bonus_count_12g} Bônus nos últimos"
        " 12g)"
    )
    print("=" * 90)
    print(
        f"🎯 ÚLTIMO RESULTADO: [{latest_sector}] | Dealer:"
        f" {latest.get('dealerName', '-')}"
    )
    print(
        f"🏆 Maior Prêmio: {latest.get('topWinnerName', '-')} com"
        f" €{latest.get('topWinnerVal', 0):,.2f} | Payout Total:"
        f" €{latest.get('totalAmount', 0):,}"
    )
    print(
        f"🎰 TOP SLOT: [{top_sector}] ({top_mult}x) | Match nos 100g:"
        f" {top_slot_matches}%"
    )

    if p_val is not None:
      print(f"🔬 Aderência Qui-Quadrado (N=200): p-value = {p_val:.4f}")

    print("=" * 90)
    seq_str = (
        " ➔ ".join([f"[{s}]" for s in current_seq])
        if current_seq
        else "Nenhuma"
    )
    print(
        f"🔍 MATRIZ DE TRANSIÇÃO & POTÊNCIA (Sequência Ativa {matched_seq_len}T:"
        f" {seq_str})"
    )
    print("-" * 90)
    print(potency_df.to_string())

    print("=" * 90)
    print("💰 MATRIZ DE VALOR ESPERADO (EV - PRÓXIMA RODADA):")
    print("-" * 90)
    print(ev_df.to_string())

    best_ev = ev_df[ev_df["EV (Valor Esperado)"] > 0]
    print("\n🏆 RECOMENDAÇÃO TÉCNICA (EV+ MATEMÁTICO):")
    if not best_ev.empty:
      for idx, row in best_ev.iterrows():
        print(
            f"🚀 ENTRADA VANTAJOSA: Apostar em [{idx}] (EV:"
            f" +{row['EV (Valor Esperado)']}, Top Slot: {row['Mult. Top Slot']}x)"
        )
    else:
      print(
          "✋ NENHUMA ENTRADA EV+ NA RODADA ATUAL. (O Top Slot atual não gera"
          " vantagem de EV)."
      )

    if latest.get("isTopSlotMatched", False):
      print(
          f"\n🔥 [MATCH DETECTADO] O Top Slot ativou no resultado:"
          f" [{latest_sector}] com multiplicador {top_mult}x!"
      )

    print("=" * 90)
    print("⚡ Monitorando em tempo real (Polling 1.5s)...")

  async def poll_realtime(self, client: httpx.AsyncClient):
    """Consulta os eventos com parâmetros da mesa em tempo real a cada 1.5s."""
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
    analyzer = CrazyTimeUnifiedAnalyzer()
    await analyzer.load_initial_history(client, pages_count=2)
    analyzer.analyze_patterns()
    await analyzer.poll_realtime(client)


if __name__ == "__main__":
  asyncio.run(main())