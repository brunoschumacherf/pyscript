import asyncio
import json
import os
from pathlib import Path
import httpx
import pandas as pd

BASE_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
JSON_FILE = Path("crazytime_history.json")

# Configurações da Roleta do Crazy Time (Total = 54 fatias)
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
THEORETICAL_PROBS = {k: v / 54.0 for k, v in SECTOR_SLOTS.items()}
EXPECTED_GAPS = {k: 54.0 / v for k, v in SECTOR_SLOTS.items()}


def clear_console():
  os.system("cls" if os.name == "nt" else "clear")


class CrazyTimeAnalyzer:

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
    """Persiste o histórico ordenado por data sem limite de rodadas."""
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

  def detect_market_phase(self, df: pd.DataFrame, window: int = 12):
    """Detecta o ciclo de pagamento atual da mesa (Rajada de Bônus vs.

    Sequência de Números).
    """
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
    """Busca a sequência mais longa (até 4 termos) que possua registros no histórico."""
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
    total_rounds = len(df)
    if total_rounds == 0:
      return None, 0, []

    transition_probs, matched_seq_len, current_seq = (
        self.analyze_sequence_transitions(df, max_seq_length=4)
    )
    phase_str, bonus_count = self.detect_market_phase(df, window=12)

    metrics = {}

    for sector, expected_gap in EXPECTED_GAPS.items():
      sector_idx = df[df["wheelSector"] == sector].index
      gap = sector_idx[0] if len(sector_idx) > 0 else total_rounds
      delay_ratio = gap / expected_gap

      capped_delay_ratio = min(delay_ratio, 1.5)

      sector_df = df[df["wheelSector"] == sector]
      avg_amount = (
          sector_df["totalAmount"].mean() if len(sector_df) > 0 else 0
      )
      avg_multiplier = (
          sector_df["maxMultiplier"].mean() if len(sector_df) > 0 else 1
      )

      observed_trans_prob = transition_probs.get(sector, 0.0)
      theoretical_prob = THEORETICAL_PROBS[sector]

      affinity_ratio = (
          observed_trans_prob / theoretical_prob if theoretical_prob > 0 else 0
      )

      # Pontuação Base (70% Afinidade de Sequência + 30% Atraso)
      potency_score = (affinity_ratio * 70) + (capped_delay_ratio * 30)

      # Ajuste Dinâmico pelo Ciclo da Mesa
      if "QUENTE" in phase_str and sector in BONUS_SECTORS:
        potency_score *= 1.25  # Aumenta peso de Bônus em fase quente
      elif "FRIA" in phase_str and sector in BONUS_SECTORS:
        potency_score *= 0.80  # Diminui peso de Bônus em fase fria

      seq_header = (
          f"Pós-{'➔'.join(current_seq)}" if current_seq else "Pós-[Seq]"
      )

      metrics[sector] = {
          "Score": round(potency_score, 2),
          "Atraso": gap,
          "Relativo": f"{delay_ratio:.1f}x",
          seq_header: f"{(observed_trans_prob * 100):.1f}%",
          "Teórico": f"{(theoretical_prob * 100):.1f}%",
          "Média Pago": f"€{avg_amount:,.0f}",
          "Mult": f"{avg_multiplier:.1f}x",
      }

    res_df = pd.DataFrame(metrics).T.sort_values(by="Score", ascending=False)
    return res_df, matched_seq_len, current_seq

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

    phase_str, bonus_count_12g = self.detect_market_phase(df, window=12)
    potency_df, matched_seq_len, current_seq = self.calculate_potency(df)
    top_3 = potency_df.index[:3].tolist()

    clear_console()

    print("=" * 85)
    print(f"📁 ELEMENTOS ACUMULADOS NO JSON: {len(self.history_dict)} rodadas")
    print(f"📜 ÚLTIMAS 10 RODADAS: {last_10_str}")
    print(
        f"📊 CICLO DA MESA: {phase_str} ({bonus_count_12g} Bônus nos últimos"
        " 12g)"
    )
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
    print("=" * 85)

    seq_str = (
        " ➔ ".join([f"[{s}]" for s in current_seq])
        if current_seq
        else "Nenhuma"
    )
    print(
        f"🔍 MATRIZ DE TRANSIÇÃO (Sequência Ativa de {matched_seq_len} termo(s):"
        f" {seq_str})"
    )
    print("-" * 85)
    print(potency_df.to_string())

    print("\n🏆 TOP 3 SUGESTÕES DE ENTRADA:")
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