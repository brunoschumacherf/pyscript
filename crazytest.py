import asyncio
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import httpx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# ----------------------------------------------------------------------
# 1. CONSTANTES E TEORIA DA ROLETA
# ----------------------------------------------------------------------
BASE_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
JSON_FILE = Path("crazytime_history.json")

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

TOTAL_SLOTS = sum(SECTOR_SLOTS.values())  # 54
SECTORS = list(SECTOR_SLOTS.keys())
THEORETICAL_PROBS = {k: v / TOTAL_SLOTS for k, v in SECTOR_SLOTS.items()}

SECTOR_TO_INT = {sec: i for i, sec in enumerate(SECTORS)}
INT_TO_SECTOR = {i: sec for i, sec in enumerate(SECTORS)}


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


# ----------------------------------------------------------------------
# 2. ANÁLISES ESTATÍSTICAS E DESCRITIVAS
# ----------------------------------------------------------------------
class DescriptiveAnalyzer:
    def __init__(self, sequence: List[str]):
        self.sequence = sequence
        self.n = len(sequence)

    def summary_table(self) -> pd.DataFrame:
        counts = pd.Series(self.sequence).value_counts().reindex(SECTORS, fill_value=0)
        df = pd.DataFrame({
            'Frequencia_Obs': counts,
            'Perc_Obs': counts / self.n if self.n > 0 else 0,
            'Prob_Teorica': [THEORETICAL_PROBS[s] for s in SECTORS],
            'Esperado_Teorico': [THEORETICAL_PROBS[s] * self.n for s in SECTORS]
        })
        df['Desvio_Abs'] = df['Frequencia_Obs'] - df['Esperado_Teorico']
        
        df['Z_Score'] = df.apply(
            lambda row: (row['Frequencia_Obs'] - row['Esperado_Teorico']) / 
                        math.sqrt(self.n * row['Prob_Teorica'] * (1 - row['Prob_Teorica']))
                        if self.n > 0 else 0.0, axis=1
        )
        return df


class SequentialAnalyzer:
    def __init__(self, sequence: List[str]):
        self.sequence = sequence

    def transition_matrix(self) -> pd.DataFrame:
        if len(self.sequence) < 2:
            return pd.DataFrame(0.0, index=SECTORS, columns=SECTORS)
        df_seq = pd.DataFrame({'curr': self.sequence[:-1], 'next': self.sequence[1:]})
        ct = pd.crosstab(df_seq['curr'], df_seq['next'], normalize='index')
        return ct.reindex(index=SECTORS, columns=SECTORS, fill_value=0.0)

    def delay_analysis(self) -> Tuple[Dict[str, int], pd.DataFrame]:
        last_seen = {s: -1 for s in SECTORS}
        intervals = {s: [] for s in SECTORS}

        for idx, sec in enumerate(self.sequence):
            if last_seen[sec] != -1:
                intervals[sec].append(idx - last_seen[sec])
            last_seen[sec] = idx

        n_tot = len(self.sequence)
        current_delays = {s: (n_tot - 1 - last_seen[s]) if last_seen[s] != -1 else n_tot
                          for s in SECTORS}

        stats_list = []
        for s in SECTORS:
            arr = intervals[s]
            if len(arr) > 0:
                stats_list.append({
                    'Setor': s,
                    'Atraso_Atual': current_delays[s],
                    'Intervalo_Medio': round(np.mean(arr), 1),
                    'Max_Atraso': np.max(arr)
                })
            else:
                stats_list.append({
                    'Setor': s, 'Atraso_Atual': current_delays[s],
                    'Intervalo_Medio': np.nan, 'Max_Atraso': np.nan
                })
        return current_delays, pd.DataFrame(stats_list).set_index('Setor')


class StatisticalTests:
    def __init__(self, sequence: List[str]):
        self.sequence = sequence
        self.n = len(sequence)

    def run_all_tests(self) -> Dict[str, Any]:
        if self.n < 10:
            return {'Chi2_PValue': 1.0, 'Indep_PValue': 1.0, 'Runs_Test_PValue': 1.0}

        counts = pd.Series(self.sequence).value_counts().reindex(SECTORS, fill_value=0)
        expected = [THEORETICAL_PROBS[s] * self.n for s in SECTORS]

        chi2_stat, chi2_p = stats.chisquare(f_obs=counts, f_exp=expected)

        df_seq = pd.DataFrame({'curr': self.sequence[:-1], 'next': self.sequence[1:]})
        ct = pd.crosstab(df_seq['curr'], df_seq['next'])
        if ct.size > 0:
            chi2_ind, p_ind, _, _ = stats.chi2_contingency(ct)
        else:
            p_ind = 1.0

        binary_seq = np.array([1 if s in ["1", "2", "5", "10"] else 0 for s in self.sequence])
        runs_p = self._runs_test(binary_seq)

        return {
            'Chi2_PValue': chi2_p,
            'Indep_PValue': p_ind,
            'Runs_Test_PValue': runs_p
        }

    def _runs_test(self, binary_arr: np.ndarray) -> float:
        n = len(binary_arr)
        if n < 2: return 1.0
        n1 = np.sum(binary_arr == 1)
        n0 = np.sum(binary_arr == 0)
        if n1 == 0 or n0 == 0: return 1.0
        runs = np.diff(binary_arr) != 0
        R = np.sum(runs) + 1
        mean_R = 1 + (2 * n1 * n0) / n
        var_R = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
        if var_R <= 0: return 1.0
        Z = (R - mean_R) / np.sqrt(var_R)
        return float(2 * (1 - stats.norm.cdf(abs(Z))))


# ----------------------------------------------------------------------
# 3. MODELAGEM, BACKTESTING E ENSEMBLE (CORRIGIDO)
# ----------------------------------------------------------------------
class ModelEvaluator:
    def __init__(self, sequence: List[str]):
        self.sequence = sequence
        self.encoded_seq = [SECTOR_TO_INT[s] for s in sequence if s in SECTOR_TO_INT]
        self.n = len(self.encoded_seq)

    def run_walk_forward_backtest(self, min_train: int = 30, lookback: int = 3) -> Dict[str, Any]:
        p_teor_default = np.array([THEORETICAL_PROBS[INT_TO_SECTOR[i]] for i in range(8)])

        if self.n <= min_train + 5:
            return {
                "metrics": pd.DataFrame(),
                "next_predictions": {
                    'Theoretical': p_teor_default,
                    'Historical': p_teor_default,
                    'Recency_Weighted': p_teor_default,
                    'Markov_Chain': p_teor_default,
                    'Logistic_Reg': p_teor_default,
                    'Random_Forest': p_teor_default,
                    'Ensemble_Optimized': p_teor_default
                }
            }

        models_names = ['Theoretical', 'Historical', 'Recency_Weighted', 'Markov_Chain', 'Logistic_Reg', 'Random_Forest']
        preds_probs = {m: [] for m in models_names}
        actuals = []
        sector_indices = np.array(self.encoded_seq)

        for t in range(min_train, self.n):
            train_seq = self.sequence[:t]
            actual_next = self.encoded_seq[t]
            actuals.append(actual_next)

            # 1. Baseline Teórico
            preds_probs['Theoretical'].append(p_teor_default)

            # 2. Histórico
            hist_counts = pd.Series(train_seq).value_counts().reindex(SECTORS, fill_value=0)
            preds_probs['Historical'].append((hist_counts / len(train_seq)).values)

            # 3. Recência Exponencial
            weights = np.exp(np.linspace(-2, 0, len(train_seq)))
            rec_counts = pd.Series(train_seq).groupby(train_seq).apply(lambda x: weights[x.index].sum()).reindex(SECTORS, fill_value=0.0)
            preds_probs['Recency_Weighted'].append((rec_counts / rec_counts.sum()).values)

            # 4. Markov
            curr_state = SECTOR_TO_INT[train_seq[-1]]
            sub_seq_enc = sector_indices[:t]
            transitions = np.zeros((8, 8))
            for i in range(len(sub_seq_enc) - 1):
                transitions[sub_seq_enc[i], sub_seq_enc[i+1]] += 1
            row_sum = transitions[curr_state, :].sum()
            p_markov = transitions[curr_state, :] / row_sum if row_sum > 0 else p_teor_default
            preds_probs['Markov_Chain'].append(p_markov)

            # Datasets para ML
            X_tr, y_tr = [], []
            for i in range(lookback, t):
                X_tr.append(sector_indices[i-lookback:i])
                y_tr.append(sector_indices[i])
            X_curr = np.array([sector_indices[t-lookback:t]])

            # 5. Regressão Logística (Corrigido para evitar TypeError no scikit-learn recente)
            if len(set(y_tr)) > 1:
                clf_lr = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
                p_lr_raw = clf_lr.predict_proba(X_curr)[0]
                p_lr = np.zeros(8)
                for idx, cls in enumerate(clf_lr.classes_): p_lr[cls] = p_lr_raw[idx]
            else: p_lr = p_teor_default
            preds_probs['Logistic_Reg'].append(p_lr)

            # 6. Random Forest
            if len(set(y_tr)) > 1:
                clf_rf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42).fit(X_tr, y_tr)
                p_rf_raw = clf_rf.predict_proba(X_curr)[0]
                p_rf = np.zeros(8)
                for idx, cls in enumerate(clf_rf.classes_): p_rf[cls] = p_rf_raw[idx]
            else: p_rf = p_teor_default
            preds_probs['Random_Forest'].append(p_rf)

        # Cálculo de Métricas no Backtest
        actuals_arr = np.array(actuals)
        metrics = {}
        for m in models_names:
            probs = np.clip(np.array(preds_probs[m]), 1e-15, 1 - 1e-15)
            probs = probs / probs.sum(axis=1, keepdims=True)

            acc = accuracy_score(actuals_arr, np.argmax(probs, axis=1))
            top3 = np.mean([actuals_arr[i] in np.argsort(probs[i])[-3:] for i in range(len(actuals_arr))])
            ll = log_loss(actuals_arr, probs, labels=list(range(8)))

            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(len(actuals_arr)), actuals_arr] = 1
            brier = np.mean(np.sum((probs - y_onehot)**2, axis=1))

            metrics[m] = {'Accuracy': acc, 'Top3_Accuracy': top3, 'Log_Loss': ll, 'Brier_Score': brier}

        # Pesos Ensemble
        inv_losses = {m: 1.0 / metrics[m]['Log_Loss'] for m in models_names}
        tot_inv = sum(inv_losses.values())
        ensemble_weights = {m: inv_losses[m] / tot_inv for m in models_names}

        # Previsão da Próxima Rodada (instante t = N)
        next_probs = self._predict_next_step(models_names, ensemble_weights, lookback)

        return {
            "metrics": pd.DataFrame(metrics).T,
            "weights": ensemble_weights,
            "next_predictions": next_probs
        }

    def _predict_next_step(self, models_names: List[str], weights: Dict[str, float], lookback: int) -> Dict[str, np.ndarray]:
        predictions = {}
        p_teor = np.array([THEORETICAL_PROBS[INT_TO_SECTOR[i]] for i in range(8)])
        predictions['Theoretical'] = p_teor

        hist_counts = pd.Series(self.sequence).value_counts().reindex(SECTORS, fill_value=0)
        predictions['Historical'] = (hist_counts / self.n).values

        w_vec = np.exp(np.linspace(-2, 0, self.n))
        rec_counts = pd.Series(self.sequence).groupby(self.sequence).apply(lambda x: w_vec[x.index].sum()).reindex(SECTORS, fill_value=0.0)
        predictions['Recency_Weighted'] = (rec_counts / rec_counts.sum()).values

        curr_state = SECTOR_TO_INT[self.sequence[-1]]
        transitions = np.zeros((8, 8))
        for i in range(len(self.encoded_seq) - 1):
            transitions[self.encoded_seq[i], self.encoded_seq[i+1]] += 1
        row_s = transitions[curr_state, :].sum()
        predictions['Markov_Chain'] = transitions[curr_state, :] / row_s if row_s > 0 else p_teor

        X_tr, y_tr = [], []
        for i in range(lookback, self.n):
            X_tr.append(self.encoded_seq[i-lookback:i])
            y_tr.append(self.encoded_seq[i])
        X_curr = np.array([self.encoded_seq[-lookback:]])

        if len(set(y_tr)) > 1:
            clf_lr = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
            p_lr_raw = clf_lr.predict_proba(X_curr)[0]
            p_lr = np.zeros(8)
            for idx, cls in enumerate(clf_lr.classes_): p_lr[cls] = p_lr_raw[idx]
        else: p_lr = p_teor
        predictions['Logistic_Reg'] = p_lr

        if len(set(y_tr)) > 1:
            clf_rf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42).fit(X_tr, y_tr)
            p_rf_raw = clf_rf.predict_proba(X_curr)[0]
            p_rf = np.zeros(8)
            for idx, cls in enumerate(clf_rf.classes_): p_rf[cls] = p_rf_raw[idx]
        else: p_rf = p_teor
        predictions['Random_Forest'] = p_rf

        ens_p = np.zeros(8)
        for m in models_names:
            ens_p += weights[m] * predictions[m]
        predictions['Ensemble_Optimized'] = ens_p / ens_p.sum()

        return predictions


# ----------------------------------------------------------------------
# 4. CAPTURA API, PERSISTÊNCIA EM JSON E MONITORAMENTO EM TEMPO REAL
# ----------------------------------------------------------------------
class CrazyTimeRealTimeEngine:
    def __init__(self):
        self.history_dict = {}
        self.load_from_json()

    def load_from_json(self):
        """Carrega dados locais acumulados no arquivo JSON."""
        if JSON_FILE.exists():
            try:
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        if "id" in item:
                            self.history_dict[item["id"]] = item
            except Exception as e:
                print(f"⚠️ Erro ao ler JSON: {e}")

    def save_to_json(self):
        """Persiste o histórico acumulado no JSON ordenado do mais recente ao mais antigo."""
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
            "isTopSlotMatched": result.get("isTopSlotMatchedToWheelResult", False),
            "totalWinners": item.get("totalWinners", 0),
            "totalAmount": item.get("totalAmount", 0),
            "numOfParticipants": data.get("numOfParticipants", 0),
            "topWinnerName": top_winner_name,
            "topWinnerVal": top_winner_val,
            "maxMultiplier": result.get("maxMultiplier", 1),
            "bonusType": bonus_data.get("type"),
            "bonusMultiplier": bonus_data.get("bonusMultiplier", {}).get("value"),
        }

    async def sync_initial_history(self, client: httpx.AsyncClient, pages_count: int = 3):
        print("📥 Sincronizando e acumulando rodadas recentes da API no JSON...")
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
        sorted_items = sorted(
            list(self.history_dict.values()),
            key=lambda x: x.get("settledAt", ""),
            reverse=True,
        )
        return pd.DataFrame(sorted_items)

    def run_full_analysis_and_print(self):
        df = self.get_ordered_df()
        if df.empty:
            return

        # Sequência CRONOLÓGICA [Antigo ---> Recente]
        chrono_sequence = df["wheelSector"].tolist()[::-1]
        n_rounds = len(chrono_sequence)

        # 1. Análise Descritiva e Delays
        desc = DescriptiveAnalyzer(chrono_sequence)
        df_summary = desc.summary_table()

        seq_anal = SequentialAnalyzer(chrono_sequence)
        df_trans = seq_anal.transition_matrix()
        _, df_delays = seq_anal.delay_analysis()

        # 2. Testes de Hipótese
        tests = StatisticalTests(chrono_sequence)
        test_res = tests.run_all_tests()

        # 3. Modelos de Machine Learning e Backtest
        evaluator = ModelEvaluator(chrono_sequence)
        backtest_res = evaluator.run_walk_forward_backtest(min_train=max(20, int(n_rounds * 0.2)))

        df_metrics = backtest_res['metrics']
        next_probs = backtest_res['next_predictions']['Ensemble_Optimized']
        top3_idx = np.argsort(next_probs)[-3:][::-1]

        # Diagnóstico de Sinais
        has_temporal_bias = test_res['Indep_PValue'] < 0.05 or test_res['Runs_Test_PValue'] < 0.05
        has_freq_bias = test_res['Chi2_PValue'] < 0.05
        has_signal = has_temporal_bias or has_freq_bias

        best_model = df_metrics['Log_Loss'].idxmin() if not df_metrics.empty else 'Theoretical'

        # Cabeçalhos do Console
        latest = df.iloc[0]
        last_10 = df["wheelSector"].head(10).tolist()
        last_10_str = " ➔ ".join([f"[{s}]" for s in reversed(last_10)])

        clear_console()
        print("=" * 75)
        print("ANÁLISE PREDITIVA E MONITORAMENTO — CRAZY TIME")
        print("=" * 75)
        print(f"📁 ELEMENTOS ACUMULADOS NO JSON: {len(self.history_dict)} rodadas")
        print(f"📜 ÚLTIMAS 10 RODADAS: {last_10_str}")
        print("=" * 75)
        print(f"🎯 ÚLTIMO RESULTADO: [{latest['wheelSector']}] | Dealer: {latest['dealerName']} | Payout: €{latest['totalAmount']:,}")
        print(f"🎰 TOP SLOT: [{latest['topSlotSector']}] ({latest['topSlotMultiplier']}x)")
        print("=" * 75)

        print("\n🏆 TOP 3 — PRÓXIMA SAÍDA (ENSEMBLE DE ML)")
        print("-" * 75)
        medals = ['🥇', '🥈', '🥉']
        for idx, i in enumerate(top3_idx):
            sec = INT_TO_SECTOR[i]
            p_est = next_probs[i] * 100
            p_teor = THEORETICAL_PROBS[sec] * 100
            diff = p_est - p_teor
            sign = "+" if diff >= 0 else ""
            print(f"{medals[idx]} {sec:<10} → Estimada: {p_est:5.2f}% | Teórica: {p_teor:5.2f}% | Dif: {sign}{diff:.2f} pp")

        print("\n" + "=" * 75)
        print("DIAGNÓSTICO ESTATÍSTICO")
        print("=" * 75)
        print(f"Sinal estatístico encontrado: {'SIM' if has_signal else 'NÃO (Padrão Aleatório)'}")
        print(f"Dependência temporal detectada: {'SIM' if has_temporal_bias else 'NÃO'}")
        print(f"Modelo com melhor desempenho no Backtest: {best_model}")

        print("\n" + "=" * 75)
        print("AVALIAÇÃO DE ATRASOS (DELAYS E INTERVALOS MÉDIOS)")
        print("=" * 75)
        print(df_delays[['Atraso_Atual', 'Intervalo_Medio', 'Max_Atraso']].to_string())

        print("\n" + "=" * 75)
        print("⚡ Monitorando em tempo real... O arquivo JSON é atualizado continuamente.")

    async def start_realtime_polling(self, client: httpx.AsyncClient):
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
                        self.run_full_analysis_and_print()

            except Exception:
                pass

            await asyncio.sleep(1.5)


# ----------------------------------------------------------------------
# 5. PONTO DE ENTRADA DO SCRIPT
# ----------------------------------------------------------------------
async def main():
    clear_console()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    async with httpx.AsyncClient(headers=headers) as client:
        engine = CrazyTimeRealTimeEngine()
        await engine.sync_initial_history(client, pages_count=3)
        engine.run_full_analysis_and_print()
        await engine.start_realtime_polling(client)


if __name__ == "__main__":
    asyncio.run(main())