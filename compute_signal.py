#!/usr/bin/env python3
"""
AI オルカン研究所 - 積立タイミング判定スクリプト

やっていること（概要）:
  1. 投資信託協会（一般社団法人、業界公式団体）が無料公開している
     基準価額CSV（設定来の全履歴）を取得する。
     eMAXIS Slim 全世界株式（オール・カントリー）、ISIN: JP90C000H1T1。
  2. 日足しかないため、FX/BTC/Gold版のような「5分・15分・1時間足」ではなく、
     同じ日次系列に対して短期(20日)・中期(60日)・長期(120日)の3つの
     lookback幅で線形回帰チャネルを計算し、方向一致を判定する。
  3. 3つとも「上方向」で一致した時だけBUY候補とし、さらに短期(10日)チャネルで
     直近の押し目からの反発（REVERT_WINDOW=5日以内、2%以上の戻り）を検出できた
     場合のみ、実際のBUYシグナルとして確定する（それ以外はWAIT）。
  4. 投資信託は空売りができない商品のため、SELLシグナルは存在しない
    （バックテストでBUY側のみ明確な優位性(PF4.59)が出たことを踏まえた設計。
     SELL側はPF0.53と明確に機能しなかった）。
  5. 結果を signal.json として書き出す。

データソースの注意:
  投資信託協会のCSVは、その日の基準価額が確定・公表されるまで前営業日までの
  データしか含まれないことがある（海外資産を含むファンドは基準価額の確定に
  時間がかかるため）。1日1回、日本時間の夜〜翌朝に実行すれば十分。
"""

import csv
import io
import json
import os
import statistics
import sys
import urllib.request
from datetime import datetime, timezone

NAV_CSV_URL = (
    "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download"
    "?isinCd=JP90C000H1T1&associFundCd=0331418A"
)

SHORT_LB = 20
MEDIUM_LB = 60
LONG_LB = 120
TRIGGER_LB = 10
REVERT_WINDOW = 5
EDGE_THRESHOLD = 1.3
REVERT_MIN_PCT = 2.0   # トリガーチャネルの谷からこれ以上戻ったら反発とみなす(%)
TARGET_BUFFER_PCT = 1.5  # 谷からの下振れ許容幅(%)。SLではなく「この水準を割ったら判定取り消し」の目安


def fetch_nav_history():
    """
    投資信託協会のCSV（Shift-JIS）を取得し、[(date_str, nav), ...] を古い順で返す。
    """
    req = urllib.request.Request(NAV_CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as res:
        raw = res.read()
    text = raw.decode("shift_jis", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = []
    for i, row in enumerate(reader):
        if i == 0 or not row or not row[0]:
            continue
        d = row[0].replace("年", "-").replace("月", "-").replace("日", "")
        try:
            nav = float(row[1])
        except (ValueError, IndexError):
            continue
        rows.append((d, nav))
    if not rows:
        raise RuntimeError("基準価額CSVからデータを取得できませんでした")
    return rows


def linear_regression_channel(closes, lookback):
    series = closes[-lookback:] if len(closes) > lookback else closes[:]
    n = len(series)
    if n < 5:
        return None
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(series) / n
    num = sum((xs[i] - x_mean) * (series[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x for x in xs]
    residuals = [series[i] - fitted[i] for i in range(n)]
    sigma = statistics.pstdev(residuals) if n > 1 else 0.0
    sigma = sigma if sigma > 1e-9 else 1e-9
    mid = fitted[-1]
    upper = mid + 2 * sigma
    lower = mid - 2 * sigma
    latest = series[-1]
    position = (latest - mid) / sigma
    return {"mid": mid, "upper": upper, "lower": lower, "sigma": sigma, "position": position, "slope": slope}


def momentum_direction(ch):
    if ch is None:
        return "FLAT"
    pos = ch["position"]
    if pos >= EDGE_THRESHOLD:
        return "UP"
    if pos <= -EDGE_THRESHOLD:
        return "DOWN"
    return "FLAT"


def detect_reversal_setup(recent_closes, ch):
    """押し目からの反発（BUY方向のみ判定。空売り不可のためSELL側は存在しない）。"""
    if len(recent_closes) < REVERT_WINDOW or ch is None:
        return None
    closes = recent_closes[-REVERT_WINDOW:]
    latest = closes[-1]
    sigma = ch["sigma"]
    mid = ch["mid"]
    trough_idx = min(range(len(closes)), key=lambda i: closes[i])
    trough = closes[trough_idx]
    if trough_idx == len(closes) - 1:
        return None  # 最新日がまだ谷=反発が始まっていない
    if (trough - mid) / sigma > -EDGE_THRESHOLD:
        return None  # チャネル下限際まで達していない
    if (latest - trough) / trough * 100 < REVERT_MIN_PCT:
        return None  # 戻り幅が不十分
    return trough


def build_market_context(bias, candidate, latest_nav, day_change_pct):
    change_txt = f"{day_change_pct:+.2f}%"
    if bias == "BUY":
        stance = "短期・中期・長期のいずれも上向きが揃う中、直近の押し目から反発したタイミング"
        outlook = "積立額を少し増やす、または新規に買い増しを検討しやすい局面。"
    elif bias == "CAUTION":
        stance = "短期・中期・長期のいずれも下向きが揃っている局面"
        outlook = "投資信託は空売りができないため利益を狙う手段はありませんが、無理に買い増さず、通常額のままか一時的な減額を検討してもよい局面。"
    elif candidate == "BUY":
        stance = "短期・中期・長期の方向は揃っているが、直近の押し目からの反発はまだ確認できていない"
        outlook = "方向感は出ているため、押し目からの反発を待ちたい局面。"
    else:
        stance = "短期・中期・長期の方向が揃っておらず、方向感に乏しい局面"
        outlook = "通常のペースでの積立を継続するのが無難な局面。"
    return (
        f"基準価額は現在{latest_nav:,.0f}円付近（前営業日比{change_txt}）。{stance}。"
        f"{outlook}"
        "※これは長期の積立投資向けの参考情報であり、短期売買を推奨するものではありません。"
    )


def build_signal(out_path=None):
    now = datetime.now(timezone.utc)
    rows = fetch_nav_history()
    dates = [r[0] for r in rows]
    closes = [r[1] for r in rows]

    latest_nav = closes[-1]
    latest_date = dates[-1]
    day_change_pct = 0.0
    if len(closes) >= 2 and closes[-2]:
        day_change_pct = (closes[-1] - closes[-2]) / closes[-2] * 100

    ch_short = linear_regression_channel(closes, SHORT_LB)
    ch_medium = linear_regression_channel(closes, MEDIUM_LB)
    ch_long = linear_regression_channel(closes, LONG_LB)
    ch_trigger = linear_regression_channel(closes, TRIGGER_LB)

    d_short = momentum_direction(ch_short)
    d_medium = momentum_direction(ch_medium)
    d_long = momentum_direction(ch_long)

    if d_short == "UP" and d_medium == "UP" and d_long == "UP":
        candidate = "BUY"
    elif d_short == "DOWN" and d_medium == "DOWN" and d_long == "DOWN":
        candidate = "CAUTION"
    else:
        candidate = None

    # 反発トリガーはBUY判定にのみ使う。投資信託は空売りできないため、CAUTION（下向き一致）は
    # 「利益を狙って売る」判定ではなく「無理に買い増さない方がいいかもしれない」という
    # リスク回避目的の注意喚起であり、反発確認なしで下向き一致のみで判定する。
    extreme = detect_reversal_setup(closes, ch_trigger) if candidate == "BUY" else None
    if candidate == "BUY" and extreme is not None:
        bias = "BUY"
    elif candidate == "CAUTION":
        bias = "CAUTION"
    else:
        bias = "WAIT"

    if bias == "BUY":
        avg_abs_pos = sum(abs(ch["position"]) for ch in (ch_short, ch_medium, ch_long)) / 3
        confidence = 50 + 30 + min(avg_abs_pos, 3.0) * 5
        confidence = max(50, min(95, round(confidence)))
        stars = max(1, min(5, round(confidence / 20)))
        target_low = round(extreme * (1 - TARGET_BUFFER_PCT / 100), 0)
        lead = "押し目買いのチャンス ― 短期・中期・長期が揃って上向き、直近の押し目から反発"
    elif bias == "CAUTION":
        avg_abs_pos = sum(abs(ch["position"]) for ch in (ch_short, ch_medium, ch_long)) / 3
        confidence = 50 + 20 + min(avg_abs_pos, 3.0) * 3
        confidence = max(50, min(85, round(confidence)))
        stars = max(1, min(3, round(confidence / 25)))
        target_low = None
        lead = "一時減額を検討 ― 短期・中期・長期が揃って下向き。無理に買い増さず、通常額または減額も選択肢"
    else:
        confidence = 50
        stars = 2
        target_low = None
        if candidate == "BUY":
            lead = "様子見 ― 方向は揃っているが、押し目からの反発待ち"
        else:
            lead = "様子見 ― 短期・中期・長期の方向が一致していない"

    market_context = build_market_context(bias, candidate, latest_nav, day_change_pct)

    timeframes = [
        {"key": "short", "label": f"短期（{SHORT_LB}日）", "channel": ch_short, "momentum": d_short},
        {"key": "medium", "label": f"中期（{MEDIUM_LB}日）", "channel": ch_medium, "momentum": d_medium},
        {"key": "long", "label": f"長期（{LONG_LB}日）", "channel": ch_long, "momentum": d_long},
    ]
    momentum_ja = {"UP": "上方向", "DOWN": "下方向", "FLAT": "中央"}
    timeframes_note = " / ".join(f"{tf['label']}:{momentum_ja[tf['momentum']]}" for tf in timeframes)

    result = {
        "generated_at_utc": now.isoformat(),
        "fund_name": "eMAXIS Slim 全世界株式（オール・カントリー）",
        "fund_code": "0331418A",
        "latest_nav": round(latest_nav, 0),
        "latest_date": latest_date,
        "day_change_pct": round(day_change_pct, 2),
        "signal": {
            "bias": bias,
            "bias_label": {"BUY": "押し目買いのチャンス", "CAUTION": "一時減額を検討", "WAIT": "通常運用でOK"}[bias],
            "stars": stars,
            "confidence": confidence,
            "confidence_breakdown": {"timeframes_note": timeframes_note},
        },
        "guidance": {
            "lead": lead,
            "reference_low": target_low,
        },
        "regression_channels": [
            {
                "key": tf["key"],
                "label": tf["label"],
                "momentum": tf["momentum"],
                "position_sigma": round(tf["channel"]["position"], 2) if tf["channel"] else None,
                "mid": round(tf["channel"]["mid"], 0) if tf["channel"] else None,
                "upper": round(tf["channel"]["upper"], 0) if tf["channel"] else None,
                "lower": round(tf["channel"]["lower"], 0) if tf["channel"] else None,
            }
            for tf in timeframes
        ],
        "market_context": market_context,
        "disclaimer": (
            "本データはルールベースの参考情報であり、投資成果を保証するものではありません。"
            "投資信託は空売りができないため、本シグナルは「押し目買いのチャンス」「一時減額を検討」"
            "「通常運用でOK」の3段階のみを判定し、下落方向で利益を狙うシグナルは存在しません。"
            "「押し目買いのチャンス」の過去のバックテスト（2018年設定来）では勝率83.3%・"
            "プロフィットファクター4.59でしたが、対象は12件と少なく、将来の成果を保証するものではありません。"
            "「一時減額を検討」は反発確認を伴わない下向き一致のみでの判定のため、バックテストによる優位性の検証はしていません。"
        ),
    }
    return result


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "signal.json")
    out_path = os.path.abspath(out_path)
    try:
        signal = build_signal()
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] シグナル計算に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)
    print(f"書き出し完了: {out_path}")
    print(json.dumps(signal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
