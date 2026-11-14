"""
Prop Helper - NBA stats app. Runs on your computer, opens in your browser.

Setup (one time):   pip install flask nba_api pandas requests
Run:                py nba_app.py
"""

import json
import math
import os
import random
import re
import socket
import threading
import time
import traceback
import webbrowser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from flask import Flask, jsonify, request
from nba_api.stats.endpoints import commonallplayers, leaguestandingsv3, playergamelogs, scoreboardv2
from nba_api.stats.static import teams as nba_teams

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # no timezone data: use this computer's clock
    ET = None

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, "nba_cache.json")
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
REFRESH_HOURS = 3
TEAMS = {t["id"]: t for t in nba_teams.get_teams()}

# stat -> (label, lines, getter). A game row is [date, matchup, wl, min, pts, reb, ast, fg3m]
STATS = {
    "points": ("Points", [10, 15, 20, 25, 30, 35], lambda g: g[4]),
    "threes": ("3-pointers", [1, 2, 3, 4, 5], lambda g: g[7]),
    "rebounds": ("Rebounds", [4, 6, 8, 10, 12], lambda g: g[5]),
    "assists": ("Assists", [2, 4, 6, 8, 10, 12], lambda g: g[6]),
    "pra": ("P+R+A", [15, 20, 25, 30, 35, 40], lambda g: g[4] + g[5] + g[6]),
}
LBL = {"points": "points", "threes": "3-pointers", "rebounds": "rebounds", "assists": "assists", "pra": "P+R+A"}
HOT_MIN = {5: 4, 10: 7}  # hits needed in the window
MAX_BASE, MIN_LIFT = 0.65, 0.20
MIN_DELTA = {"points": 3, "rebounds": 1.5, "assists": 1.5, "threes": 0.8, "pra": 4}


def et_now():
    return datetime.now(ET) if ET else datetime.now()


def secs_to_midnight():
    now = et_now()
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((nxt - now).total_seconds()))


def season_str(offset=0):
    now = et_now()
    start = (now.year if now.month >= 10 else now.year - 1) - offset
    return f"{start}-{str(start + 1)[-2:]}"


def iv(v):
    try:
        x = float(v)
        return 0 if x != x else int(round(x))
    except (TypeError, ValueError):
        return 0


def abbr(tid):
    return TEAMS.get(tid, {}).get("abbreviation", "?")


app = Flask(__name__)


# =============================== data (cached in memory + on disk) ===============================
D = {"data": None, "building": False, "error": None, "step": "", "last_try": 0}
DL = threading.Lock()
COLS = ["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "FG3M"]


def step(msg):
    D["step"] = msg
    print("  data:", msg, flush=True)


def pull_season(off):
    kw = {}
    if off == 1:  # last season: only its final stretch (smaller, faster)
        kw["date_from_nullable"] = f"02/01/{int(season_str(1)[:4]) + 1}"
    df = playergamelogs.PlayerGameLogs(season_nullable=season_str(off), season_type_nullable="Regular Season",
                                       timeout=60, **kw).get_data_frames()[0]
    if "WL" not in df.columns:
        df["WL"] = ""
    return df[COLS]


def pull_logs():
    with ThreadPoolExecutor(2) as ex:  # this season and last season at the same time
        cur, prev = ex.submit(pull_season, 0), ex.submit(pull_season, 1)
        frames = [cur.result()]
        try:
            frames.append(prev.result())
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True)


def pull_standings():
    df = leaguestandingsv3.LeagueStandingsV3(season=season_str(), season_type="Regular Season", timeout=30).get_data_frames()[0]
    return [{"id": int(r["TeamID"]), "conf": str(r["Conference"]), "w": iv(r["WINS"]), "l": iv(r["LOSSES"]),
             "home": str(r.get("HOME", "")), "road": str(r.get("ROAD", "")), "l10": str(r.get("L10", "")),
             "streak": str(r.get("strCurrentStreak", ""))} for r in df.to_dict("records")]


def refresh_data():
    step("Downloading rosters, standings and player stats (about 15 seconds)")
    with ThreadPoolExecutor(3) as ex:  # all three downloads run at the same time
        f_ap = ex.submit(lambda: commonallplayers.CommonAllPlayers(is_only_current_season=1, season=season_str(), timeout=30).get_data_frames()[0])
        f_st, f_df = ex.submit(pull_standings), ex.submit(pull_logs)
        ap, df = f_ap.result(), f_df.result()
        try:
            st = f_st.result()
        except Exception as err:
            print("  data: standings failed:", err, flush=True)
            st = (D["data"] or {}).get("standings", [])
    meta = {int(r.PERSON_ID): {"name": r.DISPLAY_FIRST_LAST, "team_id": iv(r.TEAM_ID)}
            for r in ap.itertuples() if iv(r.TEAM_ID) != 0 and iv(r.ROSTERSTATUS) == 1}
    df = df.sort_values("GAME_DATE", ascending=False)
    step("Crunching the numbers")
    by_game = defaultdict(dict)
    for r in df.groupby(["GAME_ID", "TEAM_ID"]).agg(pts=("PTS", "sum"), date=("GAME_DATE", "first")).reset_index().itertuples():
        by_game[r.GAME_ID][int(r.TEAM_ID)] = (iv(r.pts), str(r.date)[:10])
    tg = defaultdict(list)  # team -> [date, margin, win] newest first
    for teams in by_game.values():
        if len(teams) == 2:
            (a, (pa, da)), (b, (pb, db)) = teams.items()
            tg[a].append([da, pa - pb, int(pa > pb)])
            tg[b].append([db, pb - pa, int(pb > pa)])
    for t in tg:
        tg[t] = sorted(tg[t], reverse=True)[:10]
    logs = {}
    for pid, g in df[df["PLAYER_ID"].isin(list(meta))].groupby("PLAYER_ID"):
        logs[int(pid)] = [[str(r.GAME_DATE)[:10], str(r.MATCHUP), r.WL if isinstance(r.WL, str) else "",
                           iv(r.MIN), iv(r.PTS), iv(r.REB), iv(r.AST), iv(r.FG3M)] for r in g.head(30).itertuples()]
    data = {"built": time.time(), "date": et_now().date().isoformat(), "logs": logs, "meta": meta, "tg": dict(tg), "standings": st}
    D["data"], D["error"] = data, None
    try:
        json.dump(data, open(CACHE_FILE + ".tmp", "w"))
        os.replace(CACHE_FILE + ".tmp", CACHE_FILE)
    except Exception as ex:
        print("  data: could not save cache:", ex, flush=True)
    step("Done")


def load_cache():
    try:
        raw = json.load(open(CACHE_FILE))
        for k in ("logs", "meta", "tg"):
            raw[k] = {int(a): b for a, b in raw[k].items()}
        return raw
    except Exception:
        return None


def needs_refresh():
    d = D["data"]
    return d is None or time.time() - d["built"] > REFRESH_HOURS * 3600 or d["date"] != et_now().date().isoformat()


def _worker():
    try:
        refresh_data()
    except Exception as ex:
        traceback.print_exc()
        D["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        D["building"] = False


def maybe_refresh():
    with DL:
        if D["building"] or not needs_refresh():
            return
        if D["error"] and time.time() - D["last_try"] < 60:
            return
        D.update(building=True, last_try=time.time())
    threading.Thread(target=_worker, daemon=True).start()


def watch():
    while True:  # every 30s: rebuild at midnight (Eastern) or when data is a few hours old
        time.sleep(30)
        try:
            maybe_refresh()
            if ND["wanted"]:
                nfl_maybe_refresh()
        except Exception:
            pass


def get_data():
    """(data, None) or (None, response to send while loading / on error)."""
    maybe_refresh()
    if D["data"]:
        return D["data"], None
    if D["error"] and not D["building"]:
        return None, jsonify(error=f"Could not load NBA data: {D['error']}")
    return None, jsonify(loading=True, step=D["step"])


# =============================== schedule ===============================
_sched = {"t": 0, "j": None}
_gcache = {}


def fmt_time(g):
    try:
        t = datetime.strptime((g.get("gameDateTimeEst") or "")[11:16], "%H:%M")
        return f"{t.hour % 12 or 12}:{t.minute:02d} {'am' if t.hour < 12 else 'pm'} ET"
    except Exception:
        return str(g.get("gameStatusText", "")).strip()


ESPN_FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "CHO": "CHA"}
_gerr = {"msg": ""}


def games_from_nba_cdn(day):
    if _sched["j"] is None or time.time() - _sched["t"] > 1800:
        r = requests.get(SCHEDULE_URL, timeout=15, headers=BROWSER)
        r.raise_for_status()
        _sched.update(t=time.time(), j=r.json())
    key, out = day.strftime("%m/%d/%Y"), []
    for gd in _sched["j"]["leagueSchedule"]["gameDates"]:
        if str(gd.get("gameDate", "")).startswith(key):
            for g in gd["games"]:
                if str(g["gameId"])[:3] in ("002", "004", "005", "006"):  # not preseason
                    out.append({"gid": str(g["gameId"]), "home_id": int(g["homeTeam"]["teamId"]),
                                "away_id": int(g["awayTeam"]["teamId"]), "time": fmt_time(g)})
    return out


def games_from_espn(day):
    ids = {t["abbreviation"]: t["id"] for t in TEAMS.values()}
    r = requests.get(ESPN_NBA + "/scoreboard", params={"dates": day.strftime("%Y%m%d")}, timeout=12,
                     headers={"User-Agent": BROWSER["User-Agent"]})
    r.raise_for_status()
    out = []
    for e in r.json().get("events", []):
        cs = {c["homeAway"]: c["team"].get("abbreviation", "") for c in e["competitions"][0]["competitors"]}
        h, a = ids.get(ESPN_FIX.get(cs["home"], cs["home"])), ids.get(ESPN_FIX.get(cs["away"], cs["away"]))
        if not h or not a:
            continue
        t = utc_dt(e.get("date", ""))
        out.append({"gid": "espn" + str(e["id"]), "home_id": h, "away_id": a,
                    "time": f"{t.hour % 12 or 12}:{t.minute:02d} {'am' if t.hour < 12 else 'pm'} ET" if t else ""})
    return out


def games_from_stats_api(day):
    sb = scoreboardv2.ScoreboardV2(game_date=day.strftime("%m/%d/%Y"), timeout=30).get_data_frames()[0]
    out, seen = [], set()
    for r in sb.itertuples():
        if r.GAME_ID not in seen:
            seen.add(r.GAME_ID)
            out.append({"gid": str(r.GAME_ID), "home_id": int(r.HOME_TEAM_ID), "away_id": int(r.VISITOR_TEAM_ID),
                        "time": str(r.GAME_STATUS_TEXT).strip()})
    return out


def get_games(day):
    hit = _gcache.get(day)
    if hit and hit[0] > time.time():
        return hit[1]
    out, errs, answered = [], [], False
    for fn in (games_from_nba_cdn, games_from_espn, games_from_stats_api):
        try:
            out = fn(day)
            answered = True  # this source loaded fine; an empty list means no games that day
            if out:
                break
        except Exception as ex:
            errs.append(f"{fn.__name__}: {ex}")
            print("  schedule source failed:", errs[-1], flush=True)
    _gerr["msg"] = "" if answered else "; ".join(errs)[:240]
    _gcache[day] = (time.time() + (1800 if out else 120), out)
    return out


# =============================== money line model ===============================
def rec(s):
    try:
        a, b = str(s).split("-")
        return int(a), int(b)
    except Exception:
        return 0, 0


def shrink(w, l, prior=6):  # pulls tiny samples toward .500
    return (w + prior / 2) / (w + l + prior)


def team_stand(d, tid):
    return next((s for s in d["standings"] if s["id"] == tid), None)


def rate(d, tid, home):
    s = team_stand(d, tid)
    if not s:
        return None
    gms = d["tg"].get(tid, [])
    margin = sum(g[1] for g in gms) / len(gms) if gms else 0.0
    locrec = s["home"] if home else s["road"]
    return {"s": s, "season": shrink(s["w"], s["l"]), "form": shrink(*rec(s["l10"])), "loc": shrink(*rec(locrec)),
            "mpct": min(1, max(0, 0.5 + margin / 40)), "margin": margin, "locrec": locrec}


def moneyline(d, hid, aid):
    h, a = rate(d, hid, True), rate(d, aid, False)
    if not h or not a:
        return None
    val = lambda r: 0.40 * r["season"] + 0.25 * r["form"] + 0.20 * r["loc"] + 0.15 * r["mpct"]
    ph = 1 / (1 + math.exp(-4 * (val(h) - val(a) + 0.04)))  # +0.04 = home-court edge
    home_pick = ph >= 0.5
    p = ph if home_pick else 1 - ph
    (pt, pr), (ot, orr) = ((hid, h), (aid, a)) if home_pick else ((aid, a), (hid, h))
    pa, oa = abbr(pt), abbr(ot)
    sgn = lambda x: f"{x:+.1f}"
    reasons = [
        f"Season record: {pa} {pr['s']['w']}-{pr['s']['l']}, {oa} {orr['s']['w']}-{orr['s']['l']}",
        f"Last 10 games: {pa} {pr['s']['l10']}, {oa} {orr['s']['l10']}",
        f"Average winning margin, last 10: {pa} {sgn(pr['margin'])}, {oa} {sgn(orr['margin'])}",
        f"{'Home' if home_pick else 'Road'} record for {pa}: {pr['locrec']}. {'Road' if home_pick else 'Home'} record for {oa}: {orr['locrec']}",
    ]
    return {"team_id": pt, "team": pa, "name": TEAMS[pt]["full_name"], "prob": round(100 * p), "other_prob": round(100 * (1 - p)),
            "other": oa, "label": "Strong lean" if p >= 0.70 else "Lean" if p >= 0.60 else "Close game", "reasons": reasons}


# =============================== player picks ===============================
_cc = {}


def fresh(games, today, w):
    try:
        last = datetime.strptime(games[0][0], "%Y-%m-%d").date()
    except Exception:
        return False
    return (today - last).days <= 14 and sum(g[3] for g in games[:w]) / w >= 15


def cands_for(d, w):
    key = (d["built"], w)
    if key in _cc:
        return _cc[key]
    out, today = [], et_now().date()
    for pid, games in d["logs"].items():
        meta = d["meta"].get(pid)
        if not meta or len(games) < w + 8 or not fresh(games, today, w):
            continue
        recent, base = games[:w], games[w:30]
        for stat, (label, lines, f) in STATS.items():
            rv, bv, best = [f(g) for g in recent], [f(g) for g in base], None
            for line in lines:
                hits = sum(v >= line for v in rv)
                if hits < HOT_MIN[w]:
                    continue
                b, r = sum(v >= line for v in bv) / len(bv), hits / w
                if b > MAX_BASE or r - b < MIN_LIFT:
                    continue
                if best is None or (r - b, line) > best[0]:
                    best = ((r - b, line), line, hits, b, r)
            if best:
                _, line, hits, b, r = best
                out.append({"id": pid, "name": meta["name"], "team_id": meta["team_id"], "stat": stat, "line": line,
                            "text": f"Over {line - 0.5:g} {LBL[stat]}", "vals": rv, "hits": hits, "base": round(100 * b),
                            "p": round(0.6 * r + 0.4 * b, 3), "lift": round(100 * (r - b))})
    out.sort(key=lambda c: (-c["p"], -c["lift"]))
    if len(_cc) > 6:
        _cc.clear()
    _cc[key] = out
    return out


def rising_for(d, w, playing):
    res, today = [], et_now().date()
    for pid, games in d["logs"].items():
        meta = d["meta"].get(pid)
        if not meta or meta["team_id"] not in playing or len(games) < w + 8 or not fresh(games, today, w):
            continue
        recent, base, best = games[:w], games[w:w + 15], None
        for stat, (label, lines, f) in STATS.items():
            ar, ab = sum(f(g) for g in recent) / w, sum(f(g) for g in base) / len(base)
            delta = ar - ab
            if ab <= 0 or delta < MIN_DELTA[stat] or delta / ab < 0.15:
                continue
            score = delta / max(ab, 1)
            if best is None or score > best[0]:
                best = (score, stat, ar, ab, delta)
        if best:
            score, stat, ar, ab, delta = best
            f, p = STATS[stat][2], playing[meta["team_id"]]
            res.append({"id": pid, "name": meta["name"], "team": abbr(meta["team_id"]), "opp": p["opp"], "where": p["where"],
                        "stat": stat, "label": STATS[stat][0], "w": w, "series": [f(g) for g in reversed(games[:w + 10])],
                        "avg_recent": round(ar, 1), "avg_base": round(ab, 1), "delta": round(delta, 1), "score": score})
    res.sort(key=lambda c: -c["score"])
    return res[:8]


def today_info(d, w):
    games, playing, gl = get_games(et_now().date()), {}, []
    for g in games:
        h, a = g["home_id"], g["away_id"]
        playing[h] = {"opp": abbr(a), "where": "vs", "time": g["time"]}
        playing[a] = {"opp": abbr(h), "where": "@", "time": g["time"]}
        pk = moneyline(d, h, a)
        gl.append({"gid": g["gid"], "home": abbr(h), "away": abbr(a), "home_id": h, "away_id": a, "time": g["time"],
                   "pick": {k: pk[k] for k in ("team", "prob", "label")} if pk else None})
    strict = [c for c in cands_for(d, w) if c["team_id"] in playing]
    steady = team_best(d["logs"], d["meta"], set(playing), w, STATS, fresh_fn=lambda g: fresh(g, et_now().date(), min(w, len(g))))
    return gl, with_backup(strict, steady, playing, abbr), playing


def pick_legs(pool, n, rng, cap=2):
    legs, per_team, pool = [], {}, list(pool)
    while pool and len(legs) < n:
        c = rng.choices(pool, weights=[x["p"] ** 2 for x in pool])[0] if rng else pool[0]
        pool.remove(c)
        if per_team.get(c["team"], 0) >= cap:  # spread across teams (no limit for one-team bets)
            continue
        per_team[c["team"]] = per_team.get(c["team"], 0) + 1
        legs.append(c)
    legs.sort(key=lambda c: -c["p"])
    return legs


# =============================== API ===============================
def wparam():
    return 10 if request.args.get("w") == "10" else 5


@app.route("/api/teams")
def api_teams():
    return jsonify([{"id": t["id"], "name": t["full_name"], "nick": t["nickname"], "abbr": t["abbreviation"], "city": t["city"]}
                    for t in sorted(TEAMS.values(), key=lambda t: t["full_name"])])


@app.route("/api/today")
def api_today():
    d, resp = get_data()
    if resp:
        return resp
    day = et_now().date()
    gl, c5, playing = today_info(d, 5)
    c10 = today_info(d, 10)[1]
    return jsonify(label=f"{day:%A, %B} {day.day}", games=gl, cands={"5": c5[:80], "10": c10[:80]},
                   rising={"5": rising_for(d, 5, playing), "10": rising_for(d, 10, playing)},
                   next_refresh_in=secs_to_midnight(), sched_error=_gerr["msg"] if not gl else "")


@app.route("/api/parlay")
def api_parlay():
    d, resp = get_data()
    if resp:
        return resp
    w, n, variant = wparam(), max(1, min(10, int(request.args.get("n", 5)))), int(request.args.get("variant", 0))
    return make_parlay(today_info(d, w)[1], n, variant)


def team_best(logs, meta, tids, w, statdefs, fresh_fn=None):
    """Steady picks: for each player, the HIGHEST number they reached in at least ~60% of their last w games."""
    need, out = -(-6 * w // 10), []
    tids = {tids} if isinstance(tids, int) else set(tids)
    lbl = LBL if statdefs is STATS else NLBL
    for pid, games in logs.items():
        m = meta.get(pid)
        if not m or m["team_id"] not in tids or len(games) < min(w, 3) or (fresh_fn and not fresh_fn(games)):
            continue
        for stat, (label, lines, f, *rest) in statdefs.items():
            if rest and rest[0] and m.get("pos") not in rest[0]:
                continue
            rv, bv, best = [f(g) for g in games[:w]], [f(g) for g in games[w:w + 15]], None
            for line in lines:
                hits = sum(v >= line for v in rv)
                if hits >= need and (best is None or line > best[1]):
                    best = (hits, line)
            if best:
                hits, line = best
                b = sum(v >= line for v in bv) / len(bv) if bv else 0
                r = hits / len(rv)
                out.append({"id": pid, "name": m["name"], "pos": m.get("pos", ""), "team_id": m["team_id"], "stat": stat, "line": line,
                            "text": f"Over {line - 0.5:g} {lbl[stat]}", "vals": rv, "hits": hits, "base": round(100 * b),
                            "p": round(0.6 * r + 0.4 * b - 0.02, 3), "lift": round(100 * (r - b)), "kind": "steady"})
    return out


def with_backup(strict, steady, playing, abbr_fn):
    """Fresh hot streaks first, then steady picks for anyone not already listed. Both limited to teams in `playing`."""
    seen = {(c["id"], c["stat"]) for c in strict}
    hot = [{**c, "kind": "hot"} for c in strict]
    steady = sorted((s for s in steady if (s["id"], s["stat"]) not in seen), key=lambda c: (-c["p"], -c["lift"]))
    return [{**c, "team": abbr_fn(c["team_id"]), **playing[c["team_id"]]} for c in hot + steady if c["team_id"] in playing]


def make_parlay(pool, n, variant, cap=2):
    best = {}
    for c in pool:  # one leg per player: their strongest, and a real hot streak beats a steady one
        k = (c.get("kind") != "steady", c["p"])
        if c["id"] not in best or k > (best[c["id"]].get("kind") != "steady", best[c["id"]]["p"]):
            best[c["id"]] = c
    pool = sorted(best.values(), key=lambda c: (c.get("kind") == "steady", -c["p"], -c["lift"]))
    if variant == 0:
        legs = pick_legs(pool, n, None, cap)
    else:
        rng = random.Random(variant)
        legs = pick_legs(pool[:max(2 * n, n + 6)], n, rng, cap)
        if len(legs) < n:
            legs = pick_legs(pool, n, rng, cap)
    chance = 1.0
    for c in legs:
        chance *= c["p"]
    return jsonify(legs=legs, chance=round(100 * chance, 1) if legs else 0)


@app.route("/api/team_parlay")
def api_team_parlay():
    d, resp = get_data()
    if resp:
        return resp
    tid, w = int(request.args["team_id"]), wparam()
    n, variant = max(1, min(10, int(request.args.get("n", 5)))), int(request.args.get("variant", 0))
    strict = [c for c in cands_for(d, w) if c["team_id"] == tid]
    steady = team_best(d["logs"], d["meta"], tid, w, STATS, fresh_fn=lambda g: fresh(g, et_now().date(), min(w, len(g))))
    blank = {tid: {"opp": "", "where": "", "time": ""}}
    return make_parlay(with_backup(strict, steady, blank, abbr), n, variant, cap=99)


@app.route("/api/game")
def api_game():
    d, resp = get_data()
    if resp:
        return resp
    gid = request.args.get("gid", "")
    g = next((x for x in get_games(et_now().date()) if x["gid"] == gid), None)
    if not g:
        return jsonify(error="That game isn't on today's schedule anymore.")
    today, sides = et_now().date(), []
    for tid, where in ((g["away_id"], "Away"), (g["home_id"], "Home")):
        pl = []
        for pid, meta in d["meta"].items():
            games = d["logs"].get(pid, [])[:10]
            if meta["team_id"] == tid and len(games) >= 3 and (today - datetime.strptime(games[0][0], "%Y-%m-%d").date()).days <= 21:
                pl.append({"id": pid, "name": meta["name"], "games": games})
        pl.sort(key=lambda p: -sum(x[3] for x in p["games"][:5]) / min(5, len(p["games"])))
        s = team_stand(d, tid)
        sides.append({"id": tid, "name": TEAMS[tid]["full_name"], "abbr": abbr(tid), "where": where,
                      "record": f"{s['w']}-{s['l']}" if s else "", "players": pl[:12]})
    return jsonify(time=g["time"], pick=moneyline(d, g["home_id"], g["away_id"]), sides=sides)


@app.route("/api/props")
def api_props():
    d, resp = get_data()
    if resp:
        return resp
    tid, stat, line, w = int(request.args["team_id"]), request.args["stat"], int(request.args["line"]), wparam()
    f, rows, today = STATS[stat][2], [], et_now().date()
    for pid, meta in d["meta"].items():
        games = d["logs"].get(pid, [])[:w]
        if meta["team_id"] != tid or len(games) < min(w, 3):
            continue
        if (today - datetime.strptime(games[0][0], "%Y-%m-%d").date()).days > 30:
            continue
        vals = [f(g) for g in games]
        hits = sum(v >= line for v in vals)
        rows.append({"id": pid, "name": meta["name"], "vals": vals, "hits": hits, "n": len(vals), "pct": round(100 * hits / len(vals))})
    rows.sort(key=lambda r: (-r["pct"], -sum(r["vals"])))
    return jsonify(rows=rows)


@app.route("/api/search")
def api_search():
    d = D["data"]
    q = request.args.get("q", "").strip().lower()
    if not d or not q:
        return jsonify([])
    hits = [(pid, m) for pid, m in d["meta"].items() if q in m["name"].lower()]
    hits.sort(key=lambda x: (not any(w.startswith(q) for w in x[1]["name"].lower().split()),
                             not x[1]["name"].lower().startswith(q), x[1]["name"]))
    return jsonify([{"id": pid, "name": m["name"], "team": abbr(m["team_id"])} for pid, m in hits[:6]])


@app.route("/api/player")
def api_player():
    d, resp = get_data()
    if resp:
        return resp
    pid = request.args.get("id", type=int)
    if pid is None:
        q = request.args.get("name", "").strip().lower()
        hits = [p for p, m in d["meta"].items() if m["name"].lower() == q] or \
               [p for p, m in d["meta"].items() if q and q in m["name"].lower()]
        pid = hits[0] if hits else None
    m = d["meta"].get(pid)
    if not m:
        return jsonify(error="No current player found.")
    games = d["logs"].get(pid, [])[:10]
    if not games:
        return jsonify(error=f"No recent games found for {m['name']}.")
    out = [{"date": g[0], "matchup": g[1], "wl": g[2], "min": g[3], "pts": g[4], "reb": g[5], "ast": g[6], "fg3": g[7],
            "pra": g[4] + g[5] + g[6]} for g in games]
    avg = lambda gs: {k: round(sum(g[k] for g in gs) / len(gs), 1) for k in ("pts", "reb", "ast", "fg3", "pra", "min")}
    return jsonify(id=pid, name=m["name"], team=TEAMS[m["team_id"]]["full_name"], games=out, avg5=avg(out[:5]), avg10=avg(out))


@app.route("/api/standings")
def api_standings():
    d, resp = get_data()
    if resp:
        return resp
    conf = {"East": [], "West": []}
    for s in d["standings"]:
        if s["conf"] in conf:
            conf[s["conf"]].append(s)
    out = {}
    for c, rows in conf.items():
        rows.sort(key=lambda s: (-(s["w"] / max(1, s["w"] + s["l"])), -s["w"]))
        lw, ll = (rows[0]["w"], rows[0]["l"]) if rows else (0, 0)
        out[c] = [{"rank": i + 1, "id": s["id"], "name": TEAMS[s["id"]]["full_name"], "abbr": abbr(s["id"]), "nick": TEAMS[s["id"]]["nickname"], "w": s["w"], "l": s["l"],
                   "pct": f"{s['w'] / max(1, s['w'] + s['l']):.3f}".lstrip("0"), "gb": f"{((lw - s['w']) + (s['l'] - ll)) / 2:g}" if i else "-",
                   "l10": s["l10"], "streak": s["streak"]} for i, s in enumerate(rows)]
    return jsonify(east=out["East"], west=out["West"])


LIVE_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com", "Accept": "application/json, text/plain, */*"}
_live = {"t": 0, "out": None}
_box = {}


def live_from_nba():
    r = requests.get(LIVE_URL, timeout=8, headers=BROWSER)
    r.raise_for_status()
    games = []
    for g in r.json()["scoreboard"]["games"]:
        h, a = g["homeTeam"], g["awayTeam"]
        games.append({"src": "nba", "gid": str(g["gameId"]), "status": iv(g.get("gameStatus")), "text": str(g.get("gameStatusText", "")).strip(),
                      "home": h["teamTricode"], "away": a["teamTricode"], "home_id": iv(h["teamId"]),
                      "away_id": iv(a["teamId"]), "home_score": iv(h.get("score")), "away_score": iv(a.get("score"))})
    return games


def live_from_espn():
    r = requests.get(ESPN_NBA + "/scoreboard", timeout=8, headers={"User-Agent": BROWSER["User-Agent"]})
    r.raise_for_status()
    games = []
    for e in r.json().get("events", []):
        ty = (e.get("status") or {}).get("type") or {}
        cs = {c["homeAway"]: c for c in e["competitions"][0]["competitors"]}
        logo = lambda c: f"https://a.espncdn.com/i/teamlogos/nba/500/{c['team'].get('abbreviation', '').lower()}.png"
        games.append({"src": "espn", "gid": str(e["id"]), "status": {"pre": 1, "in": 2, "post": 3}.get(ty.get("state"), 1),
                      "text": ty.get("shortDetail", ""), "home": cs["home"]["team"].get("abbreviation", ""), "away": cs["away"]["team"].get("abbreviation", ""),
                      "home_id": 0, "away_id": 0, "home_logo": logo(cs["home"]), "away_logo": logo(cs["away"]),
                      "home_score": iv(cs["home"].get("score")), "away_score": iv(cs["away"].get("score"))})
    return games


@app.route("/api/live")
def api_live():
    if _live["out"] is None or time.time() - _live["t"] > 8:
        games, errs = None, []
        for fn in (live_from_nba, live_from_espn):
            try:
                games = fn()
                break
            except Exception as ex:
                errs.append(f"{fn.__name__}: {ex}")
                print("  live scores failed:", errs[-1], flush=True)
        if games is not None:
            _live.update(t=time.time(), out={"games": games, "updated": et_now().strftime("%I:%M:%S %p").lstrip("0")})
        elif _live["out"] is None:
            return jsonify(error="Could not reach the NBA or ESPN live feeds: " + "; ".join(errs)[:200])
    return jsonify(_live["out"])


def clock_nba(s):
    m = re.match(r"PT(\d+)M([\d.]+)S", str(s or ""))
    return f"{int(m.group(1))}:{int(float(m.group(2))):02d}" if m else ""


def box_nba(gid):
    j = requests.get(NBA_BOX_URL.format(gid), timeout=8, headers=BROWSER)
    j.raise_for_status()
    g = j.json()["game"]
    def side(t):
        pl = []
        for p in t.get("players", []):
            s = p.get("statistics") or {}
            mn = clock_nba(s.get("minutes"))
            if not mn or mn == "0:00":
                continue
            pl.append({"id": p.get("personId"), "name": p.get("name", ""), "min": mn, "pts": iv(s.get("points")), "reb": iv(s.get("reboundsTotal")),
                       "ast": iv(s.get("assists")), "fg3": iv(s.get("threePointersMade")), "stl": iv(s.get("steals")), "blk": iv(s.get("blocks")),
                       "tov": iv(s.get("turnovers")), "fg": f"{iv(s.get('fieldGoalsMade'))}-{iv(s.get('fieldGoalsAttempted'))}",
                       "on": p.get("oncourt") == "1"})
        pl.sort(key=lambda x: (-x["pts"], -x["reb"]))
        return {"abbr": t.get("teamTricode", ""), "score": iv(t.get("score")), "players": pl}
    return {"status": g.get("gameStatusText", "").strip(), "away": side(g["awayTeam"]), "home": side(g["homeTeam"]), "photo": "nba"}


def box_espn(eid):
    j = requests.get(ESPN_NBA + "/summary", params={"event": eid}, timeout=8, headers={"User-Agent": BROWSER["User-Agent"]})
    j.raise_for_status()
    j = j.json()
    comp = j.get("header", {}).get("competitions", [{}])[0]
    scores = {c["homeAway"]: iv(c.get("score")) for c in comp.get("competitors", [])}
    sides = {}
    for t in j["boxscore"]["players"]:
        pl = []
        for blk in t.get("statistics", []):
            lab = blk.get("labels") or blk.get("keys") or []
            for a in blk.get("athletes", []):
                v = dict(zip(lab, a.get("stats", [])))
                mn = str(v.get("MIN", "") or v.get("minutes", ""))
                if not mn or mn in ("0", "--"):
                    continue
                fl = lambda k: iv(str(v.get(k, 0)).split("-")[0] or 0)
                pl.append({"id": a["athlete"].get("id"), "name": a["athlete"].get("displayName", ""), "min": mn, "pts": fl("PTS"), "reb": fl("REB"),
                           "ast": fl("AST"), "fg3": fl("3PT"), "stl": fl("STL"), "blk": fl("BLK"), "tov": fl("TO"), "fg": str(v.get("FG", "")), "on": False})
        pl.sort(key=lambda x: (-x["pts"], -x["reb"]))
        sides[t["team"].get("homeAway") or ""] = {"abbr": t["team"].get("abbreviation", ""), "players": pl}
    for k in ("home", "away"):
        s = sides.setdefault(k, {"abbr": "", "players": []})
        s["score"] = scores.get(k, 0)
    return {"status": (comp.get("status") or {}).get("type", {}).get("shortDetail", ""), "away": sides["away"], "home": sides["home"], "photo": "espn"}


@app.route("/api/live/box")
def api_live_box():
    src_, gid = request.args.get("src", "nba"), request.args.get("gid", "")
    key = (src_, gid)
    hit = _box.get(key)
    if hit and time.time() - hit[0] < 6:
        return jsonify(hit[1])
    try:
        out = box_espn(gid) if src_ == "espn" else box_nba(gid)
    except Exception as ex:
        print("  live box score failed:", ex, flush=True)
        if hit:
            return jsonify(hit[1])
        return jsonify(error="Player stats for this game are not available yet. They appear once the game starts.")
    out["updated"] = et_now().strftime("%I:%M:%S %p").lstrip("0")
    _box[key] = (time.time(), out)
    return jsonify(out)


# =============================== NFL (data from ESPN's public JSON) ===============================
NFL_TEAMS = {  # espn id: (abbr, full name, nickname, city, conference)
    1: ("ATL", "Atlanta Falcons", "Falcons", "Atlanta", "NFC"), 2: ("BUF", "Buffalo Bills", "Bills", "Buffalo", "AFC"),
    3: ("CHI", "Chicago Bears", "Bears", "Chicago", "NFC"), 4: ("CIN", "Cincinnati Bengals", "Bengals", "Cincinnati", "AFC"),
    5: ("CLE", "Cleveland Browns", "Browns", "Cleveland", "AFC"), 6: ("DAL", "Dallas Cowboys", "Cowboys", "Dallas", "NFC"),
    7: ("DEN", "Denver Broncos", "Broncos", "Denver", "AFC"), 8: ("DET", "Detroit Lions", "Lions", "Detroit", "NFC"),
    9: ("GB", "Green Bay Packers", "Packers", "Green Bay", "NFC"), 10: ("TEN", "Tennessee Titans", "Titans", "Tennessee", "AFC"),
    11: ("IND", "Indianapolis Colts", "Colts", "Indianapolis", "AFC"), 12: ("KC", "Kansas City Chiefs", "Chiefs", "Kansas City", "AFC"),
    13: ("LV", "Las Vegas Raiders", "Raiders", "Las Vegas", "AFC"), 14: ("LAR", "Los Angeles Rams", "Rams", "Los Angeles", "NFC"),
    15: ("MIA", "Miami Dolphins", "Dolphins", "Miami", "AFC"), 16: ("MIN", "Minnesota Vikings", "Vikings", "Minnesota", "NFC"),
    17: ("NE", "New England Patriots", "Patriots", "New England", "AFC"), 18: ("NO", "New Orleans Saints", "Saints", "New Orleans", "NFC"),
    19: ("NYG", "New York Giants", "Giants", "New York", "NFC"), 20: ("NYJ", "New York Jets", "Jets", "New York", "AFC"),
    21: ("PHI", "Philadelphia Eagles", "Eagles", "Philadelphia", "NFC"), 22: ("ARI", "Arizona Cardinals", "Cardinals", "Arizona", "NFC"),
    23: ("PIT", "Pittsburgh Steelers", "Steelers", "Pittsburgh", "AFC"), 24: ("LAC", "Los Angeles Chargers", "Chargers", "Los Angeles", "AFC"),
    25: ("SF", "San Francisco 49ers", "49ers", "San Francisco", "NFC"), 26: ("SEA", "Seattle Seahawks", "Seahawks", "Seattle", "NFC"),
    27: ("TB", "Tampa Bay Buccaneers", "Buccaneers", "Tampa Bay", "NFC"), 28: ("WSH", "Washington Commanders", "Commanders", "Washington", "NFC"),
    29: ("CAR", "Carolina Panthers", "Panthers", "Carolina", "NFC"), 30: ("JAX", "Jacksonville Jaguars", "Jaguars", "Jacksonville", "AFC"),
    33: ("BAL", "Baltimore Ravens", "Ravens", "Baltimore", "AFC"), 34: ("HOU", "Houston Texans", "Texans", "Houston", "AFC"),
}
ESPN_NFL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
NFL_CACHE_FILE = os.path.join(HERE, "nfl_cache.json")
# stat -> (label, lines, getter, positions). A game row is
# [date, matchup, wl, pass yds, pass td, rush yds, rush td, rec, rec yds, rec td, pass att, carries, targets]
NSTATS = {
    "rec": ("Receptions", [3, 4, 5, 6, 7, 8, 10], lambda g: g[7], {"WR", "TE", "RB"}),
    "recyds": ("Receiving yards", [30, 40, 50, 60, 70, 80, 100], lambda g: g[8], {"WR", "TE", "RB"}),
    "rushyds": ("Rushing yards", [30, 40, 50, 60, 70, 80, 100], lambda g: g[5], {"RB", "QB"}),
    "passyds": ("QB passing yards", [150, 175, 200, 225, 250, 275, 300], lambda g: g[3], {"QB"}),
    "passtd": ("Passing TDs", [1, 2, 3, 4], lambda g: g[4], {"QB"}),
    "td": ("Touchdowns", [1, 2], lambda g: g[6] + g[9], {"RB", "WR", "TE", "QB"}),
}
NLBL = {"rec": "receptions", "recyds": "receiving yards", "rushyds": "rushing yards", "passyds": "passing yards",
        "passtd": "passing TDs", "td": "touchdowns"}
N_DELTA = {"rec": 1.5, "recyds": 15, "rushyds": 15, "passyds": 40, "passtd": 0.7, "td": 0.35}
ND = {"data": None, "building": False, "error": None, "step": "", "last_try": 0, "wanted": False}
NSTORE = {"events": {}, "weeks": {}, "bad": []}  # saved to disk so restarts are fast
NLOCK = threading.Lock()


def nabbr(tid):
    return NFL_TEAMS.get(tid, ("?",))[0]


def nlogo(a):
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{str(a).lower()}.png"


def nstep(msg):
    ND["step"] = msg
    print("  nfl:", msg, flush=True)


def espn(path, **params):
    for i in range(3):
        try:
            r = requests.get(ESPN_NFL + path, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == 2:
                raise
            time.sleep(1 + i)


def fl(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return 0.0


def utc_dt(iso):
    for f in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            t = datetime.strptime(iso, f).replace(tzinfo=timezone.utc)
            return t.astimezone(ET) if ET else t
        except ValueError:
            pass
    return None


def et_label(iso):
    t = utc_dt(iso)
    return f"{t:%a} {t.hour % 12 or 12}:{t.minute:02d} {'am' if t.hour < 12 else 'pm'} {'ET' if ET else 'UTC'}" if t else ""


def norm_event(e, season=None, week=None):
    cs = {c["homeAway"]: c for c in e["competitions"][0]["competitors"]}
    ty = (e.get("status") or {}).get("type") or {}
    side = lambda c: {"id": int(c["team"]["id"]), "abbr": c["team"].get("abbreviation", ""), "score": iv(c.get("score"))}
    return {"eid": str(e["id"]), "date": e.get("date", ""), "state": ty.get("state", ""), "done": bool(ty.get("completed")),
            "detail": ty.get("shortDetail", ""), "season": season, "week": week or (e.get("week") or {}).get("number"),
            "home": side(cs["home"]), "away": side(cs["away"])}


PASS_K = {"passingYards": "pyd", "YDS": "pyd", "passingTouchdowns": "ptd", "TD": "ptd", "completions/passingAttempts": "patt", "C/ATT": "patt"}
RUSH_K = {"rushingYards": "ryd", "YDS": "ryd", "rushingTouchdowns": "rtd", "TD": "rtd", "rushingAttempts": "car", "CAR": "car"}
REC_K = {"receptions": "rec", "REC": "rec", "receivingYards": "recyd", "YDS": "recyd", "receivingTouchdowns": "rectd", "TD": "rectd",
         "receivingTargets": "tgt", "TGTS": "tgt"}
CAT_K = {"passing": PASS_K, "rushing": RUSH_K, "receiving": REC_K}
POS_FIX = {"FB": "RB", "HB": "RB"}


def parse_box(j):
    """player id -> [name, pos, team id, pass yds, pass td, rush yds, rush td, rec, rec yds, rec td, pass att, carries, targets]"""
    out = {}
    for side in j["boxscore"]["players"]:
        tid = int(side["team"]["id"])
        for cat in side.get("statistics", []):
            kmap = CAT_K.get(cat.get("name"))
            if not kmap:
                continue
            keys = cat.get("keys") or cat.get("labels") or []
            for a in cat.get("athletes", []):
                ath = a["athlete"]
                pos = (ath.get("position") or {}).get("abbreviation", "")
                p = out.setdefault(int(ath["id"]), {"name": ath.get("displayName", ""), "pos": POS_FIX.get(pos, pos), "tid": tid, "v": defaultdict(float)})
                for k, v in zip(keys, a.get("stats", [])):
                    f = kmap.get(k)
                    if f == "patt":
                        p["v"][f] += fl(str(v).split("/")[-1])
                    elif f:
                        p["v"][f] += fl(v)
    return {pid: [p["name"], p["pos"], p["tid"]] + [int(p["v"][k]) for k in ("pyd", "ptd", "ryd", "rtd", "rec", "recyd", "rectd", "patt", "car", "tgt")]
            for pid, p in out.items()}


def fetch_box(ev):
    """(record, ok). ok=False only when the request itself failed (so we retry next time)."""
    try:
        j = espn("/summary", event=ev["eid"])
    except Exception:
        return None, False
    try:
        players = parse_box(j)
    except Exception:
        return None, True
    t = utc_dt(ev["date"])
    return {"date": t.date().isoformat() if t else ev["date"][:10], "week": ev["week"], "season": ev["season"],
            "home": ev["home"], "away": ev["away"], "players": {str(k): v for k, v in players.items()}}, True


def nfl_build(season):
    events = sorted(NSTORE["events"].values(), key=lambda e: (e["date"], str(e["week"])))
    recs = {t: {"w": 0, "l": 0, "t": 0, "pf": 0, "pa": 0, "home": [0, 0], "road": [0, 0], "res": []} for t in NFL_TEAMS}
    for e in events:
        if e["season"] != season:
            continue
        for me, op, ishome in ((e["home"], e["away"], True), (e["away"], e["home"], False)):
            r = recs.get(me["id"])
            if r is None:
                continue
            win = 1 if me["score"] > op["score"] else 0 if me["score"] < op["score"] else 0.5
            r["pf"] += me["score"]
            r["pa"] += op["score"]
            r["w" if win == 1 else "l" if win == 0 else "t"] += 1
            if win != 0.5:
                r["home" if ishome else "road"][0 if win == 1 else 1] += 1
            r["res"].append([win, me["score"] - op["score"]])
    logs, meta = defaultdict(list), {}
    for e in reversed(events):  # newest first
        h, a = e["home"], e["away"]
        for pid, row in e["players"].items():
            pid, ishome = int(pid), row[2] == h["id"]
            opp, mine, theirs = (a, h["score"], a["score"]) if ishome else (h, a["score"], h["score"])
            logs[pid].append([e["date"], ("vs " if ishome else "@ ") + opp["abbr"], "W" if mine > theirs else "L" if mine < theirs else "T"] + row[3:])
            if pid not in meta:
                meta[pid] = {"name": row[0], "pos": row[1], "team_id": row[2]}
    for pid, m in meta.items():
        gs = logs[pid]
        if m["pos"] not in ("QB", "RB", "WR", "TE"):
            m["pos"] = "QB" if sum(g[10] for g in gs) >= 10 else "RB" if sum(g[11] for g in gs) > sum(g[7] for g in gs) else "WR"
        logs[pid] = gs[:30]
    return {"built": time.time(), "date": et_now().date().isoformat(), "season": season, "logs": dict(logs), "meta": meta, "recs": recs}


def nfl_season_year():
    now = et_now()
    return now.year if now.month >= 8 else now.year - 1


def _store_boxes(pending):
    done = 0
    with ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(fetch_box, e): e for e in pending}
        for fut in as_completed(futs):
            e = futs[fut]
            rec, ok = fut.result()
            if rec:
                NSTORE["events"][e["eid"]] = rec
            elif ok:
                NSTORE["bad"].append(e["eid"])
            done += 1
            if done % 10 == 0 or done == len(pending):
                nstep(f"Downloading game box scores ({done}/{len(pending)})")


def nfl_refresh_core():
    nstep("Checking the NFL schedule")
    cur = espn("/scoreboard")
    info = cur.get("season") or {}
    season, stype = int(info.get("year") or nfl_season_year()), int(info.get("type") or 2)
    wk = int((cur.get("week") or {}).get("number") or 1)
    plan = []
    if stype == 2:
        plan += [(season, w) for w in range(1, wk + 1)]
    elif stype == 3:
        plan += [(season, w) for w in range(1, 19)]
    if stype == 1 or (stype == 2 and wk <= 14):  # early in the season: also read the end of last season
        plan += [(season - 1, w) for w in range(12, 19)]
    todo = [(s, w) for s, w in plan if not NSTORE["weeks"].get(f"{s}:{w}")]

    def read_week(sw):
        s, w = sw
        j = espn("/scoreboard", dates=s, seasontype=2, week=w, limit=100)
        return sw, [norm_event(e, s, w) for e in j.get("events", [])]

    nstep(f"Reading {len(todo)} weeks of schedule")
    weeks = {}
    with ThreadPoolExecutor(12) as ex:
        for sw, evs in ex.map(read_week, todo):
            weeks[sw] = evs
    pending = [e for evs in weeks.values() for e in evs if e["done"] and e["eid"] not in NSTORE["events"] and e["eid"] not in NSTORE["bad"]]
    _store_boxes(pending)
    for (s, w), evs in weeks.items():  # a week is finished once every game in it is final and stored
        if evs and all(e["done"] and (e["eid"] in NSTORE["events"] or e["eid"] in NSTORE["bad"]) for e in evs):
            NSTORE["weeks"][f"{s}:{w}"] = True
    return season


def nfl_save():
    try:
        json.dump(NSTORE, open(NFL_CACHE_FILE + ".tmp", "w"))
        os.replace(NFL_CACHE_FILE + ".tmp", NFL_CACHE_FILE)
    except Exception as ex:
        print("  nfl: could not save cache:", ex, flush=True)


def nfl_load_cache():
    try:
        raw = json.load(open(NFL_CACHE_FILE))
        NSTORE.update(events=raw["events"], weeks=raw["weeks"], bad=raw.get("bad", []))
        seasons = [e["season"] for e in NSTORE["events"].values() if e.get("season")]
        if seasons:
            ND["data"] = nfl_build(max(seasons))
    except Exception:
        pass


def nfl_needs_refresh():
    d = ND["data"]
    return d is None or time.time() - d["built"] > 3600 or d["date"] != et_now().date().isoformat()


def _nworker():
    try:
        season = nfl_refresh_core()
        nstep("Crunching the numbers")
        ND["data"], ND["error"] = nfl_build(season), None
        nfl_save()
        nstep("Done")
    except Exception as ex:
        traceback.print_exc()
        ND["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        ND["building"] = False


def nfl_maybe_refresh():
    ND["wanted"] = True
    with NLOCK:
        if ND["building"] or not nfl_needs_refresh():
            return
        if ND["error"] and time.time() - ND["last_try"] < 60:
            return
        ND.update(building=True, last_try=time.time())
    threading.Thread(target=_nworker, daemon=True).start()


def nfl_data():
    nfl_maybe_refresh()
    if ND["data"]:
        return ND["data"], None
    if ND["error"] and not ND["building"]:
        return None, jsonify(error=f"Could not load NFL data: {ND['error']}")
    return None, jsonify(loading=True, step=ND["step"])


# ---- this week's games ----
_nslate = {"t": 0, "events": [], "label": ""}


def nfl_slate():
    if time.time() - _nslate["t"] < 300 and _nslate["label"]:
        return _nslate
    try:
        j = espn("/scoreboard")
        info = j.get("season") or {}
        season, stype = int(info.get("year") or nfl_season_year()), int(info.get("type") or 2)
        wk = int((j.get("week") or {}).get("number") or 1)
        evs = [norm_event(e) for e in j.get("events", [])]
        left = [e for e in evs if e["state"] in ("pre", "in")]
        if not left and stype in (2, 3) and wk < 22:  # this week is over: show the next one
            try:
                j2 = espn("/scoreboard", dates=season, seasontype=stype, week=wk + 1)
                left = [e for e in (norm_event(x) for x in j2.get("events", [])) if e["state"] in ("pre", "in")]
                wk += 1 if left else 0
            except Exception:
                pass
        label = f"Week {wk}" if stype == 2 else ("Playoffs" if stype == 3 else "Preseason")
        _nslate.update(t=time.time(), events=left, label=label)
    except Exception as ex:
        print("  nfl: schedule failed:", ex, flush=True)
        _nslate["t"] = time.time() - 240  # try again in a minute
        _nslate["label"] = _nslate["label"] or "This week"
    return _nslate


# ---- money line + picks ----
def rec_str(r):
    return f"{r['w']}-{r['l']}" + (f"-{r['t']}" if r["t"] else "")


def nrate(d, tid, home):
    r = d["recs"].get(tid)
    if not r:
        return None
    last = r["res"][-5:]
    lw = sum(x[0] for x in last)
    margin = sum(x[1] for x in last) / len(last) if last else 0.0
    loc = r["home"] if home else r["road"]
    return {"rec": rec_str(r), "l5": f"{int(lw)}-{int(len(last) - lw)}", "margin": margin, "locrec": f"{loc[0]}-{loc[1]}",
            "season": shrink(r["w"] + r["t"] / 2, r["l"] + r["t"] / 2, 4), "form": shrink(lw, len(last) - lw, 4),
            "loc": shrink(loc[0], loc[1], 4), "mpct": min(1, max(0, 0.5 + margin / 50))}


def nfl_ml(d, hid, aid):
    h, a = nrate(d, hid, True), nrate(d, aid, False)
    if not h or not a:
        return None
    val = lambda r: 0.40 * r["season"] + 0.25 * r["form"] + 0.20 * r["loc"] + 0.15 * r["mpct"]
    ph = 1 / (1 + math.exp(-4 * (val(h) - val(a) + 0.03)))  # +0.03 = home-field edge
    home_pick = ph >= 0.5
    p = ph if home_pick else 1 - ph
    (pt, pr), (ot, orr) = ((hid, h), (aid, a)) if home_pick else ((aid, a), (hid, h))
    pa, oa, sg = nabbr(pt), nabbr(ot), (lambda x: f"{x:+.1f}")
    reasons = [f"Season record: {pa} {pr['rec']}, {oa} {orr['rec']}",
               f"Last 5 games: {pa} {pr['l5']}, {oa} {orr['l5']}",
               f"Average scoring margin, last 5: {pa} {sg(pr['margin'])}, {oa} {sg(orr['margin'])}",
               f"{'Home' if home_pick else 'Road'} record for {pa}: {pr['locrec']}. {'Road' if home_pick else 'Home'} record for {oa}: {orr['locrec']}"]
    return {"team_id": pt, "team": pa, "name": NFL_TEAMS[pt][1], "prob": round(100 * p), "other_prob": round(100 * (1 - p)), "other": oa,
            "label": "Strong lean" if p >= 0.70 else "Lean" if p >= 0.60 else "Close game", "reasons": reasons}


def nfresh(games, today):
    try:
        return (today - datetime.strptime(games[0][0], "%Y-%m-%d").date()).days <= 21
    except Exception:
        return False


_ncc = {}


def ncands_for(d, w):
    key = (d["built"], w)
    if key in _ncc:
        return _ncc[key]
    out, today = [], et_now().date()
    for pid, games in d["logs"].items():
        m = d["meta"].get(pid)
        if not m or len(games) < w + 4 or not nfresh(games, today):
            continue
        recent, base = games[:w], games[w:w + 12]
        for stat, (label, lines, f, poss) in NSTATS.items():
            if m["pos"] not in poss:
                continue
            rv, bv, best = [f(g) for g in recent], [f(g) for g in base], None
            for line in lines:
                hits = sum(v >= line for v in rv)
                if hits < HOT_MIN[w]:
                    continue
                b, r = sum(v >= line for v in bv) / len(bv), hits / w
                if b > MAX_BASE or r - b < MIN_LIFT:
                    continue
                if best is None or (r - b, line) > best[0]:
                    best = ((r - b, line), line, hits, b, r)
            if best:
                _, line, hits, b, r = best
                out.append({"id": pid, "name": m["name"], "pos": m["pos"], "team_id": m["team_id"], "stat": stat, "line": line,
                            "text": f"Over {line - 0.5:g} {NLBL[stat]}", "vals": rv, "hits": hits, "base": round(100 * b),
                            "p": round(0.6 * r + 0.4 * b, 3), "lift": round(100 * (r - b))})
    out.sort(key=lambda c: (-c["p"], -c["lift"]))
    if len(_ncc) > 6:
        _ncc.clear()
    _ncc[key] = out
    return out


def nrising_for(d, w, playing):
    res, today = [], et_now().date()
    for pid, games in d["logs"].items():
        m = d["meta"].get(pid)
        if not m or m["team_id"] not in playing or len(games) < w + 4 or not nfresh(games, today):
            continue
        recent, base, best = games[:w], games[w:w + 10], None
        for stat, (label, lines, f, poss) in NSTATS.items():
            if m["pos"] not in poss:
                continue
            ar, ab = sum(f(g) for g in recent) / w, sum(f(g) for g in base) / len(base)
            delta = ar - ab
            if ab <= 0 or delta < N_DELTA[stat] or delta / ab < 0.15:
                continue
            score = delta / max(ab, 1)
            if best is None or score > best[0]:
                best = (score, stat, ar, ab, delta)
        if best:
            score, stat, ar, ab, delta = best
            f, p = NSTATS[stat][2], playing[m["team_id"]]
            res.append({"id": pid, "name": m["name"], "pos": m["pos"], "team": nabbr(m["team_id"]), "opp": p["opp"], "where": p["where"],
                        "stat": stat, "label": NSTATS[stat][0], "w": w, "series": [f(g) for g in reversed(games[:w + 10])],
                        "avg_recent": round(ar, 1), "avg_base": round(ab, 1), "delta": round(delta, 1), "score": score})
    res.sort(key=lambda c: -c["score"])
    return res[:8]


def nfl_today_info(d, w):
    sl, playing, gl = nfl_slate(), {}, []
    for e in sl["events"]:
        h, a, t = e["home"], e["away"], et_label(e["date"])
        playing[h["id"]] = {"opp": a["abbr"], "where": "vs", "time": t}
        playing[a["id"]] = {"opp": h["abbr"], "where": "@", "time": t}
        pk = nfl_ml(d, h["id"], a["id"])
        gl.append({"gid": e["eid"], "home": h["abbr"], "away": a["abbr"], "home_id": h["id"], "away_id": a["id"],
                   "home_logo": nlogo(h["abbr"]), "away_logo": nlogo(a["abbr"]), "time": t or e["detail"],
                   "pick": {k: pk[k] for k in ("team", "prob", "label")} if pk else None})
    strict = [c for c in ncands_for(d, w) if c["team_id"] in playing]
    steady = team_best(d["logs"], d["meta"], set(playing), w, NSTATS, fresh_fn=lambda g: nfresh(g, et_now().date()))
    return gl, with_backup(strict, steady, playing, nabbr), playing


# ---- NFL API ----
@app.route("/api/nfl/teams")
def api_nfl_teams():
    return jsonify([{"id": i, "name": t[1], "nick": t[2], "abbr": t[0], "city": t[3], "logo": nlogo(t[0])}
                    for i, t in sorted(NFL_TEAMS.items(), key=lambda x: x[1][1])])


@app.route("/api/nfl/today")
def api_nfl_today():
    d, resp = nfl_data()
    if resp:
        return resp
    gl, c5, playing = nfl_today_info(d, 5)
    c10 = nfl_today_info(d, 10)[1]
    return jsonify(label=_nslate["label"], games=gl, cands={"5": c5[:60], "10": c10[:60]},
                   rising={"5": nrising_for(d, 5, playing), "10": nrising_for(d, 10, playing)}, next_refresh_in=secs_to_midnight())


@app.route("/api/nfl/parlay")
def api_nfl_parlay():
    d, resp = nfl_data()
    if resp:
        return resp
    n, variant = max(1, min(10, int(request.args.get("n", 5)))), int(request.args.get("variant", 0))
    return make_parlay(nfl_today_info(d, 5)[1], n, variant)


def usage(gs):
    return sum(g[3] * 0.25 + g[5] + g[8] + 8 * (g[4] + g[6] + g[9]) for g in gs) / max(1, len(gs))


@app.route("/api/nfl/team_parlay")
def api_nfl_team_parlay():
    d, resp = nfl_data()
    if resp:
        return resp
    tid, w = int(request.args["team_id"]), wparam()
    n, variant = max(1, min(10, int(request.args.get("n", 5)))), int(request.args.get("variant", 0))
    strict = [c for c in ncands_for(d, w) if c["team_id"] == tid]
    steady = team_best(d["logs"], d["meta"], tid, w, NSTATS, fresh_fn=lambda g: nfresh(g, et_now().date()))
    blank = {tid: {"opp": "", "where": "", "time": ""}}
    return make_parlay(with_backup(strict, steady, blank, nabbr), n, variant, cap=99)


@app.route("/api/nfl/game")
def api_nfl_game():
    d, resp = nfl_data()
    if resp:
        return resp
    e = next((x for x in nfl_slate()["events"] if x["eid"] == request.args.get("gid", "")), None)
    if not e:
        return jsonify(error="That game isn't on the schedule anymore.")
    today, sides = et_now().date(), []
    for t, where in ((e["away"], "Away"), (e["home"], "Home")):
        pl = [{"id": pid, "name": m["name"], "pos": m["pos"], "games": d["logs"][pid][:10]}
              for pid, m in d["meta"].items() if m["team_id"] == t["id"] and nfresh(d["logs"][pid], today)]
        pl.sort(key=lambda p: -usage(p["games"][:5]))
        sides.append({"id": t["id"], "name": NFL_TEAMS[t["id"]][1], "abbr": t["abbr"], "logo": nlogo(t["abbr"]), "where": where,
                      "record": rec_str(d["recs"][t["id"]]), "players": pl[:14]})
    return jsonify(time=et_label(e["date"]) or e["detail"], pick=nfl_ml(d, e["home"]["id"], e["away"]["id"]), sides=sides)


@app.route("/api/nfl/props")
def api_nfl_props():
    d, resp = nfl_data()
    if resp:
        return resp
    tid, stat, line, w = int(request.args["team_id"]), request.args["stat"], int(request.args["line"]), wparam()
    label, lines, f, poss = NSTATS[stat]
    rows, today = [], et_now().date()
    for pid, m in d["meta"].items():
        games = d["logs"].get(pid, [])[:w]
        if m["team_id"] != tid or m["pos"] not in poss or len(games) < min(w, 3) or not nfresh(games, today):
            continue
        vals = [f(g) for g in games]
        hits = sum(v >= line for v in vals)
        rows.append({"id": pid, "name": m["name"], "pos": m["pos"], "vals": vals, "hits": hits, "n": len(vals), "pct": round(100 * hits / len(vals))})
    rows.sort(key=lambda r: (-r["pct"], -sum(r["vals"])))
    return jsonify(rows=rows)


@app.route("/api/nfl/search")
def api_nfl_search():
    d, q = ND["data"], request.args.get("q", "").strip().lower()
    if not d or not q:
        return jsonify([])
    hits = [(pid, m) for pid, m in d["meta"].items() if q in m["name"].lower()]
    hits.sort(key=lambda x: (not any(w.startswith(q) for w in x[1]["name"].lower().split()), not x[1]["name"].lower().startswith(q), x[1]["name"]))
    return jsonify([{"id": pid, "name": m["name"], "pos": m["pos"], "team": nabbr(m["team_id"])} for pid, m in hits[:6]])


@app.route("/api/nfl/player")
def api_nfl_player():
    d, resp = nfl_data()
    if resp:
        return resp
    pid = request.args.get("id", type=int)
    m = d["meta"].get(pid)
    if not m:
        return jsonify(error="No current player found.")
    return jsonify(id=pid, name=m["name"], pos=m["pos"], team=NFL_TEAMS.get(m["team_id"], ("", "Free agent"))[1], games=d["logs"][pid][:10])


@app.route("/api/nfl/standings")
def api_nfl_standings():
    d, resp = nfl_data()
    if resp:
        return resp
    conf = {"AFC": [], "NFC": []}
    for tid, t in NFL_TEAMS.items():
        r, g = d["recs"][tid], 0
        g = r["w"] + r["l"] + r["t"]
        conf[t[4]].append(((r["w"] + r["t"] / 2) / g if g else 0.0, r["pf"] - r["pa"], tid, r))
    out = {}
    for c, rows in conf.items():
        rows.sort(key=lambda x: (-x[0], -x[1]))
        out[c] = []
        for i, (pct, diff, tid, r) in enumerate(rows):
            last, res = r["res"][-5:], r["res"]
            lw = sum(x[0] for x in last)
            streak = ""
            if res:
                k = res[-1][0]
                n = 0
                for x in reversed(res):
                    if x[0] != k:
                        break
                    n += 1
                streak = ("W" if k == 1 else "L" if k == 0 else "T") + str(n)
            t = NFL_TEAMS[tid]
            out[c].append({"rank": i + 1, "id": tid, "nick": t[2], "abbr": t[0], "logo": nlogo(t[0]), "rec": rec_str(r),
                           "pct": (f"{pct:.3f}".lstrip("0") if pct < 1 else "1.000") if pct else ".000", "diff": f"{diff:+d}",
                           "l5": f"{int(lw)}-{int(len(last) - lw)}", "streak": streak})
    return jsonify(afc=out["AFC"], nfc=out["NFC"])


_nlive = {"t": 0, "out": None}


@app.route("/api/nfl/live")
def api_nfl_live():
    if _nlive["out"] is None or time.time() - _nlive["t"] > 15:
        try:
            games = []
            for e in espn("/scoreboard").get("events", []):
                ev = norm_event(e)
                st = {"pre": 1, "in": 2, "post": 3}.get(ev["state"], 1)
                games.append({"status": st, "text": ev["detail"] if st != 1 else (et_label(ev["date"]) or ev["detail"]),
                              "home": ev["home"]["abbr"], "away": ev["away"]["abbr"], "home_id": ev["home"]["id"], "away_id": ev["away"]["id"],
                              "home_logo": nlogo(ev["home"]["abbr"]), "away_logo": nlogo(ev["away"]["abbr"]),
                              "home_score": ev["home"]["score"], "away_score": ev["away"]["score"]})
            _nlive.update(t=time.time(), out={"games": games})
        except Exception as ex:
            print("  nfl live scores failed:", ex, flush=True)
            if _nlive["out"] is None:
                return jsonify(error="Live scores are unavailable right now.")
    return jsonify(_nlive["out"])


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/")
def index():
    return PAGE


# =============================== page ===============================
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prop Helper</title>
<meta name="description" content="NBA and NFL player prop hit rates, hot streaks and bet builder.">
<meta name="theme-color" content="#0b1020">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='46' fill='%23f97316'/%3E%3C/svg%3E">
<script>document.documentElement.dataset.theme=localStorage.getItem('theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light')</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#EEF1F6;--card:#fff;--soft:#F4F6FA;--ink:#0F1B2D;--mute:#5F6B7E;--line:#D9DFE9;--hero:#12284A;--org:#F26B21;--orgtext:#C24E0C;--hit:#0B6B33;--hitbg:#D9F3E3;--miss:#9B2323;--missbg:#FBE1E1;--amb:#8A6200;--ambbg:#FFF0C7;--ambtx:#6b4c00;--s-hi:#1DB35B;--s-mid:#E5A800;--s-lo:#D64545;--sh:0 1px 2px rgba(15,27,45,.06),0 6px 18px rgba(15,27,45,.06);--live:#E5322D}
[data-theme=dark]{--bg:#0B1220;--card:#141E33;--soft:#101A2C;--ink:#E8EDF6;--mute:#93A0B8;--line:#25324D;--hero:#1B3866;--org:#FF7A33;--orgtext:#FF9A5C;--hit:#55DB8E;--hitbg:#12382A;--miss:#FF9090;--missbg:#442029;--amb:#F2C14E;--ambbg:#3f3414;--ambtx:#F2C14E;--sh:0 0 0 1px rgba(255,255,255,.04),0 8px 22px rgba(0,0,0,.35)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
h1,h2,h3,.d{font-family:"Barlow Condensed","Arial Narrow",sans-serif}
button,input{font:inherit}
button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid var(--org);outline-offset:2px}
.wrap{max-width:1000px;margin:0 auto}
.hide{display:none!important}
/* header */
header{background:var(--hero);color:#fff;padding:12px 16px 0;position:relative;z-index:30}
@media(min-width:900px){header{position:sticky;top:0}}
.top{display:flex;align-items:center;gap:12px}
.brand{font:800 30px "Barlow Condensed",sans-serif;letter-spacing:.01em}
.brand i{color:var(--org);font-style:normal}
.sports{display:flex;gap:4px;background:rgba(255,255,255,.12);border-radius:12px;padding:3px}
.sp{border:0;background:none;color:#cfd9ea;font:800 20px "Barlow Condensed",sans-serif;padding:5px 18px;border-radius:9px;cursor:pointer}
.sp.on{background:#fff;color:#12284A}
.ib{margin-left:auto;background:rgba(255,255,255,.12);color:#fff;border:0;border-radius:10px;padding:8px 12px;font-size:14px;cursor:pointer}
.sbar{position:relative;margin-top:12px}
.sbar input{width:100%;padding:14px 16px 14px 44px;border:0;border-radius:14px;font-size:17px;background:#fff;color:#0F1B2D;box-shadow:0 2px 10px rgba(0,0,0,.15)}
[data-theme=dark] .sbar input{background:#0B1220;color:#E8EDF6}
.sbar svg{position:absolute;left:15px;top:50%;transform:translateY(-50%);width:20px;height:20px;stroke:#5F6B7E;fill:none;stroke-width:2.2;stroke-linecap:round}
.sugbox{position:absolute;left:0;right:0;top:calc(100% + 6px);background:var(--card);color:var(--ink);border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.28);max-height:min(70vh,520px);overflow:auto;padding:6px}
.grp{padding:8px 12px 4px;font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--mute)}
.si{display:flex;align-items:center;gap:12px;width:100%;text-align:left;background:none;border:0;border-radius:12px;padding:8px 10px;cursor:pointer;color:var(--ink)}
.si:hover,.si:focus-visible{background:var(--soft)}
.si img{width:40px;height:40px;border-radius:50%;object-fit:cover;object-position:top;background:var(--line)}
.si img.lg{border-radius:0;background:none;object-fit:contain}
.si b{display:block;line-height:1.2}.si span{font-size:13px;color:var(--mute)}
.tabs{display:flex;gap:2px;margin-top:6px;overflow-x:auto}
.tab{background:none;border:0;color:#b9c6dc;font:700 21px "Barlow Condensed",sans-serif;padding:10px 16px;cursor:pointer;border-bottom:4px solid transparent;white-space:nowrap}
.tab.on{color:#fff;border-color:var(--org)}
.boot{position:sticky;top:0;z-index:40;background:var(--org);color:#fff;text-align:center;padding:10px;font-weight:600}
.boot small{opacity:.9;font-weight:400}
main{padding:20px 16px 110px}
#say{min-height:4px;margin:0 2px 8px;font-size:14px;color:var(--mute)}
#say.err,.err{color:var(--miss)}
/* sections */
.step{margin-top:26px}
.step h2,.h2{margin:0 0 4px;font-size:28px;font-weight:700;display:flex;align-items:center;gap:10px}
.step h2::before{content:"";width:5px;height:24px;border-radius:3px;background:var(--org)}
.step p,.note{margin:0 0 12px;color:var(--mute);font-size:14px}
.hero2{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:6px}
.hero2 h2{margin:0;font-size:40px;font-weight:800;line-height:1}
.hero2 p{margin:4px 0 0;color:var(--mute);font-size:14px}
.clock{background:var(--hero);color:#fff;border-radius:999px;padding:7px 16px;font:700 19px "Barlow Condensed",sans-serif}
.seg{display:inline-flex;background:var(--card);box-shadow:var(--sh);border-radius:14px;padding:4px;flex-wrap:wrap;gap:4px}
.seg button{border:0;background:none;border-radius:10px;padding:9px 16px;font:700 20px "Barlow Condensed",sans-serif;color:var(--mute);cursor:pointer}
.seg button.on{background:var(--hero);color:#fff}
.teams{display:grid;grid-template-columns:repeat(auto-fill,minmax(76px,1fr));gap:8px}
.t{background:var(--card);box-shadow:var(--sh);border:2px solid transparent;border-radius:14px;padding:9px 2px 7px;cursor:pointer;display:flex;flex-direction:column;align-items:center;gap:2px;font:700 15px "Barlow Condensed",sans-serif;color:var(--ink)}
.t img{width:42px;height:42px}.t.on{border-color:var(--org);background:var(--soft)}
.chips{display:flex;flex-wrap:wrap;gap:10px}
.chip{min-width:80px;padding:10px 18px;border-radius:14px;border:2px solid var(--line);background:var(--card);box-shadow:var(--sh);color:var(--ink);font:800 28px "Barlow Condensed",sans-serif;cursor:pointer}
.chip.on{background:var(--org);border-color:var(--org);color:#fff}
.bigseg button{font-size:24px;padding:12px 22px}
/* game tiles */
.slate{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-top:14px}
.gm{background:var(--card);box-shadow:var(--sh);border:2px solid transparent;border-radius:18px;padding:14px;text-align:left;color:var(--ink);display:flex;flex-direction:column;gap:6px}
button.gm{cursor:pointer}button.gm:hover{border-color:var(--org)}
.gm .tm{display:flex;align-items:center;gap:10px;font:700 24px "Barlow Condensed",sans-serif;line-height:1}
.gm .tm img{width:32px;height:32px}
.gm .tm small{margin-left:auto;font:600 13px system-ui,sans-serif;color:var(--mute)}
.gm .pk{font-size:13px;font-weight:600;color:var(--orgtext);margin-top:2px}
.mini{height:6px;border-radius:3px;background:var(--line);overflow:hidden}.mini i{display:block;height:100%;background:var(--org)}
/* bet builder */
.bet{background:linear-gradient(135deg,var(--hero),#0d1d38);color:#fff;border-radius:22px;padding:22px;margin-top:26px;box-shadow:var(--sh)}
[data-theme=dark] .bet{background:linear-gradient(135deg,#1B3866,#142a4d)}
.bet h2{margin:0 0 4px;font-size:34px;font-weight:800}
.bet>p{margin:0 0 14px;color:#c4d0e4;font-size:14px}
.ncs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.nc{width:46px;height:46px;border-radius:12px;border:2px solid rgba(255,255,255,.3);background:none;color:#fff;font:800 24px "Barlow Condensed",sans-serif;cursor:pointer}
.nc.on{background:var(--org);border-color:var(--org)}
.go,.again,.retry{background:var(--org);color:#fff;border:0;border-radius:14px;padding:14px 22px;font:800 24px "Barlow Condensed",sans-serif;cursor:pointer}
.go{width:100%}.again{background:none;border:2px solid rgba(255,255,255,.3);margin-top:6px;font-size:20px}.retry{margin-top:10px}
#bet{margin-top:16px}.bet .row{color:var(--ink)}
.sum2{display:flex;flex-direction:column;margin-bottom:10px}.sum2 b.d{font-size:30px}.sum2 span{color:#c4d0e4}.sum2 span b{color:#fff}
.fine{color:#c4d0e4;font-size:13px;margin:12px 0 0}
/* rising */
.rgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px}
.rc{background:var(--card);box-shadow:var(--sh);border-radius:18px;padding:14px}
.top2{display:flex;align-items:center;gap:10px}
.top2 b{display:block;line-height:1.2}.top2 span{color:var(--mute);font-size:13px}
.up{margin-left:auto;text-align:right;color:var(--hit);font:800 30px "Barlow Condensed",sans-serif;line-height:1}
.up small{display:block;font:600 12px system-ui,sans-serif;color:var(--mute)}
.rc svg{width:100%;height:auto;margin-top:6px}
.br{fill:var(--org)}.bb{fill:var(--mute);opacity:.4}
.bl{fill:var(--ink);font:700 11px system-ui,sans-serif}.bl.sm{font-size:9px}
.lb{stroke:var(--ink);stroke-width:2;stroke-dasharray:5 4;opacity:.7}
.go2{margin-top:8px;font-size:12px;color:var(--orgtext);font-weight:600}
.rc{cursor:pointer;border:2px solid transparent}.rc:hover,.rc:focus-visible{border-color:var(--org)}
.cap{display:flex;justify-content:space-between;font-size:13px;color:var(--mute)}
.sw{display:inline-block;width:14px;height:4px;border-radius:2px;margin-right:5px;vertical-align:middle;background:var(--mute)}.sw.o{background:var(--org)}
/* rows */
.sum{margin:2px 0 8px;font-size:17px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;color:var(--mute);font-size:14px;margin-bottom:14px}
.legend .g{display:inline-block;padding:2px 10px;font-size:16px}
.row{display:grid;grid-template-columns:56px 1fr auto;gap:8px 12px;align-items:center;background:var(--card);box-shadow:var(--sh);border-radius:18px;padding:12px;margin-bottom:10px;border-left:6px solid var(--line)}
.row.hi{border-left-color:var(--s-hi)}.row.mid{border-left-color:var(--s-mid)}.row.lo{border-left-color:var(--s-lo)}
.face{width:56px;height:56px;border-radius:50%;object-fit:cover;object-position:top;background:var(--line)}
.face.s{width:44px;height:44px}
.who b{display:block;font-size:18px;line-height:1.2}
.who span{display:block;color:var(--mute);font-size:14px}.who .why{color:var(--hit);font-weight:600}.who strong{color:var(--orgtext)}
.pct{font:800 40px "Barlow Condensed",sans-serif;line-height:1}
.hi .pct{color:var(--hit)}.mid .pct{color:var(--amb)}.lo .pct{color:var(--miss)}
.games{grid-column:1/-1;display:grid;gap:5px}
.g{border-radius:8px;text-align:center;padding:4px 0 2px;font:700 22px "Barlow Condensed",sans-serif;line-height:1.1}
.games.ten .g{font-size:17px}
.g small{display:block;font:600 11px system-ui,sans-serif}
.g.h{background:var(--hitbg);color:var(--hit)}.g.m{background:var(--missbg);color:var(--miss)}
@media(min-width:760px){.row{grid-template-columns:56px 1.1fr 84px 2fr}.games{grid-column:auto}}
.load,.empty{background:var(--card);box-shadow:var(--sh);border-radius:16px;padding:22px;text-align:center;color:var(--mute)}
.spin{display:inline-block;width:18px;height:18px;border:3px solid var(--line);border-top-color:var(--org);border-radius:50%;animation:s .8s linear infinite;vertical-align:-3px;margin-right:8px}
.boot .spin{border-color:rgba(255,255,255,.4);border-top-color:#fff}
@keyframes s{to{transform:rotate(360deg)}}@keyframes p{50%{opacity:.35}}
@media(prefers-reduced-motion:reduce){.spin,.dot.on{animation:none}}
/* player + tables */
.hero{display:flex;gap:14px;align-items:center;margin:8px 0}
.hero .face{width:88px;height:88px}.hero h2{margin:0;font-size:38px;font-weight:800;line-height:1}.hero span{color:var(--mute)}
.avl{margin:14px 0 4px;font-size:13px;font-weight:700;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
.avgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(84px,1fr));gap:8px}
.big{background:var(--card);box-shadow:var(--sh);border-radius:14px;padding:10px;text-align:center;color:var(--mute);font-size:13px}
.big b{display:block;font:800 32px "Barlow Condensed",sans-serif;color:var(--ink);line-height:1.1}
.tbl{background:var(--card);box-shadow:var(--sh);border-radius:16px;overflow-x:auto;margin:14px 0}
table{width:100%;border-collapse:collapse;min-width:420px}
th,td{padding:10px;text-align:right;border-bottom:1px solid var(--line);font-size:15px}
tr:last-child td{border-bottom:0}
th{color:var(--mute);font-size:13px;font-weight:600}
th:first-child,td:first-child{text-align:left}
td.n{font:700 21px "Barlow Condensed",sans-serif}
.W{color:var(--hit);font-weight:700}.L{color:var(--miss);font-weight:700}
.st h3{margin:0;padding:14px 14px 4px;font-size:28px}
.st td:nth-child(2),.st th:nth-child(2){text-align:left}
.st tr.po td:first-child{box-shadow:inset 4px 0 var(--s-hi)}.st tr.pi td:first-child{box-shadow:inset 4px 0 var(--s-mid)}
.st img{width:24px;height:24px;vertical-align:middle;margin-right:8px}
.back{background:none;border:0;color:var(--orgtext);font-weight:700;cursor:pointer;padding:0;font-size:16px}
.gvh{display:flex;align-items:center;justify-content:center;gap:18px;margin:14px 0 4px;font:800 40px "Barlow Condensed",sans-serif;flex-wrap:wrap}
.gvh img{width:60px;height:60px;vertical-align:middle}.gvh small{display:block;font:500 14px system-ui,sans-serif;color:var(--mute);text-align:center}
.ml{background:linear-gradient(135deg,var(--hero),#0d1d38);color:#fff;border-radius:22px;padding:20px;box-shadow:var(--sh)}
[data-theme=dark] .ml{background:linear-gradient(135deg,#1B3866,#142a4d)}
.ml h3{margin:0;font-size:32px;font-weight:800}.ml p{margin:2px 0 12px;color:#c4d0e4;font-size:14px}
.pbar{display:flex;height:38px;border-radius:12px;overflow:hidden;font:800 21px "Barlow Condensed",sans-serif}
.pbar div{display:flex;align-items:center;padding:0 12px;white-space:nowrap}
.pbar .a{background:var(--org);color:#fff}.pbar .b{background:rgba(255,255,255,.2);justify-content:flex-end;flex:1}
.ml ul{margin:14px 0 0;padding-left:20px;font-size:14px;color:#e4ebf7}
.pl{background:var(--card);box-shadow:var(--sh);border-radius:16px;margin-bottom:8px;overflow:hidden}
.plh{display:flex;align-items:center;gap:12px;width:100%;background:none;border:0;padding:10px 12px;text-align:left;cursor:pointer;color:var(--ink)}
.plh .nm b{display:block}.plh .nm span{font-size:13px;color:var(--mute)}.plh .ch{margin-left:auto;color:var(--mute)}
.pld{padding:4px 12px 14px}
.hrs{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 6px;align-items:center}
.hrs b{width:100%;font-size:13px;color:var(--mute);margin-top:8px}
.hc{padding:3px 10px;border-radius:999px;font-size:14px;font-weight:600}
.hc.hi{background:var(--hitbg);color:var(--hit)}.hc.mid{background:var(--ambbg);color:var(--ambtx)}.hc.lo{background:var(--missbg);color:var(--miss)}
.soon{text-align:center;background:var(--card);box-shadow:var(--sh);border-radius:24px;padding:60px 20px;margin-top:10px}
.soon .em{font-size:64px}.soon h2{font-size:44px;margin:6px 0}.soon p{color:var(--mute);margin:0}
.ctl{display:flex;flex-wrap:wrap;gap:10px}
.stgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}
.stgrid .tbl{margin:14px 0 0}.st table{min-width:0}.st .ab{display:none}
@media(max-width:899px){.st .opt,.st .fn,.st .cf{display:none}.st .ab{display:inline}.st th,.st td{padding:8px 4px;font-size:14px}.st img{width:20px;height:20px;margin-right:5px}.st h3{font-size:24px;padding:12px 8px 4px}}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;background:var(--soft);font:700 12px system-ui,sans-serif;font-style:normal;color:var(--mute);margin-left:8px;vertical-align:middle}
/* live box score window */
.lmodal{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:60;display:flex;align-items:flex-start;justify-content:center;padding:16px;overflow:auto}
.lbox{background:var(--bg);color:var(--ink);border-radius:20px;padding:16px;width:min(760px,100%);margin:auto}
.lclose{background:var(--hero);color:#fff;border:0;border-radius:10px;padding:8px 14px;cursor:pointer;font-weight:700}
.lmh{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;margin:12px 0 4px;color:var(--mute);font-size:13px}.lmh .lv{color:var(--live);font-weight:700}
.lbox h3{margin:10px 0 0;font-size:30px}.lbox h3 b{color:var(--orgtext)}
.lbox td:first-child{white-space:nowrap}.face.xs{width:26px;height:26px;margin-right:8px;vertical-align:middle}
tr.oc td:first-child{box-shadow:inset 4px 0 var(--s-hi)}.onc{font-size:11px;color:var(--hit);font-style:normal;font-weight:700}
.clk{cursor:pointer}.lg.clk:hover{outline:2px solid var(--org)}.lmore{font-size:12px;color:var(--orgtext);font-weight:600;padding:4px 2px 0}
/* live corner panel */
.live{position:fixed;right:14px;bottom:14px;width:min(330px,calc(100vw - 28px));background:var(--card);box-shadow:0 10px 40px rgba(0,0,0,.3);border:1px solid var(--line);border-radius:18px;z-index:35;overflow:hidden}
.lh{display:flex;align-items:center;gap:8px;width:100%;background:var(--hero);color:#fff;border:0;padding:11px 14px;cursor:pointer;font:800 20px "Barlow Condensed",sans-serif;letter-spacing:.04em;text-align:left}
.dot{width:10px;height:10px;border-radius:50%;background:#7b8ba6}.dot.on{background:var(--live);animation:p 1.4s infinite}
.lh .cnt{margin-left:auto;font:600 13px system-ui,sans-serif;opacity:.9}
.lb2{max-height:min(55vh,420px);overflow:auto;padding:8px}
.lg{padding:10px 10px 8px;border-radius:12px;background:var(--soft);margin-bottom:8px}
.lt{display:flex;justify-content:space-between;font-size:12px;font-weight:700;color:var(--mute);margin-bottom:4px}
.lt .lv{color:var(--live)}
.ls{display:flex;align-items:center;gap:8px;font:700 22px "Barlow Condensed",sans-serif}
.ls img{width:24px;height:24px}.ls b{margin-left:auto;font-size:26px}.ls.lead{color:var(--ink)}.ls.trail{color:var(--mute)}
.lnote{font-size:13px;color:var(--mute);padding:4px 6px 8px}
.lf{font-size:13px;color:var(--mute);display:flex;justify-content:space-between;padding:4px 6px}
.lf b{color:var(--ink)}
</style></head><body>
<header><div class="wrap">
<div class="top"><div class="brand">Prop<i>Helper</i></div>
<nav class="sports"><button class="sp on" data-sp="nba">NBA</button><button class="sp" data-sp="nfl">NFL</button></nav>
<button class="ib" id="theme"></button></div>
<div class="sbar" id="sbar"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg><input id="q" autocomplete="off" placeholder="Search a team or player…" aria-label="Search a team or player"><div class="sugbox hide" id="sug"></div></div>
<div class="tabs" id="ntabs"><button class="tab on" data-tab="today">Today</button><button class="tab" data-tab="standings">Standings</button><button class="tab" data-tab="team">Team props</button></div>
<div class="sbar hide" id="nf-sbar"></div>
<div class="tabs hide" id="nf-tabs"></div>
</div></header>
<div id="boot" class="boot hide"></div>
<main class="wrap">
<div id="say"></div>
<section id="nba">
  <section id="today">
    <div class="hero2"><div><h2 id="tlabel" class="d">Today</h2><p id="tsub"></p></div><div class="clock" id="clock"></div></div>
    <div class="note" style="margin-top:10px">Tap a game to see every player, their stats, and who I think wins.</div>
    <div class="slate" id="slate"></div>
    <div class="bet"><h2 class="d">Build a bet</h2><p>Choose how many players. I'll pick players from today's games who just started hitting a number they don't usually hit.</p>
      <div class="ncs" id="ncs"></div><button class="go" id="build">Build my bet</button><div id="bet"></div></div>
    <div class="step"><h2>Players on the rise</h2><p id="risenote"></p><div class="seg" id="risew"></div><div class="rgrid" id="rise" style="margin-top:12px"></div></div>
    <div class="step"><h2>Players on a hot streak</h2><p id="hotnote"></p><div class="ctl"><div class="seg" id="hotw"></div><div class="seg" id="tstats"></div></div><div id="hot" style="margin-top:12px"></div></div>
  </section>

  <section id="gameview" class="hide"><button class="back" id="back">← Back to today</button><div id="gv"></div></section>
  <section id="player" class="hide"><button class="back" id="back2">← Back</button><div id="pres"></div></section>
  <section id="standings" class="hide"><div id="stbody"></div></section>

  <section id="team" class="hide">
    <div class="step" style="margin-top:0"><h2>1. Pick a team</h2><div class="teams" id="teams"><div class="load"><i class="spin"></i>Loading teams…</div></div></div>
    <div class="bet hide" id="tbet"><h2 class="d" id="tbtitle">Build a bet</h2><p id="tbnote"></p><div class="ncs" id="tncs"></div><button class="go" id="tbuild">Build my bet</button><div id="tbres"></div></div>
    <div class="step"><h2>2. Pick a stat</h2><div class="seg" id="stats"></div></div>
    <div class="step hide" id="s3"><h2>3. How many recent games?</h2><div class="seg bigseg" id="gw"></div></div>
    <div class="step hide" id="s4"><h2>4. Pick a number</h2><p>Tap one. You'll see every player on the team and how often they reached at least that number.</p><div class="chips" id="lines"></div></div>
    <div id="res" style="margin-top:28px"></div>
  </section>

  <div class="lmodal hide" id="lmodal"><div class="lbox"><button class="lclose" id="lclose">✕ Close</button><div id="lmbody"></div></div></div>
  <aside class="live" id="live"><button class="lh" id="lhead" aria-expanded="true"><span class="dot" id="ldot"></span>LIVE<span class="cnt" id="lcnt"></span></button><div class="lb2" id="lbody"></div></aside>
</section>

<section id="nfl" class="hide"></section>
<footer style="text-align:center;opacity:.6;font-size:12px;padding:24px 12px">For information only, not betting advice. Stats come from unofficial public feeds and may be wrong. 21+ where legal. Help: 1-800-GAMBLER.</footer>
</main>
<script>
const STATS={points:{label:'Points',lines:[10,15,20,25,30,35],f:g=>g[4]},threes:{label:'3-pointers',lines:[1,2,3,4,5],f:g=>g[7]},rebounds:{label:'Rebounds',lines:[4,6,8,10,12],f:g=>g[5]},assists:{label:'Assists',lines:[2,4,6,8,10,12],f:g=>g[6]},pra:{label:'P+R+A',lines:[15,20,25,30,35,40],f:g=>g[4]+g[5]+g[6]}};
const ALIAS={point:'points',points:'points',pt:'points',pts:'points',three:'threes',threes:'threes','3':'threes','3s':'threes','3pt':'threes',rebound:'rebounds',rebounds:'rebounds',reb:'rebounds',rebs:'rebounds',assist:'assists',assists:'assists',ast:'assists',asts:'assists',pra:'pra','p+r+a':'pra'};
const $=s=>document.querySelector(s),$$=s=>document.querySelectorAll(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const face=id=>`https://cdn.nba.com/headshots/nba/latest/260x190/${id}.png`;
const logo=id=>`https://cdn.nba.com/logos/nba/${id}/global/L/logo.svg`;
const hide=`loading="lazy" decoding="async" onerror="this.style.visibility='hidden'"`;
const spin=t=>`<div class="load"><i class="spin"></i>${t}</div>`;
let ntab='today',TEAMS=[],team=null,stat=null,gw=null,line=null,TODAY=null,GAME=null,GSIDE=0;

async function api(u){
  const t0=Date.now();
  while(true){
    let r,d;
    try{r=await fetch(u)}catch(e){return{error:'Could not reach the app. Is it still running on your computer?'}}
    try{d=await r.json()}catch(e){return{error:'Something went wrong in the app. Look at the Command Prompt window for details.'}}
    if(!d.loading){$('#boot').classList.add('hide');return d}
    $('#boot').classList.remove('hide');
    $('#boot').innerHTML=`<i class="spin"></i>${esc(d.step||'Starting')}… ${Math.round((Date.now()-t0)/1000)}s <small>(only slow the first time. After this, everything is instant.)</small>`;
    await new Promise(r=>setTimeout(r,1000));
  }
}
function say(m,e){const s=$('#say');s.textContent=m||'';s.className=e?'err':''}

/* ---------- theme, sport, tabs ---------- */
function applyTheme(t){document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);$('#theme').textContent=t==='dark'?'☀ Light mode':'☾ Dark mode'}
$('#theme').onclick=()=>applyTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
applyTheme(document.documentElement.dataset.theme);
$$('.sp').forEach(b=>b.onclick=()=>{$$('.sp').forEach(x=>x.classList.toggle('on',x===b));const n=b.dataset.sp==='nba';
  ['#nba','#sbar','#ntabs'].forEach(s=>$(s).classList.toggle('hide',!n));
  ['#nfl','#nf-sbar','#nf-tabs'].forEach(s=>$(s).classList.toggle('hide',n));
  if(!n)NFL.start();say('')});
function tab(n){ntab=n;const hl=n==='gameview'?'today':n;$$('.tab').forEach(b=>b.classList.toggle('on',b.dataset.tab===hl));
  ['today','gameview','player','standings','team'].forEach(s=>$('#'+s).classList.toggle('hide',n!==s));
  if(n==='standings')loadStandings();window.scrollTo({top:0})}
$$('.tab').forEach(b=>b.onclick=()=>{tab(b.dataset.tab);say('')});
$('#back').onclick=()=>tab('today');$('#back2').onclick=()=>tab('today');

/* ---------- shared pieces ---------- */
function boxes(vals,ln){const n=vals.length;return `<div class="games ${n>5?'ten':''}" style="grid-template-columns:repeat(${n},1fr)">`+vals.map(v=>`<div class="g ${v>=ln?'h':'m'}">${v}<small>${v>=ln?'✓':'✗'}</small></div>`).join('')+'</div>'}
function pickRow(c,F=face){const n=c.vals.length;
  return `<div class="row ${c.hits>=n-1?'hi':'mid'}"><img class="face" src="${F(c.id)}" alt="" ${hide}><div class="who"><b>${esc(c.name)}</b><span><strong>${esc(c.text)}</strong> (${c.line}+)</span>${c.opp?`<span>${esc(c.team)} ${c.where} ${esc(c.opp)} • ${esc(c.time)}</span>`:`<span>${esc(c.team)}</span>`}<span class="why">${c.kind==='steady'?`Steady: ${c.hits} of last ${n}, ${c.base}% the games before`:`Heating up: ${c.hits} of last ${n}, was ${c.base}% before that`}</span></div><div class="pct">${c.hits}/${n}</div>${boxes(c.vals,c.line)}</div>`}
function gameChip(g,click){const pk=g.pick?`<span class="pk">Pick: ${esc(g.pick.team)} ${g.pick.prob}% • ${esc(g.pick.label)}</span><div class="mini"><i style="width:${g.pick.prob}%"></i></div>`:'';const T=click?'button':'div';
  return `<${T} class="gm" ${click?`data-gid="${g.gid}"`:''}><span class="tm"><img src="${g.away_logo||logo(g.away_id)}" alt="" ${hide}>${esc(g.away)}<small>${esc(g.time)}</small></span><span class="tm"><img src="${g.home_logo||logo(g.home_id)}" alt="" ${hide}>${esc(g.home)}<small>home</small></span>${pk}</${T}>`}

/* ---------- today (always uses each player's last 5 games) ---------- */
let betN=5,betV=0,hotStat='all',hotW=5,riseW=5,tTimer=0,lastLoad=0,lastT=0;
const HOTSTAT={all:'All',points:'Points',threes:'3-pointers',rebounds:'Rebounds',assists:'Assists',pra:'P+R+A'};
function wBtns(id,cur,fn){$(id).innerHTML=[5,10].map(n=>`<button class="${n===cur?'on':''}" data-w="${n}">Last ${n} games</button>`).join('');$$(id+' button').forEach(b=>b.onclick=()=>fn(+b.dataset.w))}
function riseCard(c,F=face){
  const s=c.series,n=s.length,Wd=340,H=150,top=20,bot=6,L=4,mx=Math.max(...s,c.avg_recent,c.avg_base,1),bw=(Wd-2*L)/n,gap=Math.min(6,bw*.28),sp=n-c.w;
  const y=v=>H-bot-(v/mx)*(H-top-bot);
  const bars=s.map((v,i)=>{const x=L+i*bw+gap/2,w=bw-gap,yy=y(v);
    return `<rect x="${x}" y="${yy}" width="${w}" height="${Math.max(1,H-bot-yy)}" rx="3" class="${i>=sp?'br':'bb'}"/><text x="${x+w/2}" y="${yy-4}" text-anchor="middle" class="bl${n>15?' sm':''}">${v}</text>`}).join('');
  return `<div class="rc" role="button" tabindex="0" data-id="${c.id}" aria-label="Open ${esc(c.name)}"><div class="top2"><img class="face s" src="${F(c.id)}" alt="" ${hide}><div><b>${esc(c.name)}</b><span>${esc(c.team)} ${c.where} ${esc(c.opp)}</span></div><div class="up">+${c.delta}<small>${esc(c.label)}</small></div></div>
  <svg viewBox="0 0 ${Wd} ${H}" role="img" aria-label="${esc(c.name)} ${esc(c.label)} per game, rising">${bars}<line class="lb" x1="${L}" x2="${Wd-L}" y1="${y(c.avg_base)}" y2="${y(c.avg_base)}"/></svg>
  <div class="cap"><span><i class="sw"></i>Before: ${c.avg_base} (gray, dashed)</span><span><i class="sw o"></i>Last ${c.w}: ${c.avg_recent}</span></div><div class="go2">Oldest game on the left, newest on the right. Tap for full stats →</div></div>`}
function drawRise(){
  wBtns('#risew',riseW,n=>{riseW=n;drawRise()});
  $('#risenote').textContent=`Their average over the last ${riseW} games compared to the games before that. Each bar is one game.`;
  const L=TODAY?(TODAY.rising[riseW]||[]):[];
  $('#rise').innerHTML=L.length?L.map(c=>riseCard(c)).join(''):'<div class="empty">No clear risers in today\'s games.</div>';
  $$('#rise .rc').forEach(el=>{el.onclick=()=>showPlayer(el.dataset.id);el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();showPlayer(el.dataset.id)}}})}
function drawHot(){
  wBtns('#hotw',hotW,n=>{hotW=n;drawHot()});
  $('#hotnote').textContent=`Heating up means they just started hitting a number they didn't hit before. Steady means they've hit it most of their last ${hotW} games. Heating up comes first.`;
  $('#tstats').innerHTML=Object.entries(HOTSTAT).map(([k,v])=>`<button class="${k===hotStat?'on':''}" data-k="${k}">${v}</button>`).join('');
  $$('#tstats button').forEach(b=>b.onclick=()=>{hotStat=b.dataset.k;drawHot()});
  const L=(TODAY?(TODAY.cands[hotW]||[]):[]).filter(c=>hotStat==='all'||c.stat===hotStat).slice(0,40);
  $('#hot').innerHTML=L.length?L.map(c=>pickRow(c)).join(''):'<div class="empty">Nobody fits right now. That can happen on a small slate.</div>'}
async function loadToday(){
  clearTimeout(tTimer);lastLoad=Date.now();
  if(!TODAY)$('#slate').innerHTML=spin('Loading…');
  const d=await api('/api/today?w=5');
  if(d.error){$('#slate').innerHTML=`<div class="empty err">${esc(d.error)}<br><button class="retry" onclick="loadToday()">Try again</button></div>`;return}
  TODAY=d;const n=d.games.length;
  $('#tlabel').textContent=n?d.label:(d.sched_error?d.label:'No games today');
  $('#tsub').textContent=n?n+' game'+(n>1?'s':'')+' today. Rebuilds by itself at midnight Eastern.':(d.sched_error?"I couldn't load today's schedule ("+d.sched_error+"). Check your internet connection, then refresh.":'No NBA games today. Rebuilds by itself at midnight Eastern.');
  $('#slate').innerHTML=d.games.map(g=>gameChip(g,true)).join('');
  $$('#slate .gm').forEach(b=>b.onclick=()=>openGame(b.dataset.gid));
  drawRise();drawHot();tTimer=setTimeout(loadToday,(d.next_refresh_in+15)*1000)}
function drawN(){$('#ncs').innerHTML=[1,2,3,4,5,6,7,8,9,10].map(n=>`<button class="nc ${n===betN?'on':''}" data-n="${n}">${n}</button>`).join('');
  $$('#ncs .nc').forEach(b=>b.onclick=()=>{betN=+b.dataset.n;betV=0;drawN();$('#bet').innerHTML=''})}
async function build(v){
  $('#bet').innerHTML=spin('Picking your players…');
  const d=await api(`/api/parlay?n=${betN}&variant=${v}&w=5`);
  if(d.error||!d.legs.length){$('#bet').innerHTML=`<div class="empty err">${esc(d.error||'No players fit today. That happens on small slates. Check back tomorrow.')}</div>`;return}
  const short=d.legs.length<betN?` I could only find ${d.legs.length} that fit today.`:'';
  $('#bet').innerHTML=`<div class="sum2"><b class="d">${d.legs.length}-player bet</b><span>Rough chance every leg hits: <b>${d.chance}%</b>${short}</span></div>${d.legs.map(c=>pickRow(c)).join('')}<button class="again" onclick="build(++betV)">Show me a different one</button><p class="fine">Based only on recent games. It doesn't know about injuries, matchups or sportsbook odds, so check that each player is playing and what line your sportsbook offers. More players means a lower chance, since every leg has to hit.</p>`}
$('#build').onclick=()=>build(betV=0);

/* ---------- game view ---------- */
const avg=(gs,f)=>gs.length?gs.reduce((a,x)=>a+f(x),0)/gs.length:0;
const one=x=>(Math.round(x*10)/10).toString();
async function openGame(gid){GAME=null;GSIDE=0;tab('gameview');$('#gv').innerHTML=spin('Loading game…');
  const d=await api('/api/game?gid='+gid);if(d.error){$('#gv').innerHTML=`<div class="empty err">${esc(d.error)}</div>`;return}GAME=d;drawGame()}
function hitChips(gs){const g5=gs.slice(0,5);return Object.values(STATS).map(s=>`<b>${s.label}: how often they reached each number (last 5 · last 10)</b>`+s.lines.map(l=>{const h5=g5.filter(g=>s.f(g)>=l).length,h10=gs.filter(g=>s.f(g)>=l).length,p=h5/g5.length;return `<span class="hc ${p>=.7?'hi':p>=.5?'mid':'lo'}">${l}+ &nbsp;${h5}/${g5.length} · ${h10}/${gs.length}</span>`}).join('')).join('')}
function drawGame(){
  if(!GAME)return;const d=GAME,A=d.sides[0],H=d.sides[1],p=d.pick;
  const pb=p?`<div class="ml"><h3 class="d">Money line: ${esc(p.name)}</h3><p>${esc(p.label)}. My estimate from records, recent form, scoring margin and home or road results.</p>
    <div class="pbar"><div class="a" style="width:${p.prob}%">${esc(p.team)} ${p.prob}%</div><div class="b">${esc(p.other)} ${p.other_prob}%</div></div>
    <ul>${p.reasons.map(r=>`<li>${esc(r)}</li>`).join('')}</ul><p style="margin:12px 0 0">It doesn't know about injuries or the betting odds. A big favorite may pay very little.</p></div>`:'<div class="empty">Not enough standings data for a money line pick yet.</div>';
  const side=d.sides[GSIDE];
  const list=side.players.map((pl,i)=>{const gs=pl.games.slice(0,5);
    return `<div class="pl"><button class="plh" data-i="${i}"><img class="face s" src="${face(pl.id)}" alt="" ${hide}><div class="nm"><b>${esc(pl.name)}</b><span>Last 5: PTS ${one(avg(gs,g=>g[4]))} • REB ${one(avg(gs,g=>g[5]))} • AST ${one(avg(gs,g=>g[6]))} • 3PM ${one(avg(gs,g=>g[7]))} • P+R+A ${one(avg(gs,g=>g[4]+g[5]+g[6]))} • MIN ${Math.round(avg(gs,g=>g[3]))}</span></div><span class="ch">▾</span></button><div class="pld hide" id="pd${i}"></div></div>`}).join('');
  $('#gv').innerHTML=`<div class="gvh"><span><img src="${logo(A.id)}" alt="" ${hide}> ${esc(A.abbr)}<small>${esc(A.record)}</small></span><span style="font-size:24px;color:var(--mute)">@</span><span><img src="${logo(H.id)}" alt="" ${hide}> ${esc(H.abbr)}<small>${esc(H.record)}</small></span></div><div style="text-align:center;color:var(--mute);margin:0 0 16px">${esc(d.time)}</div>${pb}
  <div class="step"><h2>Players</h2><p>Tap a player to see each of their last 10 games and how often they hit every number.</p>
  <div class="seg" id="gside">${d.sides.map((s,i)=>`<button class="${i===GSIDE?'on':''}" data-i="${i}">${esc(s.name)}</button>`).join('')}</div><div style="margin-top:12px">${list}</div></div>`;
  $$('#gside button').forEach(b=>b.onclick=()=>{GSIDE=+b.dataset.i;drawGame()});
  $$('.plh').forEach(b=>b.onclick=()=>{const i=+b.dataset.i,el=$('#pd'+i),open=el.classList.toggle('hide')===false;b.querySelector('.ch').textContent=open?'▴':'▾';
    if(open&&!el.innerHTML){const gs=side.players[i].games;
      el.innerHTML=`<div class="tbl"><table><tr><th>Date</th><th>Game</th><th>MIN</th><th>PTS</th><th>REB</th><th>AST</th><th>3PM</th><th>P+R+A</th></tr>${gs.map(g=>`<tr><td>${esc(g[0].slice(5))}</td><td>${esc(g[1])} <span class="${g[2]}">${g[2]}</span></td><td>${g[3]}</td><td class="n">${g[4]}</td><td class="n">${g[5]}</td><td class="n">${g[6]}</td><td class="n">${g[7]}</td><td class="n">${g[4]+g[5]+g[6]}</td></tr>`).join('')}</table></div><div class="hrs">${hitChips(gs)}</div>`}})}

/* ---------- standings: East on the left, West on the right ---------- */
async function loadStandings(){
  $('#stbody').innerHTML=spin('Loading standings…');
  const d=await api('/api/standings');if(d.error){$('#stbody').innerHTML=`<div class="empty err">${esc(d.error)}</div>`;return}
  const tbl=(nm,rows)=>`<div class="tbl st"><h3 class="d">${nm}<span class="cf"> Conference</span></h3><table><tr><th>#</th><th>Team</th><th>W-L</th><th class="opt">PCT</th><th>GB</th><th class="opt">L10</th><th class="opt">Strk</th></tr>${rows.map(r=>`<tr class="${r.rank<=6?'po':r.rank<=10?'pi':''}"><td>${r.rank}</td><td><img src="${logo(r.id)}" alt="" ${hide}><span class="fn">${esc(r.nick)}</span><span class="ab">${esc(r.abbr)}</span></td><td>${r.w}-${r.l}</td><td class="opt">${r.pct||'.000'}</td><td>${r.gb}</td><td class="opt">${esc(r.l10)}</td><td class="opt">${esc(r.streak)}</td></tr>`).join('')}</table></div>`;
  $('#stbody').innerHTML=`<div class="hero2"><div><h2 class="d">NBA standings</h2><p>Green stripe: top 6 go straight to the playoffs. Yellow stripe: spots 7 to 10 play in the play-in. GB means games behind the leader.</p></div></div>${d.east.length?`<div class="stgrid">${tbl('East',d.east)}${tbl('West',d.west)}</div>`:'<div class="empty">Standings are not available right now.</div>'}`}

/* ---------- team props: team, stat, how many games, number ---------- */
let tbN=5,tbV=0,tbTeam=null;
function drawTN(){$('#tncs').innerHTML=[1,2,3,4,5,6,7,8,9,10].map(n=>`<button class="nc ${n===tbN?'on':''}" data-n="${n}">${n}</button>`).join('');
  $$('#tncs .nc').forEach(b=>b.onclick=()=>{tbN=+b.dataset.n;tbV=0;drawTN();$('#tbres').innerHTML=''})}
async function tbuild(v){
  if(!team)return;$('#tbres').innerHTML=spin('Picking players from the '+esc(team.nick)+'…');
  const d=await api(`/api/team_parlay?team_id=${team.id}&n=${tbN}&variant=${v}&w=${gw||5}`);
  if(d.error||!d.legs.length){$('#tbres').innerHTML=`<div class="empty err">${esc(d.error||'No '+team.nick+' player has a reliable recent number right now. Try Last 10 games.')}</div>`;return}
  const short=d.legs.length<tbN?` Only ${d.legs.length} ${team.nick} players fit right now.`:'';
  $('#tbres').innerHTML=`<div class="sum2"><b class="d">${d.legs.length}-player ${esc(team.nick)} bet</b><span>Rough chance every leg hits: <b>${d.chance}%</b>${short}</span></div>${d.legs.map(c=>pickRow(c)).join('')}<button class="again" id="tbagain">Show me a different one</button><p class="fine">Based only on recent games. It doesn't know about injuries, matchups or sportsbook odds. More players means a lower chance, since every leg has to hit.</p>`;
  $('#tbagain').onclick=()=>tbuild(++tbV)}
$('#tbuild').onclick=()=>tbuild(tbV=0);
function drawFlow(){
  $('#tbet').classList.toggle('hide',!team);
  if(team){$('#tbtitle').textContent='Build a bet from the '+team.nick;$('#tbnote').textContent=`Only ${team.name} players. Uses each player's last ${gw||5} games${gw?'':' (pick Last 10 below to use 10)'}.`;
    if(tbTeam!==team.id){tbTeam=team.id;tbV=0;$('#tbres').innerHTML=''}}
  $('#teams').innerHTML=TEAMS.map(t=>`<button class="t ${team&&team.id===t.id?'on':''}" data-id="${t.id}" title="${esc(t.name)}"><img src="${logo(t.id)}" alt="" ${hide}>${esc(t.abbr)}</button>`).join('')||'<div class="empty err">Could not load teams. Refresh the page.</div>';
  $$('.t').forEach(b=>b.onclick=()=>{team=TEAMS.find(t=>t.id==b.dataset.id);line=null;say(team.name+' selected.');drawFlow()});
  $('#stats').innerHTML=Object.entries(STATS).map(([k,v])=>`<button class="${k===stat?'on':''}" data-k="${k}">${v.label}</button>`).join('');
  $$('#stats button').forEach(b=>b.onclick=()=>{stat=b.dataset.k;line=null;drawFlow()});
  const ready=!!(team&&stat);
  $('#s3').classList.toggle('hide',!ready);$('#s4').classList.toggle('hide',!(ready&&gw));
  $('#gw').innerHTML=[5,10].map(n=>`<button class="${n===gw?'on':''}" data-g="${n}">Last ${n} games</button>`).join('');
  $$('#gw button').forEach(b=>b.onclick=()=>{gw=+b.dataset.g;if(line)run(line);else drawFlow()});
  $('#lines').innerHTML=stat?STATS[stat].lines.map(n=>`<button class="chip ${n===line?'on':''}" data-n="${n}">${n}+</button>`).join(''):'';
  $$('.chip').forEach(b=>b.onclick=()=>run(+b.dataset.n));
  if(!(ready&&gw&&line))$('#res').innerHTML=''}
async function run(n){
  line=n;drawFlow();$('#res').innerHTML=spin('Checking every '+esc(team.nick)+' player…');
  const d=await api(`/api/props?team_id=${team.id}&stat=${stat}&line=${n}&w=${gw}`);
  if(d.error){$('#res').innerHTML=`<div class="empty err">${esc(d.error)}</div>`;return}
  if(!d.rows.length){$('#res').innerHTML='<div class="empty">No game data found for this team yet.</div>';return}
  const rows=d.rows.map(r=>{const k=r.pct>=70?'hi':r.pct>=50?'mid':'lo';
    return `<div class="row ${k}"><img class="face" src="${face(r.id)}" alt="" ${hide}><div class="who"><b>${esc(r.name)}</b><span>${r.hits} of ${r.n} games</span></div><div class="pct">${r.pct}%</div>${boxes(r.vals,n)}</div>`}).join('');
  $('#res').innerHTML=`<h2 class="d" style="margin:0;font-size:36px;font-weight:800">${esc(team.name)}</h2><div class="sum">Who reached <b>${n}+ ${STATS[stat].label}</b> in their last ${gw} games</div>
  <div class="legend"><span class="g h">✓ ${n}+</span><span class="g m">✗ under ${n}</span><span>Each box is one game, newest first. Best hit rate is at the top.</span></div>${rows}`}

/* ---------- player page ---------- */
async function showPlayer(id){
  tab('player');$('#pres').innerHTML=spin('Loading player…');
  const p=await api('/api/player?id='+id);
  if(p.error){$('#pres').innerHTML=`<div class="empty err">${esc(p.error)}</div>`;return}
  const rows=p.games.map(g=>`<tr><td>${esc(g.date.slice(5))}</td><td>${esc(g.matchup)} <span class="${g.wl}">${g.wl||''}</span></td><td>${g.min}</td><td class="n">${g.pts}</td><td class="n">${g.reb}</td><td class="n">${g.ast}</td><td class="n">${g.fg3}</td><td class="n">${g.pra}</td></tr>`).join('');
  const grid=a=>`<div class="avgs"><div class="big"><b>${a.pts}</b>points</div><div class="big"><b>${a.reb}</b>rebounds</div><div class="big"><b>${a.ast}</b>assists</div><div class="big"><b>${a.fg3}</b>3-pointers</div><div class="big"><b>${a.pra}</b>P+R+A</div><div class="big"><b>${a.min}</b>minutes</div></div>`;
  $('#pres').innerHTML=`<div class="hero"><img class="face" src="${face(p.id)}" alt="" ${hide}><div><h2 class="d">${esc(p.name)}</h2><span>${esc(p.team)}</span></div></div>
  <div class="avl">Averages, last 5 games</div>${grid(p.avg5)}<div class="avl">Averages, last 10 games</div>${grid(p.avg10)}
  <div class="tbl"><table><tr><th>Date</th><th>Game</th><th>MIN</th><th>PTS</th><th>REB</th><th>AST</th><th>3PM</th><th>P+R+A</th></tr>${rows}</table></div>`}

/* ---------- search (teams + players, narrows as you type) ---------- */
let sTimer,sActs=[];
function closeSug(){$('#sug').classList.add('hide');sActs=[]}
function pickTeam(t){closeSug();$('#q').value='';team=t;stat=null;line=null;tab('team');say(t.name+' selected. Now pick a stat.');drawFlow()}
async function suggest(){
  const raw=$('#q').value.trim(),q=raw.toLowerCase();
  if(!q||q.startsWith('!')){closeSug();return}
  const ts=TEAMS.filter(t=>[t.name,t.nick,t.abbr,t.city].some(x=>x.toLowerCase().includes(q))).sort((a,b)=>(+!a.nick.toLowerCase().startsWith(q))-(+!b.nick.toLowerCase().startsWith(q))||a.name.localeCompare(b.name)).slice(0,5);
  const ps=await fetch('/api/search?q='+encodeURIComponent(raw)).then(r=>r.json()).catch(()=>[]);
  if($('#q').value.trim().toLowerCase()!==q)return;
  sActs=[...ts.map(t=>()=>pickTeam(t)),...ps.map(p=>()=>{closeSug();$('#q').value='';showPlayer(p.id)})];
  let i=0;const btn=(img,cls,b,s)=>`<button class="si" data-i="${i++}"><img class="${cls}" src="${img}" alt="" ${hide}><div><b>${esc(b)}</b><span>${esc(s)}</span></div></button>`;
  $('#sug').innerHTML=(ts.length?'<div class="grp">Teams</div>'+ts.map(t=>btn(logo(t.id),'lg',t.name,'Team')).join(''):'')+(ps.length?'<div class="grp">Players</div>'+ps.map(p=>btn(face(p.id),'',p.name,p.team)).join(''):'')||'<div class="grp">No matches yet. Keep typing.</div>';
  $('#sug').classList.remove('hide');
  $$('#sug .si').forEach(b=>b.onclick=()=>sActs[+b.dataset.i]())}
$('#q').oninput=()=>{clearTimeout(sTimer);sTimer=setTimeout(suggest,120)};
document.addEventListener('click',e=>{if(!e.target.closest('#sbar'))closeSug()});
$('#q').onkeydown=e=>{
  if(e.key==='Escape'){closeSug();return}
  if(e.key!=='Enter')return;e.preventDefault();const v=e.target.value.trim();if(!v)return;
  if(/^!nba/i.test(v)){e.target.value='';closeSug();command(v);return}
  if(sActs.length)sActs[0]()};
function findTeam(q){q=q.toLowerCase().trim();return TEAMS.find(t=>[t.name,t.nick,t.abbr,t.city].some(x=>x.toLowerCase()===q))||TEAMS.find(t=>t.name.toLowerCase().includes(q))}
function command(v){
  const m=v.match(/^!nba\s+(.+?)(?:\s+(\d+)\s*([a-z0-9+]+))?$/i);
  if(!m)return say('Try: !NBA Lakers 20 points',true);
  const t=findTeam(m[1]);if(!t)return say('No team found for "'+m[1]+'". Try the team name, like Lakers.',true);
  team=t;line=null;tab('team');
  if(!m[2]){stat=null;say(t.name+" selected. I'm ready. Pick a stat.");return drawFlow()}
  const s=ALIAS[m[3].toLowerCase()],n=+m[2];
  if(!s)return say('I did not understand "'+m[3]+'". Try points, threes, rebounds, assists or pra.',true);
  if(!STATS[s].lines.includes(n))return say('That number is not available for '+STATS[s].label+'. Options: '+STATS[s].lines.join(', ')+'.',true);
  stat=s;gw=gw||5;run(n)}

/* ---------- live games corner ---------- */
const liveRow=g=>{const a=g.away_score,h=g.home_score,c=g.gid&&g.src?` role="button" tabindex="0" data-src="${g.src}" data-gid="${esc(g.gid)}"`:'';
  return `<div class="lg ${c?'clk':''}"${c}><div class="lt"><span class="lv">● LIVE</span><span>${esc(g.text)}</span></div>
  <div class="ls ${a>=h?'lead':'trail'}"><img src="${g.away_logo||logo(g.away_id)}" alt="" ${hide}>${esc(g.away)}<b>${a}</b></div>
  <div class="ls ${h>=a?'lead':'trail'}"><img src="${g.home_logo||logo(g.home_id)}" alt="" ${hide}>${esc(g.home)}<b>${h}</b></div>${c?'<div class="lmore">Tap for live player stats →</div>':''}</div>`};
function hookLive(){$$('#lbody .clk').forEach(el=>{el.onclick=()=>openBox(el.dataset.src,el.dataset.gid);el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openBox(el.dataset.src,el.dataset.gid)}}})}
let boxKey=null,boxTimer=0;
function closeBox(){$('#lmodal').classList.add('hide');boxKey=null;clearTimeout(boxTimer)}
async function openBox(src,gid){boxKey=src+':'+gid;$('#lmodal').classList.remove('hide');$('#lmbody').innerHTML=spin('Loading player stats…');refreshBox()}
async function refreshBox(){
  if(!boxKey)return;const [src,gid]=boxKey.split(':');
  const d=await fetch(`/api/live/box?src=${src}&gid=${gid}`).then(r=>r.json()).catch(()=>({error:'Could not reach the app.'}));
  if(!boxKey||boxKey!==src+':'+gid)return;
  if(d.error){$('#lmbody').innerHTML=`<div class="empty">${esc(d.error)}</div>`}
  else{const ph=id=>src==='nba'?`https://cdn.nba.com/headshots/nba/latest/260x190/${id}.png`:`https://a.espncdn.com/i/headshots/nba/players/full/${id}.png`;
    const side=s=>`<h3 class="d">${esc(s.abbr)} <b>${s.score}</b></h3><div class="tbl"><table><tr><th>Player</th><th>MIN</th><th>PTS</th><th>REB</th><th>AST</th><th>3PM</th><th>STL</th><th>BLK</th><th>FG</th></tr>${s.players.map(p=>`<tr class="${p.on?'oc':''}"><td><img class="face xs" src="${ph(p.id)}" alt="" ${hide}>${esc(p.name)}${p.on?' <i class="onc">on court</i>':''}</td><td>${esc(p.min)}</td><td class="n">${p.pts}</td><td class="n">${p.reb}</td><td class="n">${p.ast}</td><td class="n">${p.fg3}</td><td class="n">${p.stl}</td><td class="n">${p.blk}</td><td>${esc(p.fg)}</td></tr>`).join('')||'<tr><td colspan="9">No one has played yet.</td></tr>'}</table></div>`;
    $('#lmbody').innerHTML=`<div class="lmh"><span class="lv">● ${esc(d.status||'')}</span><span>Updated ${esc(d.updated)} • refreshes every 10 seconds</span></div>${side(d.away)}${side(d.home)}`}
  clearTimeout(boxTimer);boxTimer=setTimeout(refreshBox,10000)}
$('#lclose').onclick=closeBox;$('#lmodal').onclick=e=>{if(e.target.id==='lmodal')closeBox()};document.addEventListener('keydown',e=>{if(e.key==='Escape')closeBox()});
async function loadLive(){
  const d=await fetch('/api/live').then(r=>r.json()).catch(()=>({error:'Live scores are unavailable right now.'}));
  if(d.error){$('#lbody').innerHTML=`<div class="lnote">${esc(d.error)}</div>`;$('#ldot').classList.remove('on');return}
  const live=d.games.filter(g=>g.status===2),fin=d.games.filter(g=>g.status===3),up=d.games.filter(g=>g.status===1);
  $('#ldot').classList.toggle('on',live.length>0);$('#lcnt').textContent=live.length?live.length+' live now':(d.games.length?d.games.length+' today':'');
  $('#lbody').innerHTML=(live.length?live.map(liveRow).join(''):'<div class="lnote">No games are live right now.</div>')
   +(fin.length?'<div class="grp" style="padding:4px 6px">Final (tap for stats)</div>'+fin.map(g=>`<div class="lf clk" role="button" tabindex="0" data-src="${g.src}" data-gid="${esc(g.gid)}"><span>${esc(g.away)} <b>${g.away_score}</b> @ ${esc(g.home)} <b>${g.home_score}</b></span></div>`).join(''):'')
   +(!live.length&&up.length?'<div class="grp" style="padding:4px 6px">Up next</div>'+up.slice(0,4).map(g=>`<div class="lf"><span>${esc(g.away)} @ ${esc(g.home)}</span><span>${esc(g.text)}</span></div>`).join(''):'');hookLive()}
$('#lhead').onclick=()=>{const b=$('#lbody'),c=b.classList.toggle('hide');$('#lhead').setAttribute('aria-expanded',String(!c))};
if(matchMedia('(max-width:700px)').matches){$('#lbody').classList.add('hide');$('#lhead').setAttribute('aria-expanded','false')}
setInterval(()=>{if(!document.hidden)loadLive()},10000);

/* ---------- live midnight clock ---------- */
function etSecsLeft(){const p=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'}).formatToParts(new Date());
  const g=t=>+p.find(x=>x.type===t).value;return 86400-((g('hour')%24)*3600+g('minute')*60+g('second'))}
function tickClock(){const t=etSecsLeft(),h=Math.floor(t/3600),m=Math.floor(t%3600/60),s=t%60;
  $('#clock').textContent='Updates at midnight ET in '+h+'h '+String(m).padStart(2,'0')+'m '+String(s).padStart(2,'0')+'s';
  if(lastT&&t>lastT+3000){TODAY=null;loadToday()}lastT=t}
setInterval(tickClock,1000);tickClock();
document.addEventListener('visibilitychange',()=>{if(!document.hidden){loadLive();if(Date.now()-lastLoad>600000&&ntab==='today')loadToday()}});

/* ====================== NFL ====================== */
const NFL=(()=>{
const g=id=>document.getElementById('nf-'+id);
const AF=id=>`https://a.espncdn.com/i/headshots/nfl/players/full/${id}.png`;
const NS={
  rec:{label:'Receptions',lines:[3,4,5,6,7,8,10],f:r=>r[7],pos:['WR','TE','RB']},
  recyds:{label:'Receiving yards',lines:[30,40,50,60,70,80,100],f:r=>r[8],pos:['WR','TE','RB']},
  rushyds:{label:'Rushing yards',lines:[30,40,50,60,70,80,100],f:r=>r[5],pos:['RB','QB']},
  passyds:{label:'QB passing yards',lines:[150,175,200,225,250,275,300],f:r=>r[3],pos:['QB']},
  passtd:{label:'Passing TDs',lines:[1,2,3,4],f:r=>r[4],pos:['QB']},
  td:{label:'Touchdowns',lines:[1,2],f:r=>r[6]+r[9],pos:['RB','WR','TE','QB']}};
const AL={rec:'rec',reception:'rec',receptions:'rec','receiving yards':'recyds','rec yards':'recyds',recyds:'recyds','rushing yards':'rushyds','rush yards':'rushyds',rushyds:'rushyds','passing yards':'passyds','pass yards':'passyds',passyds:'passyds','passing touchdowns':'passtd','passing tds':'passtd','pass tds':'passtd',passtd:'passtd',td:'td',tds:'td',touchdown:'td',touchdowns:'td'};
const tdf=r=>r[6]+r[9];
const COLS={QB:[['PASS YDS',r=>r[3]],['PASS TD',r=>r[4]],['RUSH YDS',r=>r[5]],['TD',tdf]],RB:[['RUSH YDS',r=>r[5]],['REC',r=>r[7]],['REC YDS',r=>r[8]],['TD',tdf]],WR:[['REC',r=>r[7]],['REC YDS',r=>r[8]],['RUSH YDS',r=>r[5]],['TD',tdf]]};
COLS.TE=COLS.WR;
const cols=p=>COLS[p]||COLS.WR;
const sFor=p=>Object.entries(NS).filter(([k,s])=>!p||s.pos.includes(p));
const HOT={all:'All',rec:'Receptions',recyds:'Rec yards',rushyds:'Rush yards',passyds:'Pass yards',passtd:'Pass TDs',td:'Touchdowns'};
let started=false,tab='today',TEAMS=[],team=null,stat=null,gw=null,line=null,TODAY=null,GAME=null,GSIDE=0,betN=5,betV=0,hotStat='all',hotW=5,riseW=5,tTimer=0,lastLoad=0,lastT=0,sActs=[],sTimer=0;

function mount(){
  $('#nf-tabs').innerHTML=`<button class="tab on" data-t="today">This week</button><button class="tab" data-t="standings">Standings</button><button class="tab" data-t="team">Team props</button>`;
  $('#nf-sbar').innerHTML=`<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg><input id="nf-q" autocomplete="off" placeholder="Search an NFL team or player…" aria-label="Search an NFL team or player"><div class="sugbox hide" id="nf-sug"></div>`;
  $('#nfl').innerHTML=`
  <section id="nf-today">
    <div class="hero2"><div><h2 id="nf-tlabel" class="d">This week</h2><p id="nf-tsub"></p></div><div class="clock" id="nf-clock"></div></div>
    <div class="note" style="margin-top:10px">Tap a game to see every player, their stats, and who I think wins.</div>
    <div class="slate" id="nf-slate"></div>
    <div class="bet"><h2 class="d">Build a bet</h2><p>Choose how many players. I'll pick players from this week's games who just started hitting a number they don't usually hit.</p>
      <div class="ncs" id="nf-ncs"></div><button class="go" id="nf-build">Build my bet</button><div id="nf-bet"></div></div>
    <div class="step"><h2>Players on the rise</h2><p id="nf-risenote"></p><div class="seg" id="nf-risew"></div><div class="rgrid" id="nf-rise" style="margin-top:12px"></div></div>
    <div class="step"><h2>Players on a hot streak</h2><p id="nf-hotnote"></p><div class="ctl"><div class="seg" id="nf-hotw"></div><div class="seg" id="nf-tstats"></div></div><div id="nf-hot" style="margin-top:12px"></div></div>
  </section>
  <section id="nf-gameview" class="hide"><button class="back" id="nf-back">← Back to this week</button><div id="nf-gv"></div></section>
  <section id="nf-player" class="hide"><button class="back" id="nf-back2">← Back</button><div id="nf-pres"></div></section>
  <section id="nf-standings" class="hide"><div id="nf-stbody"></div></section>
  <section id="nf-team" class="hide">
    <div class="step" style="margin-top:0"><h2>1. Pick a team</h2><div class="teams" id="nf-teams"><div class="load"><i class="spin"></i>Loading teams…</div></div></div>
    <div class="bet hide" id="nf-tbet"><h2 class="d" id="nf-tbtitle">Build a bet</h2><p id="nf-tbnote"></p><div class="ncs" id="nf-tncs"></div><button class="go" id="nf-tbuild">Build my bet</button><div id="nf-tbres"></div></div>
    <div class="step"><h2>2. Pick a stat</h2><div class="seg" id="nf-stats"></div></div>
    <div class="step hide" id="nf-s3"><h2>3. How many recent games?</h2><div class="seg bigseg" id="nf-gw"></div></div>
    <div class="step hide" id="nf-s4"><h2>4. Pick a number</h2><p>Tap one. You'll see every eligible player on the team and how often they reached at least that number.</p><div class="chips" id="nf-lines"></div></div>
    <div id="nf-res" style="margin-top:28px"></div>
  </section>
  <aside class="live" id="nf-live"><button class="lh" id="nf-lhead" aria-expanded="true"><span class="dot" id="nf-ldot"></span>LIVE<span class="cnt" id="nf-lcnt"></span></button><div class="lb2" id="nf-lbody"></div></aside>`;
  document.querySelectorAll('#nf-tabs .tab').forEach(b=>b.onclick=()=>{show(b.dataset.t);say('')});
  g('back').onclick=()=>show('today');g('back2').onclick=()=>show('today');
  g('build').onclick=()=>build(betV=0);
  g('tbuild').onclick=()=>tbuild(tbV=0);
  g('lhead').onclick=()=>{const c=g('lbody').classList.toggle('hide');g('lhead').setAttribute('aria-expanded',String(!c))};
  if(matchMedia('(max-width:700px)').matches){g('lbody').classList.add('hide');g('lhead').setAttribute('aria-expanded','false')}
  g('q').oninput=()=>{clearTimeout(sTimer);sTimer=setTimeout(suggest,120)};
  g('q').onkeydown=e=>{if(e.key==='Escape'){closeSug();return}if(e.key!=='Enter')return;e.preventDefault();const v=e.target.value.trim();if(!v)return;
    if(/^!nfl/i.test(v)){e.target.value='';closeSug();command(v);return}if(sActs.length)sActs[0]()};
  document.addEventListener('click',e=>{if(!e.target.closest('#nf-sbar'))closeSug()});
}
function show(n){tab=n;document.querySelectorAll('#nf-tabs .tab').forEach(b=>b.classList.toggle('on',b.dataset.t===(n==='gameview'?'today':n)));
  ['today','gameview','player','standings','team'].forEach(s=>g(s).classList.toggle('hide',n!==s));
  if(n==='standings')loadStand();window.scrollTo({top:0})}

/* ---------- this week ---------- */
function drawRise(){
  wBtns('#nf-risew',riseW,n=>{riseW=n;drawRise()});
  g('risenote').textContent=`Their average over the last ${riseW} games compared to the games before that. Each bar is one game.`;
  const L=TODAY?(TODAY.rising[riseW]||[]):[];
  g('rise').innerHTML=L.length?L.map(c=>riseCard(c,AF)).join(''):'<div class="empty">No clear risers in this week\'s games.</div>';
  document.querySelectorAll('#nf-rise .rc').forEach(el=>{el.onclick=()=>showPlayer(el.dataset.id);el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();showPlayer(el.dataset.id)}}})}
function drawHot(){
  wBtns('#nf-hotw',hotW,n=>{hotW=n;drawHot()});
  g('hotnote').textContent=`Heating up means they just started hitting a number they didn't hit before. Steady means they've hit it most of their last ${hotW} games. Heating up comes first.`;
  g('tstats').innerHTML=Object.entries(HOT).map(([k,v])=>`<button class="${k===hotStat?'on':''}" data-k="${k}">${v}</button>`).join('');
  document.querySelectorAll('#nf-tstats button').forEach(b=>b.onclick=()=>{hotStat=b.dataset.k;drawHot()});
  const L=(TODAY?(TODAY.cands[hotW]||[]):[]).filter(c=>hotStat==='all'||c.stat===hotStat).slice(0,40);
  g('hot').innerHTML=L.length?L.map(c=>pickRow(c,AF)).join(''):'<div class="empty">Nobody fits right now. That can happen when few games are left this week.</div>'}
async function loadToday(){
  clearTimeout(tTimer);lastLoad=Date.now();
  if(!TODAY)g('slate').innerHTML=spin('Loading…');
  const d=await api('/api/nfl/today');
  if(d.error){g('slate').innerHTML=`<div class="empty err">${esc(d.error)}<br><button class="retry" id="nf-retry">Try again</button></div>`;g('retry').onclick=loadToday;return}
  TODAY=d;const n=d.games.length;
  g('tlabel').textContent=n?d.label:'No games left';
  g('tsub').textContent=(n?n+' game'+(n>1?'s':'')+' left. ':'No NFL games left right now. ')+'Rebuilds by itself at midnight Eastern and as games finish.';
  g('slate').innerHTML=d.games.map(x=>gameChip(x,true)).join('');
  document.querySelectorAll('#nf-slate .gm').forEach(b=>b.onclick=()=>openGame(b.dataset.gid));
  drawRise();drawHot();tTimer=setTimeout(loadToday,Math.min(d.next_refresh_in+15,1800)*1000)}
function drawN(){g('ncs').innerHTML=[1,2,3,4,5,6,7,8,9,10].map(n=>`<button class="nc ${n===betN?'on':''}" data-n="${n}">${n}</button>`).join('');
  document.querySelectorAll('#nf-ncs .nc').forEach(b=>b.onclick=()=>{betN=+b.dataset.n;betV=0;drawN();g('bet').innerHTML=''})}
async function build(v){
  g('bet').innerHTML=spin('Picking your players…');
  const d=await api(`/api/nfl/parlay?n=${betN}&variant=${v}`);
  if(d.error||!d.legs.length){g('bet').innerHTML=`<div class="empty err">${esc(d.error||'No players fit this week. That happens when few games are left. Check back later.')}</div>`;return}
  const short=d.legs.length<betN?` I could only find ${d.legs.length} that fit this week.`:'';
  g('bet').innerHTML=`<div class="sum2"><b class="d">${d.legs.length}-player bet</b><span>Rough chance every leg hits: <b>${d.chance}%</b>${short}</span></div>${d.legs.map(c=>pickRow(c,AF)).join('')}<button class="again" id="nf-again">Show me a different one</button><p class="fine">Based only on recent games. It doesn't know about injuries, weather, matchups or sportsbook odds, so check that each player is playing and what line your sportsbook offers. More players means a lower chance, since every leg has to hit.</p>`;
  g('again').onclick=()=>build(++betV)}

/* ---------- game view ---------- */
async function openGame(gid){GAME=null;GSIDE=0;show('gameview');g('gv').innerHTML=spin('Loading game…');
  const d=await api('/api/nfl/game?gid='+gid);if(d.error){g('gv').innerHTML=`<div class="empty err">${esc(d.error)}</div>`;return}GAME=d;drawGame()}
const chips=(gs,pos)=>sFor(pos).map(([k,s])=>`<b>${s.label}: how often they reached each number (last 5 · last 10)</b>`+s.lines.map(l=>{const g5=gs.slice(0,5),h5=g5.filter(r=>s.f(r)>=l).length,h10=gs.filter(r=>s.f(r)>=l).length,p=h5/g5.length;return `<span class="hc ${p>=.7?'hi':p>=.5?'mid':'lo'}">${l}+ &nbsp;${h5}/${g5.length} · ${h10}/${gs.length}</span>`}).join('')).join('');
const tbl=(gs,pos)=>`<div class="tbl"><table><tr><th>Date</th><th>Game</th>${cols(pos).map(c=>`<th>${c[0]}</th>`).join('')}</tr>${gs.map(r=>`<tr><td>${esc(r[0].slice(5))}</td><td>${esc(r[1])} <span class="${r[2]}">${r[2]}</span></td>${cols(pos).map(c=>`<td class="n">${c[1](r)}</td>`).join('')}</tr>`).join('')}</table></div>`;
function drawGame(){
  if(!GAME)return;const d=GAME,A=d.sides[0],H=d.sides[1],p=d.pick;
  const pb=p?`<div class="ml"><h3 class="d">Money line: ${esc(p.name)}</h3><p>${esc(p.label)}. My estimate from records, recent form, scoring margin and home or road results.</p>
    <div class="pbar"><div class="a" style="width:${p.prob}%">${esc(p.team)} ${p.prob}%</div><div class="b">${esc(p.other)} ${p.other_prob}%</div></div>
    <ul>${p.reasons.map(r=>`<li>${esc(r)}</li>`).join('')}</ul><p style="margin:12px 0 0">It doesn't know about injuries, weather or the betting odds. A big favorite may pay very little.</p></div>`:'<div class="empty">Not enough results yet for a money line pick.</div>';
  const side=d.sides[GSIDE];
  const list=side.players.map((pl,i)=>{const gs=pl.games.slice(0,5);
    return `<div class="pl"><button class="plh" data-i="${i}"><img class="face s" src="${AF(pl.id)}" alt="" ${hide}><div class="nm"><b>${esc(pl.name)}<i class="tag">${esc(pl.pos)}</i></b><span>Last 5: ${cols(pl.pos).map(c=>c[0]+' '+one(avg(gs,c[1]))).join(' • ')}</span></div><span class="ch">▾</span></button><div class="pld hide" id="nf-pd${i}"></div></div>`}).join('');
  g('gv').innerHTML=`<div class="gvh"><span><img src="${A.logo}" alt="" ${hide}> ${esc(A.abbr)}<small>${esc(A.record)}</small></span><span style="font-size:24px;color:var(--mute)">@</span><span><img src="${H.logo}" alt="" ${hide}> ${esc(H.abbr)}<small>${esc(H.record)}</small></span></div><div style="text-align:center;color:var(--mute);margin:0 0 16px">${esc(d.time)}</div>${pb}
  <div class="step"><h2>Players</h2><p>Tap a player to see each of their last 10 games and how often they hit every number.</p>
  <div class="seg" id="nf-gside">${d.sides.map((s,i)=>`<button class="${i===GSIDE?'on':''}" data-i="${i}">${esc(s.name)}</button>`).join('')}</div><div style="margin-top:12px">${list}</div></div>`;
  document.querySelectorAll('#nf-gside button').forEach(b=>b.onclick=()=>{GSIDE=+b.dataset.i;drawGame()});
  document.querySelectorAll('#nf-gv .plh').forEach(b=>b.onclick=()=>{const i=+b.dataset.i,el=g('pd'+i),open=el.classList.toggle('hide')===false;b.querySelector('.ch').textContent=open?'▴':'▾';
    if(open&&!el.innerHTML){const pl=side.players[i];el.innerHTML=tbl(pl.games,pl.pos)+`<div class="hrs">${chips(pl.games,pl.pos)}</div>`}})}

/* ---------- standings: AFC left, NFC right ---------- */
async function loadStand(){
  g('stbody').innerHTML=spin('Loading standings…');
  const d=await api('/api/nfl/standings');if(d.error){g('stbody').innerHTML=`<div class="empty err">${esc(d.error)}</div>`;return}
  const t=(nm,rows)=>`<div class="tbl st"><h3 class="d">${nm}<span class="cf"> Conference</span></h3><table><tr><th>#</th><th>Team</th><th>W-L</th><th class="opt">PCT</th><th class="opt">Diff</th><th class="opt">L5</th><th class="opt">Strk</th></tr>${rows.map(r=>`<tr class="${r.rank<=7?'po':''}"><td>${r.rank}</td><td><img src="${r.logo}" alt="" ${hide}><span class="fn">${esc(r.nick)}</span><span class="ab">${esc(r.abbr)}</span></td><td>${esc(r.rec)}</td><td class="opt">${r.pct}</td><td class="opt">${esc(r.diff)}</td><td class="opt">${esc(r.l5)}</td><td class="opt">${esc(r.streak)}</td></tr>`).join('')}</table></div>`;
  g('stbody').innerHTML=`<div class="hero2"><div><h2 class="d">NFL standings</h2><p>Green stripe: the top 7 teams by record in each conference. (Division winners get the top 4 seeds, so the real seeding can differ a little.) Diff is points scored minus points allowed.</p></div></div><div class="stgrid">${t('AFC',d.afc)}${t('NFC',d.nfc)}</div>`}

/* ---------- team props: team, stat, how many games, number ---------- */
let tbN=5,tbV=0,tbTeam=null;
function drawTN(){g('tncs').innerHTML=[1,2,3,4,5,6,7,8,9,10].map(n=>`<button class="nc ${n===tbN?'on':''}" data-n="${n}">${n}</button>`).join('');
  document.querySelectorAll('#nf-tncs .nc').forEach(b=>b.onclick=()=>{tbN=+b.dataset.n;tbV=0;drawTN();g('tbres').innerHTML=''})}
async function tbuild(v){
  if(!team)return;g('tbres').innerHTML=spin('Picking players from the '+esc(team.nick)+'…');
  const d=await api(`/api/nfl/team_parlay?team_id=${team.id}&n=${tbN}&variant=${v}&w=${gw||5}`);
  if(d.error||!d.legs.length){g('tbres').innerHTML=`<div class="empty err">${esc(d.error||'No '+team.nick+' player has a reliable recent number right now. Try Last 10 games.')}</div>`;return}
  const short=d.legs.length<tbN?` Only ${d.legs.length} ${team.nick} players fit right now.`:'';
  g('tbres').innerHTML=`<div class="sum2"><b class="d">${d.legs.length}-player ${esc(team.nick)} bet</b><span>Rough chance every leg hits: <b>${d.chance}%</b>${short}</span></div>${d.legs.map(c=>pickRow(c,AF)).join('')}<button class="again" id="nf-tbagain">Show me a different one</button><p class="fine">Based only on recent games. It doesn't know about injuries, weather, matchups or sportsbook odds. More players means a lower chance, since every leg has to hit.</p>`;
  g('tbagain').onclick=()=>tbuild(++tbV)}
function drawFlow(){
  g('tbet').classList.toggle('hide',!team);
  if(team){g('tbtitle').textContent='Build a bet from the '+team.nick;g('tbnote').textContent=`Only ${team.name} players. Uses each player's last ${gw||5} games${gw?'':' (pick Last 10 below to use 10)'}.`;
    if(tbTeam!==team.id){tbTeam=team.id;tbV=0;g('tbres').innerHTML=''}}
  g('teams').innerHTML=TEAMS.map(t=>`<button class="t ${team&&team.id===t.id?'on':''}" data-id="${t.id}" title="${esc(t.name)}"><img src="${t.logo}" alt="" ${hide}>${esc(t.abbr)}</button>`).join('')||'<div class="empty err">Could not load teams. Refresh the page.</div>';
  document.querySelectorAll('#nf-teams .t').forEach(b=>b.onclick=()=>{team=TEAMS.find(t=>t.id==b.dataset.id);line=null;say(team.name+' selected.');drawFlow()});
  g('stats').innerHTML=Object.entries(NS).map(([k,v])=>`<button class="${k===stat?'on':''}" data-k="${k}">${v.label}</button>`).join('');
  document.querySelectorAll('#nf-stats button').forEach(b=>b.onclick=()=>{stat=b.dataset.k;line=null;drawFlow()});
  const ready=!!(team&&stat);
  g('s3').classList.toggle('hide',!ready);g('s4').classList.toggle('hide',!(ready&&gw));
  g('gw').innerHTML=[5,10].map(n=>`<button class="${n===gw?'on':''}" data-g="${n}">Last ${n} games</button>`).join('');
  document.querySelectorAll('#nf-gw button').forEach(b=>b.onclick=()=>{gw=+b.dataset.g;if(line)run(line);else drawFlow()});
  g('lines').innerHTML=stat?NS[stat].lines.map(n=>`<button class="chip ${n===line?'on':''}" data-n="${n}">${n}+</button>`).join(''):'';
  document.querySelectorAll('#nf-lines .chip').forEach(b=>b.onclick=()=>run(+b.dataset.n));
  if(!(ready&&gw&&line))g('res').innerHTML=''}
async function run(n){
  line=n;drawFlow();g('res').innerHTML=spin('Checking every '+esc(team.nick)+' player…');
  const d=await api(`/api/nfl/props?team_id=${team.id}&stat=${stat}&line=${n}&w=${gw}`);
  if(d.error){g('res').innerHTML=`<div class="empty err">${esc(d.error)}</div>`;return}
  if(!d.rows.length){g('res').innerHTML='<div class="empty">No eligible players with recent games for this team.</div>';return}
  const rows=d.rows.map(r=>{const k=r.pct>=70?'hi':r.pct>=50?'mid':'lo';
    return `<div class="row ${k}"><img class="face" src="${AF(r.id)}" alt="" ${hide}><div class="who"><b>${esc(r.name)}<i class="tag">${esc(r.pos)}</i></b><span>${r.hits} of ${r.n} games</span></div><div class="pct">${r.pct}%</div>${boxes(r.vals,n)}</div>`}).join('');
  g('res').innerHTML=`<h2 class="d" style="margin:0;font-size:36px;font-weight:800">${esc(team.name)}</h2><div class="sum">Who reached <b>${n}+ ${NS[stat].label.toLowerCase()}</b> in their last ${gw} games</div>
  <div class="legend"><span class="g h">✓ ${n}+</span><span class="g m">✗ under ${n}</span><span>Each box is one game, newest first. Best hit rate is at the top.</span></div>${rows}`}

/* ---------- player page ---------- */
const grid=(gs,pos)=>`<div class="avgs">${cols(pos).map(c=>`<div class="big"><b>${one(avg(gs,c[1]))}</b>${c[0].toLowerCase()}</div>`).join('')}</div>`;
async function showPlayer(id){
  show('player');g('pres').innerHTML=spin('Loading player…');
  const p=await api('/api/nfl/player?id='+id);
  if(p.error){g('pres').innerHTML=`<div class="empty err">${esc(p.error)}</div>`;return}
  g('pres').innerHTML=`<div class="hero"><img class="face" src="${AF(p.id)}" alt="" ${hide}><div><h2 class="d">${esc(p.name)}</h2><span>${esc(p.pos)} • ${esc(p.team)}</span></div></div>
  <div class="avl">Averages, last 5 games</div>${grid(p.games.slice(0,5),p.pos)}<div class="avl">Averages, last 10 games</div>${grid(p.games,p.pos)}${tbl(p.games,p.pos)}`}

/* ---------- search ---------- */
function closeSug(){g('sug').classList.add('hide');sActs=[]}
function pickTeam(t){closeSug();g('q').value='';team=t;stat=null;line=null;show('team');say(t.name+' selected. Now pick a stat.');drawFlow()}
async function suggest(){
  const raw=g('q').value.trim(),q=raw.toLowerCase();
  if(!q||q.startsWith('!')){closeSug();return}
  const ts=TEAMS.filter(t=>[t.name,t.nick,t.abbr,t.city].some(x=>x.toLowerCase().includes(q))).slice(0,5);
  const ps=await fetch('/api/nfl/search?q='+encodeURIComponent(raw)).then(r=>r.json()).catch(()=>[]);
  if(g('q').value.trim().toLowerCase()!==q)return;
  sActs=[...ts.map(t=>()=>pickTeam(t)),...ps.map(p=>()=>{closeSug();g('q').value='';showPlayer(p.id)})];
  let i=0;const btn=(img,cls,b,s)=>`<button class="si" data-i="${i++}"><img class="${cls}" src="${img}" alt="" ${hide}><div><b>${esc(b)}</b><span>${esc(s)}</span></div></button>`;
  g('sug').innerHTML=(ts.length?'<div class="grp">Teams</div>'+ts.map(t=>btn(t.logo,'lg',t.name,'Team')).join(''):'')+(ps.length?'<div class="grp">Players</div>'+ps.map(p=>btn(AF(p.id),'',p.name,p.pos+' • '+p.team)).join(''):'')||'<div class="grp">No matches yet. Keep typing.</div>';
  g('sug').classList.remove('hide');
  document.querySelectorAll('#nf-sug .si').forEach(b=>b.onclick=()=>sActs[+b.dataset.i]())}
function findTeam(q){q=q.toLowerCase().trim();return TEAMS.find(t=>[t.name,t.nick,t.abbr,t.city].some(x=>x.toLowerCase()===q))||TEAMS.find(t=>t.name.toLowerCase().includes(q))}
function command(v){
  const m=v.match(/^!nfl\s+(.+?)(?:\s+(\d+)\s+(.+))?$/i);
  if(!m)return say('Try: !NFL Chiefs 250 passing yards',true);
  const t=findTeam(m[1]);if(!t)return say('No team found for "'+m[1]+'". Try the team name, like Chiefs.',true);
  team=t;line=null;show('team');
  if(!m[2]){stat=null;say(t.name+" selected. I'm ready. Pick a stat.");return drawFlow()}
  const s=AL[m[3].toLowerCase().trim()],n=+m[2];
  if(!s)return say('I did not understand "'+m[3]+'". Try receptions, receiving yards, rushing yards, passing yards, passing tds or touchdowns.',true);
  if(!NS[s].lines.includes(n))return say('That number is not available for '+NS[s].label+'. Options: '+NS[s].lines.join(', ')+'.',true);
  stat=s;gw=gw||5;run(n)}

/* ---------- live corner ---------- */
async function loadLive(){
  const d=await fetch('/api/nfl/live').then(r=>r.json()).catch(()=>({error:'Live scores are unavailable right now.'}));
  if(d.error){g('lbody').innerHTML=`<div class="lnote">${esc(d.error)}</div>`;g('ldot').classList.remove('on');return}
  const live=d.games.filter(x=>x.status===2),fin=d.games.filter(x=>x.status===3),up=d.games.filter(x=>x.status===1);
  g('ldot').classList.toggle('on',live.length>0);g('lcnt').textContent=live.length?live.length+' live now':(d.games.length?d.games.length+' this week':'');
  g('lbody').innerHTML=(live.length?live.map(liveRow).join(''):'<div class="lnote">No games are live right now.</div>')
   +(fin.length?'<div class="grp" style="padding:4px 6px">Final</div>'+fin.map(x=>`<div class="lf"><span>${esc(x.away)} <b>${x.away_score}</b> @ ${esc(x.home)} <b>${x.home_score}</b></span></div>`).join(''):'')
   +(!live.length&&up.length?'<div class="grp" style="padding:4px 6px">Up next</div>'+up.slice(0,4).map(x=>`<div class="lf"><span>${esc(x.away)} @ ${esc(x.home)}</span><span>${esc(x.text)}</span></div>`).join(''):'')}
function tickClock(){const t=etSecsLeft(),h=Math.floor(t/3600),m=Math.floor(t%3600/60),s=t%60;
  g('clock').textContent='Updates at midnight ET in '+h+'h '+String(m).padStart(2,'0')+'m '+String(s).padStart(2,'0')+'s';
  if(lastT&&t>lastT+3000){TODAY=null;loadToday()}lastT=t}
function start(){if(started)return;started=true;drawN();drawTN();drawFlow();loadToday();loadLive();
  fetch('/api/nfl/teams').then(r=>r.json()).then(t=>{TEAMS=Array.isArray(t)?t:[];drawFlow()}).catch(()=>{});
  setInterval(()=>{if(!document.hidden&&!$('#nfl').classList.contains('hide'))loadLive()},20000);
  setInterval(tickClock,1000);tickClock()}
return {start,mount};
})();
NFL.mount();

(async()=>{drawN();drawTN();loadToday();loadLive();
  const t=await fetch('/api/teams').then(r=>r.json()).catch(()=>[]);TEAMS=Array.isArray(t)?t:[];drawFlow()})();
</script></body></html>"""


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no data is sent; just finds your network address
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "YOUR-PC-ADDRESS"


def boot():
    D["data"] = load_cache()  # instant start from the last saved stats
    nfl_load_cache()
    if os.environ.get("HOSTED") != "1":
        nfl_maybe_refresh()  # start getting the NFL stats in the background too
    maybe_refresh()
    threading.Thread(target=watch, daemon=True).start()


if os.environ.get("HOSTED") == "1":  # running on a web host (gunicorn imports this file)
    boot()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    boot()
    print(f"\nOn this computer:           http://127.0.0.1:{port}")
    print(f"On your phone (same Wi-Fi): http://{lan_ip()}:{port}\n")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port)
