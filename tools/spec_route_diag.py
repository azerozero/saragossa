"""Diagnostic : tokens par tour DFlash, coût du tour, point mort contre MTP (campagne du 15 nov.)."""
import json, statistics as st, sys
d = json.load(open(sys.argv[1]))
rows = {}
for r in d["runs"]:
    if not r["warmup"]:
        rows.setdefault((r["case"], r["mode"]), []).append(r)
print("cas tok/tour ms/tour ms/token_MTP point_mort gagnant")
for case in ["C1", "C2", "C3", "C32", "P1", "P2", "P3", "P32"]:
    df = rows[(case, "dflash")]; mtp = rows[(case, "mtp")]
    turns = df[0]["proposed"] - df[0]["accepted"]  # un échec par tour sur le chemin parcouru
    ms_turn = st.median(r["decode_ms"] for r in df) / turns
    ms_tok = st.median(r["decode_ms"] for r in mtp) / 256
    be = ms_turn / ms_tok
    print(case, round(256 / turns, 2), round(ms_turn, 1), round(ms_tok, 1), round(be, 2),
          "dflash" if 256 / turns > be else "mtp")
