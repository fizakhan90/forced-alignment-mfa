"""Step 4: inspect and compare the TextGrids produced by each alignment run.

Usage:  python scripts/summarize_alignments.py --corpus mfa_corpus --runs outputs/alignments --out outputs/analysis
  --runs  folder whose sub-folders are runs (e.g. A_arpa_base, B_arpa_g2p, C_mfa_g2p), each holding MFA output

Writes:
  run_summary.csv       per run: files aligned/missing, words, unknown words, phones, suspicious phones
  file_summary.csv      per run x file: the same, plus MFA's own quality scores (alignment_analysis.csv)
  words_<run>.csv / phones_<run>.csv   every interval with start, end, duration
  flags.csv             suspicious intervals (very short / very long phones, <unk>/spn, long words)
  run_agreement.csv     word-boundary differences between runs on the same files
  plots/<file>.png      spectrogram + word/phone tiers of every run, stacked (same time axis)
"""
import argparse, glob, itertools, os

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from praatio import textgrid

UNK = {"<unk>", "spn", "<eps>"}
SHORT_PHONE = 0.03    # 3 frames of 10 ms = MFA's minimum per HMM state path; anything this short is suspicious
LONG_PHONE = 0.40     # a single phone longer than 400 ms is usually a boundary error (or a hesitation)


def read_tg(path):
    tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
    names = {n.lower(): n for n in tg.tierNames}
    def tier(key):
        n = next((v for k, v in names.items() if k.endswith(key)), None)
        return [(e.start, e.end, e.label) for e in tg.getTier(n).entries] if n else []
    return tier("words"), tier("phones")


def collect(corpus, run_dir):
    utts = {os.path.splitext(os.path.basename(w))[0]: w for w in glob.glob(f"{corpus}/*/*.wav")}
    words, phones, files = [], [], []
    quality = {}
    qa = os.path.join(run_dir, "alignment_analysis.csv")
    if os.path.exists(qa):
        quality = pd.read_csv(qa).set_index("file").to_dict("index")
    for utt, wav in sorted(utts.items()):
        tgs = glob.glob(f"{run_dir}/**/{utt}.TextGrid", recursive=True)
        expected = open(wav[:-4] + ".lab").read().split()
        if not tgs:
            files.append(dict(file=utt, aligned=False, expected_words=len(expected)))
            continue
        w, p = read_tg(tgs[0])
        for s, e, l in w:
            words.append(dict(file=utt, start=s, end=e, dur=e - s, label=l))
        for s, e, l in p:
            phones.append(dict(file=utt, start=s, end=e, dur=e - s, label=l))
        pd_ = np.array([e - s for s, e, l in p]) if p else np.array([0.0])
        q = quality.get(utt, {})
        files.append(dict(file=utt, aligned=True, expected_words=len(expected), aligned_words=len(w),
                          unknown_words=sum(l in UNK for _, _, l in w),
                          phones=len(p), mean_phone_ms=1000 * pd_.mean(),
                          short_phones=int((pd_ <= SHORT_PHONE + 1e-6).sum()), long_phones=int((pd_ > LONG_PHONE).sum()),
                          speech_start=w[0][0] if w else np.nan, speech_end=w[-1][1] if w else np.nan,
                          speech_log_likelihood=q.get("speech_log_likelihood"),
                          phone_duration_deviation=q.get("phone_duration_deviation")))
    return pd.DataFrame(words), pd.DataFrame(phones), pd.DataFrame(files)


def flags_for(run, words, phones):
    out = []
    for _, r in phones.iterrows():
        why = "very short phone (<=30 ms)" if r.dur <= SHORT_PHONE + 1e-6 else "very long phone (>400 ms)" if r.dur > LONG_PHONE else None
        if r.label in UNK:
            why = "unknown / spoken-noise phone"
        if why:
            out.append(dict(run=run, tier="phones", **r.to_dict(), reason=why))
    for _, r in words.iterrows():
        why = "unknown word" if r.label in UNK else "word longer than 1 s" if r.dur > 1.0 else None
        if why:
            out.append(dict(run=run, tier="words", **r.to_dict(), reason=why))
    return out


def agreement(all_words):
    rows = []
    for (ra, wa), (rb, wb) in itertools.combinations(all_words.items(), 2):
        for f in sorted(set(wa.file) & set(wb.file)):
            a, b = wa[wa.file == f], wb[wb.file == f]
            if len(a) != len(b) or list(a.label) != list(b.label):
                rows.append(dict(run_a=ra, run_b=rb, file=f, matched_words=0, note="word sequences differ"))
                continue
            d_on = np.abs(a.start.values - b.start.values) * 1000
            d_off = np.abs(a.end.values - b.end.values) * 1000
            rows.append(dict(run_a=ra, run_b=rb, file=f, matched_words=len(a),
                             mean_onset_diff_ms=d_on.mean(), median_onset_diff_ms=np.median(d_on),
                             pct_onsets_within_20ms=100 * (d_on <= 20).mean(),
                             mean_offset_diff_ms=d_off.mean(), max_onset_diff_ms=d_on.max(),
                             worst_word=a.label.values[d_on.argmax()]))
    return pd.DataFrame(rows)


def plot_file(utt, wav, runs, out_png, window=None):
    y, sr = sf.read(wav)
    if y.ndim > 1:
        y = y.mean(1)
    t0, t1 = window or (0, len(y) / sr)
    seg = y[int(t0 * sr):int(t1 * sr)]
    tiers = [(run, kind, df[(df.file == utt) & (df.end > t0) & (df.start < t1)])
             for run, (w, p) in runs.items() for kind, df in (("words", w), ("phones", p))]
    fig, axes = plt.subplots(2 + len(tiers), 1, figsize=(max(10, (t1 - t0) * 1.6), 3 + 0.55 * len(tiers)),
                             sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2] + [0.55] * len(tiers)))
    tt = np.arange(len(seg)) / sr + t0
    axes[0].plot(tt, seg, lw=0.4, color="k"); axes[0].set_yticks([]); axes[0].set_title(utt, fontsize=10, loc="left")
    axes[1].specgram(seg, NFFT=512, Fs=sr, noverlap=384, xextent=(t0, t1), cmap="Greys", vmin=-120)
    axes[1].set_ylim(0, 5000); axes[1].set_ylabel("Hz", fontsize=8)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (run, kind, df) in enumerate(tiers):
        ax = axes[2 + i]; c = colours[list(runs).index(run) % len(colours)]
        for _, r in df.iterrows():
            ax.axvspan(max(r.start, t0), min(r.end, t1), fill=False, ec=c, lw=0.8)
            if r.end - r.start > (0.04 if kind == "phones" else 0.08):
                ax.text((max(r.start, t0) + min(r.end, t1)) / 2, 0.5, r.label, ha="center", va="center",
                        fontsize=6.5 if kind == "phones" else 8, color="red" if r.label in UNK else "black", clip_on=True)
            for b in (r.start, r.end):
                if t0 <= b <= t1:
                    axes[1].axvline(b, color=c, lw=0.3, alpha=0.5) if kind == "words" and i < 2 else None
        ax.set_yticks([]); ax.set_ylabel(f"{run}\n{kind}", rotation=0, ha="right", va="center", fontsize=7)
    axes[-1].set_xlim(t0, t1); axes[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="mfa_corpus")
    p.add_argument("--runs", default="outputs/alignments")
    p.add_argument("--out", default="outputs/analysis")
    p.add_argument("--long_window", type=float, nargs=2, default=[0.0, 8.0],
                   help="time window (s) plotted for long news files")
    a = p.parse_args()
    os.makedirs(f"{a.out}/plots", exist_ok=True)
    run_dirs = {os.path.basename(d.rstrip("/")): d for d in sorted(glob.glob(f"{a.runs}/*/"))}
    all_runs, run_rows, flag_rows, file_tabs = {}, [], [], []
    for run, d in run_dirs.items():
        w, ph, f = collect(a.corpus, d)
        if w.empty:
            print(f"[{run}] no TextGrids found - skipped"); continue
        all_runs[run] = (w, ph)
        w.to_csv(f"{a.out}/words_{run}.csv", index=False); ph.to_csv(f"{a.out}/phones_{run}.csv", index=False)
        file_tabs.append(f.assign(run=run)); flag_rows += flags_for(run, w, ph)
        al = f[f.aligned]
        run_rows.append(dict(run=run, files_aligned=int(f.aligned.sum()), files_missing=int((~f.aligned).sum()),
                             words=int(al.aligned_words.sum()), unknown_words=int(al.unknown_words.sum()),
                             phones=int(al.phones.sum()), mean_phone_ms=round(ph.dur.mean() * 1000, 1),
                             short_phones=int(al.short_phones.sum()), long_phones=int(al.long_phones.sum()),
                             mean_speech_log_likelihood=al.speech_log_likelihood.astype(float).mean()))
    runs = pd.DataFrame(run_rows); runs.to_csv(f"{a.out}/run_summary.csv", index=False)
    files = pd.concat(file_tabs) if file_tabs else pd.DataFrame()
    files.round(3).to_csv(f"{a.out}/file_summary.csv", index=False)
    pd.DataFrame(flag_rows).round(3).to_csv(f"{a.out}/flags.csv", index=False)
    agr = agreement({r: w for r, (w, _) in all_runs.items()})
    agr.round(1).to_csv(f"{a.out}/run_agreement.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n=== run summary ===\n", runs.round(2).to_string(index=False))
    if not files.empty:
        print("\n=== per file ===\n", files[["run", "file", "aligned", "expected_words", "aligned_words", "unknown_words",
                                           "short_phones", "long_phones", "speech_log_likelihood"]].round(2).to_string(index=False))
    if not agr.empty:
        print("\n=== agreement between runs (word onsets) ===\n", agr.round(1).to_string(index=False))
    for wav in sorted(glob.glob(f"{a.corpus}/*/*.wav")):
        utt = os.path.splitext(os.path.basename(wav))[0]
        have = {r: v for r, v in all_runs.items() if (v[0].file == utt).any()}
        if have:
            win = None if sf.info(wav).duration < 10 else tuple(a.long_window)
            plot_file(utt, wav, have, f"{a.out}/plots/{utt}.png", win)
    print(f"\nplots -> {a.out}/plots/")


if __name__ == "__main__":
    main()
