"""Step 1: turn the provided wav/ + transcripts/ folders into an MFA corpus.

MFA wants   corpus/<speaker>/<utt>.wav  +  corpus/<speaker>/<utt>.lab  (same basename, plain-text transcript).

Usage:  python scripts/prepare_corpus.py --src "Assignment 2a" --out mfa_corpus
Writes  mfa_corpus/...  and  outputs/normalisation_log.csv  (original vs normalised text, for the report).
"""
import argparse, csv, glob, os, re, shutil

import soundfile as sf


def speaker_of(utt):
    # F2BJRLP1 -> F2B (Boston University Radio News speaker f2b); ISLE_SESS0131_... -> ISLE_SESS0131
    if utt.startswith("ISLE_"):
        return "_".join(utt.split("_")[:2])
    return utt[:3]


def spell_acronym(m):
    """'S.J.C.'s' -> 's j c's' : dotted acronyms are read letter by letter."""
    letters = re.findall(r"[A-Za-z]", m.group(1))
    tail = m.group(2) or ""
    return " ".join(letters[:-1] + [letters[-1] + tail])


def normalise(text):
    t = " ".join(text.split())                                   # join lines, collapse spaces
    t = re.sub(r"\b((?:[A-Za-z]\.){2,})('s)?", spell_acronym, t)  # S.J.C.'s -> S J C's
    t = re.sub(r"\b(WBUR)('s)?\b", lambda m: " ".join(m.group(1)[:-1]) + " " + m.group(1)[-1] + (m.group(2) or ""), t)
    t = re.sub(r"(\w)-\s+(\w)", r"\1 \2", t)                     # 'seventy- six' -> 'seventy six'
    t = re.sub(r"[^\w\s'\-]", " ", t)                            # drop punctuation, keep apostrophes / hyphens
    t = " ".join(t.split()).lower()
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder that contains wav/ and transcripts/")
    p.add_argument("--out", default="mfa_corpus")
    p.add_argument("--log", default="outputs/normalisation_log.csv")
    a = p.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    rows = []
    for wav in sorted(glob.glob(os.path.join(a.src, "wav", "*.wav"))):
        utt = os.path.splitext(os.path.basename(wav))[0]
        txt = [t for t in glob.glob(os.path.join(a.src, "transcripts", utt + ".*")) if t.lower().endswith(".txt")]
        assert len(txt) == 1, f"no transcript for {utt}"
        raw = open(txt[0], encoding="utf-8", errors="replace").read()
        norm = normalise(raw)
        spk = speaker_of(utt)
        d = os.path.join(a.out, spk); os.makedirs(d, exist_ok=True)
        shutil.copy(wav, os.path.join(d, utt + ".wav"))
        open(os.path.join(d, utt + ".lab"), "w").write(norm + "\n")
        info = sf.info(wav)
        rows.append(dict(utterance=utt, speaker=spk, duration_s=round(info.duration, 2), sample_rate=info.samplerate,
                         n_words=len(norm.split()), original=" ".join(raw.split()), normalised=norm))
        print(f"{spk:14s} {utt:36s} {info.duration:6.2f}s {info.samplerate}Hz  {len(norm.split()):3d} words")
    with open(a.log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} utterances -> {a.out}/   (log: {a.log})")


if __name__ == "__main__":
    main()
