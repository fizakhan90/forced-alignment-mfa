# Forced alignment with the Montreal Forced Aligner (MFA)

Word- and phone-level forced alignment of 6 English utterances (3 radio-news paragraphs, 3 non-native
minimal-pair sentences) with MFA 3.x, comparing pretrained acoustic models, dictionaries with and without
G2P-generated pronunciations, and a model trained only on this data. The whole pipeline runs with one command.

## Repository layout

```
run_pipeline.py                  one-command pipeline (install -> prepare -> OOV/G2P -> align -> analyse)
scripts/prepare_corpus.py        builds the MFA corpus (speaker folders, .wav + .lab) and normalises transcripts
scripts/summarize_alignments.py  parses TextGrids, flags suspicious intervals, compares runs, plots
outputs/
  alignments/<run>/<speaker>/<utt>.TextGrid   the alignments (main deliverable)
  dictionaries/                  custom dictionaries = pretrained dictionary + G2P pronunciations for OOVs
  oovs/                          out-of-vocabulary words per dictionary
  analysis/                      run_summary.csv, file_summary.csv, flags.csv, run_agreement.csv, plots/*.png
  normalisation_log.csv          original vs normalised transcript for every file
  logs/                          full MFA logs of every step
```

## Alignment runs

| Run | Acoustic model | Dictionary | Purpose |
|---|---|---|---|
| A_arpa_base   | `english_us_arpa` | `english_us_arpa` | baseline; OOV words become `<unk>` / `spn` |
| B_arpa_g2p    | `english_us_arpa` | `english_us_arpa` + G2P (`english_us_arpa` G2P) | effect of a custom dictionary |
| C_mfa_g2p     | `english_mfa`     | `english_us_mfa` + G2P (`english_us_mfa` G2P) | a second acoustic model / phone set |
| D_selftrained | trained on these 6 files only (`mfa train`) | dictionary of run B | how much pretrained data matters |

## 1. Install MFA

MFA is distributed through conda-forge (it bundles Kaldi, OpenFst and PostgreSQL).

```bash
# with conda / mamba
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
mfa version            # tested with 3.4.4
```

No conda? `run_pipeline.py` installs a private copy automatically with micromamba into `./tools/`
(Linux, needs internet):

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C tools bin/micromamba
MAMBA_ROOT_PREFIX=tools/root tools/bin/micromamba create -n mfa -c conda-forge montreal-forced-aligner
```

Python packages for the analysis script: `pip install praatio soundfile matplotlib pandas numpy`.

## 2. Data

The dataset is not redistributed here. Put it in a folder with this layout:

```
data/
  wav/          F2BJRLP1.wav ... ISLE_SESS0131_BLOCKD02_03_sprt1.wav   (16 kHz mono)
  transcripts/  F2BJRLP1.TXT ... ISLE_SESS0131_BLOCKD02_03_sprt1.txt
```

## 3. Run everything (one command)

```bash
python run_pipeline.py --src data --num_jobs 4          # add --skip_train to skip run D
```

Runtime on a 4-core CPU machine (e.g. a Kaggle notebook without GPU): roughly 15-25 min including the one-time install.

## 4. Or run the steps by hand

```bash
export MFA_ROOT_DIR=$PWD/mfa_root

# prepare the corpus: one folder per speaker, <utt>.wav + <utt>.lab
python scripts/prepare_corpus.py --src data --out mfa_corpus

# download pretrained models
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p        english_us_arpa

# validate the corpus and list OOV words
mfa validate  mfa_corpus english_us_arpa --clean
mfa find_oovs mfa_corpus english_us_arpa outputs/oovs/english_us_arpa --clean

# G2P pronunciations for the OOVs, appended to a copy of the dictionary
mfa g2p outputs/oovs/english_us_arpa/oovs_found_english_us_arpa.txt english_us_arpa outputs/dictionaries/g2p_oovs.dict --clean
cat $MFA_ROOT_DIR/pretrained_models/dictionary/english_us_arpa.dict outputs/dictionaries/g2p_oovs.dict > outputs/dictionaries/custom_english_us_arpa.dict

# align
mfa align mfa_corpus outputs/dictionaries/custom_english_us_arpa.dict english_us_arpa outputs/alignments/B_arpa_g2p \
          --beam 100 --retry_beam 400 --clean

# analyse
python scripts/summarize_alignments.py --corpus mfa_corpus --runs outputs/alignments --out outputs/analysis
```

## 5. Inspect in Praat

Open Praat -> *Open -> Read from file* -> select the `.wav`, then the matching `.TextGrid`.
Select both -> *View & Edit*. The TextGrid has two tiers, `words` and `phones`.
`outputs/analysis/plots/` has the same view (waveform, spectrogram, word and phone tiers of every run on one time axis).

## Transcript normalisation

Done in `scripts/prepare_corpus.py`, logged in `outputs/normalisation_log.csv`:
- lines joined, punctuation removed (apostrophes and in-word hyphens kept), lower-cased;
- dotted acronyms spelled out as letters: `S.J.C.'s` -> `s j c's`; station call sign `WBUR's` -> `w b u r's`;
- broken hyphenation from line wrapping joined: `seventy- six` -> `seventy six`.

## Notes

- Very short phones (≤ 30 ms, the minimum MFA can produce), phones > 400 ms, words > 1 s and `<unk>`/`spn`
  intervals are listed in `outputs/analysis/flags.csv` as candidates for alignment errors.
- `run_agreement.csv` compares word onsets between runs on the same file (mean difference, % within 20 ms).
