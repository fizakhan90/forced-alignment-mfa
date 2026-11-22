"""Full forced-alignment pipeline, end to end (one command):

    python run_pipeline.py --src "Assignment 2a"

 0. install MFA (micromamba + conda-forge) if `mfa` is not available
 1. prepare the corpus                               -> mfa_corpus/
 2. download pretrained models                       -> mfa_root/pretrained_models/
 3. validate the corpus + find OOV words per dictionary
 4. generate OOV pronunciations with G2P and build custom dictionaries -> outputs/dictionaries/
 5. align with several model/dictionary combinations -> outputs/alignments/<run>/  (TextGrids)
      A_arpa_base   english_us_arpa acoustic + english_us_arpa dictionary   (OOVs become <unk>/spn)
      B_arpa_g2p    english_us_arpa acoustic + dictionary + G2P OOVs
      C_mfa_g2p     english_mfa acoustic     + english_us_mfa dictionary + G2P OOVs
      D_selftrained acoustic model trained ONLY on these 6 files + B's dictionary   (--skip_train to omit)
 6. summarise / compare / plot                        -> outputs/analysis/
 7. zip everything                                    -> outputs.zip
"""
import argparse, glob, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"acoustic": ["english_us_arpa", "english_mfa"],
          "dictionary": ["english_us_arpa", "english_us_mfa"],
          "g2p": ["english_us_arpa", "english_us_mfa"]}


def sh(cmd, log=None, check=True):
    print("\n$ " + " ".join(cmd), flush=True)
    t = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    tail = "\n".join([l for l in out.splitlines() if "━" not in l][-12:])
    print(tail, f"\n[{time.time() - t:.0f}s, exit {p.returncode}]", flush=True)
    if log:
        open(log, "w").write(out)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return p.returncode, out


def mfa_cmd(tools):
    if shutil.which("mfa"):
        return ["mfa"]
    mm = os.path.join(tools, "bin", "micromamba")
    env = dict(os.environ, MAMBA_ROOT_PREFIX=os.path.join(tools, "root"))
    if not os.path.exists(os.path.join(tools, "root", "envs", "mfa")):
        print("Installing MFA with micromamba (one time, ~3-5 min) ...", flush=True)
        os.makedirs(tools, exist_ok=True)
        if not os.path.exists(mm):
            subprocess.run(f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C {tools} bin/micromamba",
                           shell=True, check=True)
        subprocess.run([mm, "create", "-y", "-q", "-n", "mfa", "-c", "conda-forge", "montreal-forced-aligner"],
                       env=env, check=True)
    os.environ["MAMBA_ROOT_PREFIX"] = env["MAMBA_ROOT_PREFIX"]
    return [mm, "run", "-n", "mfa", "mfa"]


def model_path(root, kind, name):
    ext = {"acoustic": ".zip", "dictionary": ".dict", "g2p": ".zip"}[kind]
    hits = glob.glob(os.path.join(root, "pretrained_models", kind, name + ext)) + \
        glob.glob(os.path.join(root, "pretrained_models", kind, name + "*"))
    return hits[0] if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder containing wav/ and transcripts/")
    p.add_argument("--work", default=".")
    p.add_argument("--num_jobs", type=int, default=2)
    p.add_argument("--beam", type=int, default=100)
    p.add_argument("--retry_beam", type=int, default=400)
    p.add_argument("--skip_train", action="store_true", help="skip the self-trained acoustic model (run D)")
    a = p.parse_args()
    W = os.path.abspath(a.work); os.chdir(W)
    os.environ["MFA_ROOT_DIR"] = root = os.path.join(W, "mfa_root")
    mfa = mfa_cmd(os.path.join(W, "tools"))
    sh(mfa + ["version"])
    out = os.path.join(W, "outputs"); logs = os.path.join(out, "logs")
    for d in ["dictionaries", "alignments", "logs", "oovs"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    common = ["--clean", "--num_jobs", str(a.num_jobs)]

    # 1. corpus
    sh([sys.executable, os.path.join(HERE, "scripts", "prepare_corpus.py"), "--src", a.src, "--out", "mfa_corpus",
        "--log", os.path.join(out, "normalisation_log.csv")])

    # 2. models
    for kind, names in MODELS.items():
        for n in names:
            if model_path(root, kind, n) is None:
                sh(mfa + ["model", "download", kind, n], check=False)
            print(f"{kind:10s} {n:16s} -> {model_path(root, kind, n)}")

    # 3 + 4. OOVs and G2P custom dictionaries
    dicts = {}
    for dname, g2p in [("english_us_arpa", "english_us_arpa"), ("english_us_mfa", "english_us_mfa")]:
        base = model_path(root, "dictionary", dname)
        if base is None:
            print(f"!! dictionary {dname} missing - skipping"); continue
        sh(mfa + ["validate", "mfa_corpus", base] + common, log=os.path.join(logs, f"validate_{dname}.log"), check=False)
        odir = os.path.join(out, "oovs", dname)
        sh(mfa + ["find_oovs", "mfa_corpus", base, odir] + common, log=os.path.join(logs, f"find_oovs_{dname}.log"))
        found = glob.glob(os.path.join(odir, "oovs_found*.txt"))
        oovs = [w.strip() for w in open(found[0]) if w.strip()] if found else []
        print(f"{dname}: {len(oovs)} OOV words: {oovs}")
        custom = os.path.join(out, "dictionaries", f"custom_{dname}.dict")
        shutil.copy(base, custom)
        gm = model_path(root, "g2p", g2p)
        if oovs and gm:
            olist = os.path.join(odir, "oov_list.txt"); open(olist, "w").write("\n".join(oovs) + "\n")
            prons = os.path.join(out, "dictionaries", f"g2p_oovs_{dname}.dict")
            sh(mfa + ["g2p", olist, gm, prons, "--clean"], log=os.path.join(logs, f"g2p_{dname}.log"))
            with open(custom, "a") as f:
                f.write("\n" + open(prons).read())
            print(open(prons).read())
        dicts[dname] = (base, custom)

    # 5. alignment runs
    runs = []
    if "english_us_arpa" in dicts and model_path(root, "acoustic", "english_us_arpa"):
        am = model_path(root, "acoustic", "english_us_arpa")
        runs += [("A_arpa_base", dicts["english_us_arpa"][0], am), ("B_arpa_g2p", dicts["english_us_arpa"][1], am)]
    if "english_us_mfa" in dicts and model_path(root, "acoustic", "english_mfa"):
        runs += [("C_mfa_g2p", dicts["english_us_mfa"][1], model_path(root, "acoustic", "english_mfa"))]
    for name, dic, am in runs:
        dest = os.path.join(out, "alignments", name)
        sh(mfa + ["align", "mfa_corpus", dic, am, dest, "--beam", str(a.beam), "--retry_beam", str(a.retry_beam)] + common,
           log=os.path.join(logs, f"align_{name}.log"), check=False)
    if not a.skip_train and "english_us_arpa" in dicts:
        dest = os.path.join(out, "alignments", "D_selftrained")
        sh(mfa + ["train", "mfa_corpus", dicts["english_us_arpa"][1], os.path.join(out, "selftrained_acoustic_model.zip"),
                  "--output_directory", dest] + common, log=os.path.join(logs, "train_D_selftrained.log"), check=False)

    # 6. analysis, 7. zip
    sh([sys.executable, os.path.join(HERE, "scripts", "summarize_alignments.py"), "--corpus", "mfa_corpus",
        "--runs", os.path.join(out, "alignments"), "--out", os.path.join(out, "analysis")], check=False)
    shutil.make_archive(os.path.join(W, "outputs"), "zip", W, "outputs")
    print("\nDONE -> outputs/ (TextGrids in outputs/alignments/*/, analysis in outputs/analysis/), outputs.zip")


if __name__ == "__main__":
    main()
