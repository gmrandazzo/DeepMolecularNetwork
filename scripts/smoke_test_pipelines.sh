#!/usr/bin/env bash
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
#
# End-to-end smoke test of every DeepMolecularNetwork pipeline on real data.
#
# It trains (with tiny budgets) and predicts with every model family exposed by
# the installed `dmn` command, using data/example_01 (SMILES, RDKit descriptors,
# solubility target):
#
#   * Tabular DNN       : tab-makedb, tab-train (simple + CV), tab-predict
#   * One-Hot SMILES CNN: smi2ohe, ohe-makedb, ohe-train (uni + multimodal), ohe-predict
#   * 3D voxel CNN      : smi2voxel (RDKit/Gasteiger voxels), voxel-makedb,
#                         voxel-makecsv, voxel-train (simple, CV, multimodal,
#                         single-channel), voxel-predict, vis-3dcnn
#   * 2D image CNN      : RDKit depictions + make_2d_cnn_model, vis-2dcnn
#
# Every step is logged separately; the script never stops at the first crash and
# prints a PASS / FAIL / SKIP summary at the end. Exit status is 0 only if every
# executed step passed.
#
# Requirements: deepmolecularnetwork installed with the [keras3,chem] extras, e.g.
#   pip install -e ".[keras3,chem]"

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----------------------------------------------------------------------------
# Defaults (override with options or environment variables)
# ----------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/example_01}"
NMOLS="${NMOLS:-200}"
EPOCHS="${EPOCHS:-3}"
OUT_DIR="${OUT_DIR:-}"
RUN_GROUPS="${GROUPS_TO_RUN:-cli,tab,ohe,voxel,img2d}"
KEEP=1
DMN="${DMN:-}"
PYTHON="${PYTHON:-}"
VOXEL_GRID="${VOXEL_GRID:-16}"
VOXEL_SPACING="${VOXEL_SPACING:-1.5}"
VOXEL_NROT="${VOXEL_NROT:-4}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -d DIR    Dataset directory with dataset.smi, dataset.desc.csv, target.csv
            (default: ${DATA_DIR})
  -n N      Number of molecules to use (default: ${NMOLS})
  -e N      Training epochs for every model (default: ${EPOCHS})
  -o DIR    Output/work directory (default: \$TMPDIR/dmn_smoke_<timestamp>)
  -g LIST   Comma-separated groups to run: cli,tab,ohe,voxel,img2d
            (default: ${RUN_GROUPS})
  -c        Clean the work directory at the end if every step passed
  -h        Show this help

Environment overrides: DMN (dmn executable), PYTHON (interpreter that has the
package installed), VOXEL_GRID, VOXEL_SPACING, VOXEL_NROT.
EOF
}

while getopts ":d:n:e:o:g:ch" opt; do
    case "${opt}" in
        d) DATA_DIR="${OPTARG}" ;;
        n) NMOLS="${OPTARG}" ;;
        e) EPOCHS="${OPTARG}" ;;
        o) OUT_DIR="${OPTARG}" ;;
        g) RUN_GROUPS="${OPTARG}" ;;
        c) KEEP=0 ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

# ----------------------------------------------------------------------------
# Environment discovery
# ----------------------------------------------------------------------------
if [[ -z "${DMN}" ]]; then
    DMN="$(command -v dmn || true)"
fi
if [[ -z "${DMN}" || ! -x "${DMN}" ]]; then
    echo "ERROR: 'dmn' executable not found. Install the package (pip install -e '.[keras3,chem]')"
    echo "       or set DMN=/path/to/dmn." >&2
    exit 2
fi
if [[ -z "${PYTHON}" ]]; then
    if [[ -x "$(dirname "${DMN}")/python" ]]; then
        PYTHON="$(dirname "${DMN}")/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi

for f in dataset.smi dataset.desc.csv target.csv; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        echo "ERROR: missing ${DATA_DIR}/${f}" >&2
        exit 2
    fi
done

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${TMPDIR:-/tmp}/dmn_smoke_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_DIR}/logs"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
LOG_DIR="${OUT_DIR}/logs"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

# ----------------------------------------------------------------------------
# Step runner
# ----------------------------------------------------------------------------
declare -a STEP_NAMES=() STEP_STATUS=() STEP_TIMES=()
declare -A STATUS_OF=()

if [[ -t 1 ]]; then
    C_OK=$'\e[32m' C_KO=$'\e[31m' C_SK=$'\e[33m' C_B=$'\e[1m' C_0=$'\e[0m'
else
    C_OK="" C_KO="" C_SK="" C_B="" C_0=""
fi

group_enabled() { [[ ",${RUN_GROUPS}," == *",$1,"* ]]; }

record() {
    STEP_NAMES+=("$1"); STEP_STATUS+=("$2"); STEP_TIMES+=("$3"); STATUS_OF["$1"]="$2"
}

# run_step NAME "DEP1 DEP2" command [args...]
# Runs a command, logging stdout/stderr to logs/NAME.log. The step is skipped if
# any dependency did not pass.
run_step() {
    local name="$1" deps="$2"
    shift 2
    local dep
    for dep in ${deps}; do
        if [[ "${STATUS_OF[${dep}]:-MISSING}" != "PASS" ]]; then
            printf '  %-34s %sSKIP%s (dependency %s not passed)\n' "${name}" "${C_SK}" "${C_0}" "${dep}"
            record "${name}" "SKIP" "-"
            return 0
        fi
    done
    local log="${LOG_DIR}/${name}.log" start end rc
    printf '  %-34s ' "${name}"
    start=$(date +%s)
    { echo "\$ $*"; echo; } >"${log}"
    ( cd "${OUT_DIR}" && "$@" ) >>"${log}" 2>&1
    rc=$?
    end=$(date +%s)
    if [[ ${rc} -eq 0 ]]; then
        printf '%sPASS%s (%ss)\n' "${C_OK}" "${C_0}" "$((end - start))"
        record "${name}" "PASS" "$((end - start))s"
    else
        printf '%sFAIL%s (rc=%s, %ss) -> %s\n' "${C_KO}" "${C_0}" "${rc}" "$((end - start))" "${log}"
        tail -n 5 "${log}" | sed 's/^/      | /'
        record "${name}" "FAIL" "$((end - start))s"
    fi
}

# Assert that files exist and are non-empty (used as post-conditions).
# Automatically checks for common extension variations (.keras, .h5, .dnndb).
expect_files() {
    local f ok=0
    for f in "$@"; do
        if [[ ! -s "${f}" && ! -d "${f}" && ! -s "${f}.keras" && ! -s "${f}.h5" && ! -d "${f}.dnndb" ]]; then
            echo "Missing or empty output: ${f}"
            ok=1
        fi
    done
    return ${ok}
}

# Run a command and then assert its outputs: with_outputs "out1 out2" cmd...
with_outputs() {
    local outs="$1"
    shift
    "$@" || return $?
    # shellcheck disable=SC2086
    expect_files ${outs}
}

section() { printf '\n%s== %s ==%s\n' "${C_B}" "$1" "${C_0}"; }

# ----------------------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------------------
section "Setup"
echo "  dmn      : ${DMN} ($("${DMN}" --version 2>/dev/null | tail -n 1))"
echo "  python   : ${PYTHON}"
echo "  data     : ${DATA_DIR}"
echo "  workdir  : ${OUT_DIR}"
echo "  molecules: ${NMOLS}   epochs: ${EPOCHS}   groups: ${RUN_GROUPS}"

D="${OUT_DIR}/data"
mkdir -p "${D}"

prepare_data() {
    # Keep the first N molecules that have a valid SMILES entry.
    head -n "${NMOLS}" "${DATA_DIR}/dataset.smi" >"${D}/subset.smi"
    cut -f2 "${D}/subset.smi" | tr -d '\r' >"${D}/names.txt"
    awk -F',' 'NR==FNR {keep[$1]=1; next} FNR==1 || ($1 in keep)' \
        "${D}/names.txt" "${DATA_DIR}/dataset.desc.csv" >"${D}/desc.csv"
    awk -F',' 'NR==FNR {keep[$1]=1; next} FNR==1 || ($1 in keep)' \
        "${D}/names.txt" "${DATA_DIR}/target.csv" >"${D}/target.csv"
    # A small descriptor subset for multimodal models (keeps them fast).
    cut -d',' -f1-21 "${D}/desc.csv" >"${D}/desc_small.csv"
    echo "subset.smi: $(wc -l <"${D}/subset.smi") desc.csv: $(($(wc -l <"${D}/desc.csv") - 1))" \
        "target.csv: $(($(wc -l <"${D}/target.csv") - 1))"
}
run_step "prepare-data" "" prepare_data

# ----------------------------------------------------------------------------
# CLI sanity
# ----------------------------------------------------------------------------
if group_enabled cli; then
    section "CLI entry points"
    run_step "cli-version" "" "${DMN}" --version
    run_step "cli-help" "" "${DMN}" --help

    check_console_scripts() {
        local bin_dir s missing=0
        bin_dir="$(dirname "${DMN}")"
        for s in dmn deepmolecularnetwork dmn-makednndb dmn-maketabmodel dmn-maketabprediction \
            dmn-makeohedb dmn-makeohemodel dmn-makeoheprediction dmn-smi2ohe dmn-makevoxeldb \
            dmn-make3dcnnmodel dmn-make3dcnnprediction dmn-makecsvdataset dmn-smi2voxel \
            dmn-img2dvis dmn-vis3dcnn makeTABModel makeTABPrediction make3DCNNModel \
            make3DCNNPrediction makeVoxelCNNDatabase makeOHEModel SMI2OHEMatrix; do
            if [[ -x "${bin_dir}/${s}" ]] || command -v "${s}" >/dev/null 2>&1; then
                echo "ok      ${s}"
            else
                echo "MISSING ${s}"
                missing=1
            fi
        done
        return ${missing}
    }
    run_step "cli-console-scripts" "" check_console_scripts
fi

# ----------------------------------------------------------------------------
# Tabular DNN
# ----------------------------------------------------------------------------
if group_enabled tab; then
    section "Tabular DNN (RDKit descriptors)"
    T="${OUT_DIR}/tabular"
    mkdir -p "${T}"
    run_step "tab-makedb" "prepare-data" with_outputs "${T}/dnndb" \
        "${DMN}" tab-makedb --xmatrix "${D}/desc.csv" --dbout "${T}/dnndb"
    run_step "tab-train" "prepare-data" with_outputs "${T}/model" \
        "${DMN}" tab-train --xmatrix "${D}/desc.csv" --ytarget "${D}/target.csv" \
        --epochs "${EPOCHS}" --batch_size 32 --nunits 32 --ndense_layers 2 --mout "${T}/model"
    run_step "tab-predict" "tab-train" with_outputs "${T}/pred.csv" \
        "${DMN}" tab-predict --desc_csv "${D}/desc.csv" --minp "${T}/model" --pout "${T}/pred.csv"
    run_step "tab-train-cv" "prepare-data" with_outputs "${T}/cv.csv ${T}/featimp.csv" \
        "${DMN}" tab-train --xmatrix "${D}/desc_small.csv" --ytarget "${D}/target.csv" \
        --epochs "${EPOCHS}" --batch_size 32 --nunits 16 --ndense_layers 1 \
        --n_splits 3 --n_repeats 1 --cvout "${T}/cv.csv" --featimp "${T}/featimp.csv"
fi

# ----------------------------------------------------------------------------
# One-hot encoded SMILES CNN
# ----------------------------------------------------------------------------
if group_enabled ohe; then
    section "One-hot encoded SMILES CNN"
    O="${OUT_DIR}/ohe"
    mkdir -p "${O}"
    OHE_PAD=128
    run_step "ohe-smi2ohe" "prepare-data" with_outputs "${O}/all.ohe.csv" \
        "${DMN}" smi2ohe "${D}/subset.smi" "${OHE_PAD}" "${O}/all.ohe.csv"

    split_ohe() {
        # smi2ohe writes all matrices to one file separated by END lines; split them
        # into one CSV per molecule (named after the SMILES identifier).
        rm -rf "${O}/csv" && mkdir -p "${O}/csv"
        awk -v outdir="${O}/csv" '
            NR==FNR { names[NR] = $0; next }
            FNR==1 || newfile { i++; f = outdir "/" names[i] ".csv"; newfile = 0 }
            { print > f }
            /^END/ { close(f); newfile = 1 }' "${D}/names.txt" "${O}/all.ohe.csv"
        local n
        n=$(find "${O}/csv" -name '*.csv' | wc -l)
        echo "split ${n} molecules"
        [[ ${n} -gt 0 ]]
    }
    run_step "ohe-split" "ohe-smi2ohe" split_ohe
    run_step "ohe-makedb" "ohe-split" with_outputs "${O}/db" \
        "${DMN}" ohe-makedb "${O}/csv" "${O}/db"
    run_step "ohe-train" "ohe-makedb" with_outputs "${O}/model" \
        "${DMN}" ohe-train --ohedir "${O}/db" --ytarget "${D}/target.csv" \
        --epochs "${EPOCHS}" --batch_size 32 --nfilters 8 --nunits 16 --verbose 0 --mout "${O}/model"
    run_step "ohe-predict" "ohe-train" with_outputs "${O}/pred.csv" \
        "${DMN}" ohe-predict --smiles "${D}/subset.smi" --minp "${O}/model" --pout "${O}/pred.csv"
    run_step "ohe-train-multimodal" "ohe-makedb" with_outputs "${O}/model_mm" \
        "${DMN}" ohe-train --ohedir "${O}/db" --ytarget "${D}/target.csv" --xmatrix "${D}/desc_small.csv" \
        --epochs "${EPOCHS}" --batch_size 32 --nfilters 8 --nunits 16 --verbose 0 --mout "${O}/model_mm"
    run_step "ohe-predict-multimodal" "ohe-train-multimodal" with_outputs "${O}/pred_mm.csv" \
        "${DMN}" ohe-predict --smiles "${D}/subset.smi" --minp "${O}/model_mm" \
        --desc_csv "${D}/desc_small.csv" --pout "${O}/pred_mm.csv"
fi

# ----------------------------------------------------------------------------
# 3D voxel CNN (RDKit conformers + Gasteiger charges)
# ----------------------------------------------------------------------------
if group_enabled voxel; then
    section "3D voxel CNN (RDKit / Gasteiger voxels)"
    V="${OUT_DIR}/voxels"
    mkdir -p "${V}"
    VOX_COMMON=(--epochs "${EPOCHS}" --n_rot_train 1 --train_steps_per_epoch 2
        --n_rot_validation 1 --val_steps_per_epoch 1 --nlayers 2 --nunits 16 --nfilters 8)

    run_step "voxel-smi2voxel" "prepare-data" \
        "${DMN}" smi2voxel --smi "${D}/subset.smi" --outdir "${V}/dx" \
        --channels charge,esp,steric,logp --grid_size "${VOXEL_GRID}" --spacing "${VOXEL_SPACING}" \
        --failed_out "${V}/smi2voxel_failed.csv"

    single_channel_dx() {
        rm -rf "${V}/dx_esp" && mkdir -p "${V}/dx_esp"
        find "${V}/dx" -name '*.esp.dx' -exec ln -s {} "${V}/dx_esp/" \;
        [[ -n "$(ls -A "${V}/dx_esp")" ]]
    }
    run_step "voxel-esp-subset" "voxel-smi2voxel" single_channel_dx

    run_step "voxel-makedb" "voxel-smi2voxel" with_outputs "${V}/db" \
        "${DMN}" voxel-makedb --dxdbs "${V}/dx" --multichannel --nrotations "${VOXEL_NROT}" --dbout "${V}/db"
    run_step "voxel-makedb-1ch" "voxel-esp-subset" with_outputs "${V}/db_esp" \
        "${DMN}" voxel-makedb --dxdbs "${V}/dx_esp" --nrotations "${VOXEL_NROT}" --dbout "${V}/db_esp"
    run_step "voxel-makecsv" "voxel-makedb-1ch" with_outputs "${V}/voxels_esp.csv" \
        "${DMN}" voxel-makecsv --activity_csv "${D}/target.csv" --db "${V}/db_esp" --csv_out "${V}/voxels_esp.csv"

    run_step "voxel-train" "voxel-makedb" with_outputs "${V}/model.h5" \
        "${DMN}" voxel-train --db "${V}/db" --target_csv "${D}/target.csv" "${VOX_COMMON[@]}" \
        --mout "${V}/model.h5"
    run_step "voxel-predict" "voxel-train" with_outputs "${V}/pred.csv" \
        "${DMN}" voxel-predict --db "${V}/db" --minp "${V}/model.h5" --n_rotations 2 --pout "${V}/pred.csv"
    run_step "voxel-train-cv" "voxel-makedb" with_outputs "${V}/cv.csv" \
        "${DMN}" voxel-train --db "${V}/db" --target_csv "${D}/target.csv" "${VOX_COMMON[@]}" \
        --n_splits 3 --n_repeats 1 --cvout "${V}/cv.csv" --mout "${V}/cv_models"
    run_step "voxel-train-multimodal" "voxel-makedb" with_outputs "${V}/model_mm.h5" \
        "${DMN}" voxel-train --db "${V}/db" --target_csv "${D}/target.csv" --desc_csv "${D}/desc_small.csv" \
        "${VOX_COMMON[@]}" --mout "${V}/model_mm.h5"
    run_step "voxel-predict-multimodal" "voxel-train-multimodal" with_outputs "${V}/pred_mm.csv" \
        "${DMN}" voxel-predict --db "${V}/db" --minp "${V}/model_mm.h5" --desc_csv "${D}/desc_small.csv" \
        --n_rotations 2 --pout "${V}/pred_mm.csv"
    run_step "voxel-train-1ch" "voxel-makedb-1ch" with_outputs "${V}/model_esp.h5" \
        "${DMN}" voxel-train --db "${V}/db_esp" --target_csv "${D}/target.csv" "${VOX_COMMON[@]}" \
        --mout "${V}/model_esp.h5"

    first_esp_dx() { find "${V}/dx_esp" -name '*.dx' | sort | head -n 1; }
    run_step "voxel-vis-3dcnn" "voxel-train-1ch" with_outputs "${V}/vis" \
        "${DMN}" vis-3dcnn --model "${V}/model_esp.h5" --dx "$(first_esp_dx)" --maxlayer 2 --outdir "${V}/vis"
fi

# ----------------------------------------------------------------------------
# 2D image CNN (RDKit depictions)
# ----------------------------------------------------------------------------
if group_enabled img2d; then
    section "2D image CNN (RDKit depictions)"
    I="${OUT_DIR}/images2d"
    mkdir -p "${I}"

    train_img2d() {
        "${PYTHON}" - "${D}/subset.smi" "${D}/target.csv" "${I}" "${EPOCHS}" <<'PYEOF'
"""Render molecules, train a binary 2D CNN (solubility above median) and save it."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from deepmolecularnetwork.framework import save_model_compat
from deepmolecularnetwork.images2d.model import make_2d_cnn_model

smi_file, target_file, outdir, epochs = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
size = 64
(outdir / "png").mkdir(parents=True, exist_ok=True)
with open(target_file) as fh:
    target = {row[0]: float(row[1]) for row in list(csv.reader(fh))[1:]}

images, values = [], []
for line in open(smi_file):
    smi, name = line.rstrip("\n").split("\t")[:2]
    mol = Chem.MolFromSmiles(smi)
    if mol is None or name not in target:
        continue
    png = outdir / "png" / f"{name}.png"
    Draw.MolToFile(mol, str(png), size=(size, size))
    images.append(np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0)
    values.append(target[name])

x = np.stack(images)
y = (np.asarray(values) > np.median(values)).astype(np.float32)
model = make_2d_cnn_model(input_shape=(size, size, 3), n_classes=1)
hist = model.fit(x, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
print({k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()})
save_model_compat(model, str(outdir / "model2d.h5"))
print(f"trained on {len(x)} images; saved model2d.h5")
PYEOF
    }
    run_step "img2d-train" "prepare-data" with_outputs "${I}/model2d.h5" train_img2d

    first_png() { find "${I}/png" -name '*.png' | sort | head -n 1; }
    run_step "img2d-vis-2dcnn" "img2d-train" \
        "${DMN}" vis-2dcnn --model "${I}/model2d.h5" --img "$(first_png)" --save "${I}/vis"
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
section "Summary"
npass=0 nfail=0 nskip=0
printf '  %-34s %-6s %s\n' "STEP" "STATUS" "TIME"
for i in "${!STEP_NAMES[@]}"; do
    case "${STEP_STATUS[$i]}" in
        PASS) c="${C_OK}"; npass=$((npass + 1)) ;;
        FAIL) c="${C_KO}"; nfail=$((nfail + 1)) ;;
        *) c="${C_SK}"; nskip=$((nskip + 1)) ;;
    esac
    printf '  %-34s %s%-6s%s %s\n' "${STEP_NAMES[$i]}" "${c}" "${STEP_STATUS[$i]}" "${C_0}" "${STEP_TIMES[$i]}"
done
echo
echo "  ${npass} passed, ${nfail} failed, ${nskip} skipped"
echo "  logs: ${LOG_DIR}"

if [[ ${nfail} -eq 0 && ${nskip} -eq 0 && ${KEEP} -eq 0 ]]; then
    rm -rf "${OUT_DIR}"
    echo "  work directory removed"
fi
[[ ${nfail} -eq 0 ]]
