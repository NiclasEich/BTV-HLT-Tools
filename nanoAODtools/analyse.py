import os
import argparse
import awkward as ak
import uproot as ur
import numpy as np


from tqdm import tqdm
from pathlib import Path
from itertools import product
from plotting import plot_effs, plot_multiple_effs
from selections import ttbar_selection, qcd_selection, trigger_path_selection

import argparse

# import mplhep as hep

from scipy import stats

ANALYSIS_PATHS = [
    "HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepJet_p71",
]
BASE_SELECTION_PATHS = ["HLT_PFHT1050"]


def binomial_ci(x, n, alpha=0.32):
    # x is number of successes, n is number of trials
    if x == 0:
        c1 = 0
    else:
        c1 = stats.beta.interval(1 - alpha, x, n - x + 1)[0]
    if x == n:
        c2 = 1
    else:
        c2 = stats.beta.interval(1 - alpha, x + 1, n - x)[1]
    return x / n - c1, c2 - x / n


def make_selections(
    ur_file: ur.reading.ReadOnlyDirectory, selections=None, selection_names=None
) -> dict:
    """
    selections: list of functions to select events
    """

    events = ur_file["Events"]

    return {sel_name: sel_func(events) for sel_name, sel_func in zip(selection_names, selections)}


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--path",
        "-p",
        metavar="p",
        type=str,
        help="Path of file",
        default="/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STEAM/savarghe/nanoaod/eraD/Fill8136/",
    )
    args = parser.parse_args()

    os.makedirs("plots", exist_ok=True)

    """
    
    Read in file contents
    
    """
    print(f"Trying to open files:\n{args.path}")
    p = Path(args.path)
    dicts = []
    for path in tqdm(list(p.glob("**/*.root"))):
        # print(path)
        with ur.open(path) as ur_file:
            try:
                cut_selections = [ttbar_selection, qcd_selection]
                cut_selection_names = ["ttbar_selection", "HLT_PFHT1050"]
                selections = make_selections(ur_file, cut_selections, cut_selection_names)

                analysis_paths = [lambda x: trigger_path_selection(x, ap) for ap in ANALYSIS_PATHS]
                analysis_path_names = [ap for ap in ANALYSIS_PATHS]

                analyses = make_selections(ur_file, analysis_paths, analysis_path_names)

                tagger_branches = [
                    lambda x: trigger_path_selection(x, ap)
                    for ap in ["Jet_btagDeepFlavB", "Jet_btagDeepB"]
                ]
                tagger_names = ["offline DeepJet", "offline DeepCSV"]
                tagger_cuts = make_selections(ur_file, tagger_branches, tagger_names)

                dicts.append(
                    {"analysis": analyses, "selection": selections, "taggers": tagger_cuts}
                )
            except KeyError as e:
                print("Caught KeyError, skipping file")
                pass
    """
    
    Merge file contents
    
    """

    all_analysis_keys = dicts[0]["analysis"].keys()
    all_selection_keys = dicts[0]["selection"].keys()
    all_taggers_keys = dicts[0]["taggers"].keys()

    all_analysis = dict.fromkeys(all_analysis_keys)
    all_selections = dict.fromkeys(all_selection_keys)
    all_taggers = dict.fromkeys(all_taggers_keys)

    combinations = product(all_analysis_keys, all_selection_keys, all_taggers_keys)

    for akey in all_analysis_keys:
        all_analysis[akey] = ak.concatenate(d["analysis"][akey] for d in dicts)
    for sk in all_selection_keys:
        all_selections[sk] = ak.concatenate(d["selection"][sk] for d in dicts)
    for tk in all_taggers_keys:
        all_taggers[tk] = ak.concatenate(d["taggers"][tk] for d in dicts)

    """
    
    compute efficiencies
    
    """

    b_tag_values = np.linspace(0, 1, 25)

    all_effs = []
    all_errs = []
    all_names = []
    for path_comb in combinations:
        effs = []
        errs = []
        all_names.append(f"{path_comb[1]}_{path_comb[2]}")

        for b_tag_value in b_tag_values:

            base_selection = all_analysis[path_comb[0]] & all_selections[path_comb[1]]
            tagger_selection = ak.any(all_taggers[path_comb[2]] > b_tag_value, axis=-1)

            n_passing = ak.sum(base_selection & tagger_selection)
            n_total = ak.sum(base_selection)
            err = binomial_ci(n_passing, n_total)

            errs.append(err)
            effs.append(n_passing / n_total)
            print("{0:1.2f}:\t {1}/{2}".format(b_tag_value, n_passing, n_total))

        all_effs.append(effs)
        all_errs.append(errs)
        plot_effs(
            b_tag_values,
            effs,
            error=errs,
            path="plots/efficiencies_{}.png".format("__".join(path_comb[1:])),
        )

    plot_multiple_effs(
        b_tag_values,
        all_effs,
        errors=all_errs,
        names=all_names,
        path="plots/efficiencies_all.png".format("__".join(path_comb)),
    )


if __name__ == "__main__":
    main()
