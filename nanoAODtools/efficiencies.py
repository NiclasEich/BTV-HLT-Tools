import os
import argparse
import awkward as ak
import uproot as ur
import numpy as np


from tqdm import tqdm
from pathlib import Path
from itertools import product
from plotting import plot_effs, plot_multiple_effs
from selections import ttbar_selection, qcd_selection, trigger_path_selection, reduce_and

import argparse

# import mplhep as hep

from scipy import stats

ANALYSIS_PATHS = [
    # "HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepJet_p71",
    "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepJet_1p5"
]
BASE_SELECTION_PATHS = [
    "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30",
    # "HLT_PFHT1050"
    ]


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

    os.makedirs("efficiencies", exist_ok=True)
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
                cut_selections = [ttbar_selection] + [lambda x: trigger_path_selection(x, ap) for ap in BASE_SELECTION_PATHS]
                cut_selection_names = ["ttbar_selection"]  + BASE_SELECTION_PATHS
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

    combinations = product(all_analysis_keys, all_taggers_keys)

    for akey in all_analysis_keys:
        all_analysis[akey] = ak.concatenate(d["analysis"][akey] for d in dicts)
    for sk in all_selection_keys:
        all_selections[sk] = ak.concatenate(d["selection"][sk] for d in dicts)
    for tk in all_taggers_keys:
        all_taggers[tk] = ak.concatenate(d["taggers"][tk] for d in dicts)

    """
    
    compute efficiencies
    
    """
    b_tag_values = np.linspace(0, 1, 12)
    b_tag_bins = np.array( list(zip(b_tag_values[:-1], b_tag_values[1:])))
    b_tag_centers = np.array( [val + diff/2 for val, diff in zip(b_tag_values[:-1], np.diff(b_tag_bins, axis=-1))])


    all_effs = []
    all_errs = []
    all_names = []
    for base_analysis, base_tagger in combinations:
        effs = []
        errs = []
        all_names.append(f"{base_tagger}")
        print()
        print(f"_{base_tagger}")
        for b_tag_low, b_tag_high in b_tag_bins:

            base_sel = reduce_and(*all_selections.values())
            path_selection = all_analysis[base_analysis]

            tagger_sel_high = ak.max(all_taggers[base_tagger], axis=-1) < b_tag_high
            tagger_sel_low = ak.max(all_taggers[base_tagger], axis=-1) > b_tag_low

            lose_wp = ak.sum( all_taggers[base_tagger] > 0.2, axis=-1) >= 2


            tagger_sel = tagger_sel_high & tagger_sel_low 

            n_passing = ak.sum( tagger_sel & path_selection & base_sel & lose_wp)
            n_total = ak.sum( tagger_sel & base_sel & lose_wp)


            err = binomial_ci(n_passing, n_total)

            errs.append(err)
            effs.append(n_passing / n_total)
            print("{0:1.2f}:\t {1}/{2}".format(b_tag_low, n_passing, n_total))

        all_effs.append(effs)
        all_errs.append(errs)
        plot_effs(
            b_tag_centers,
            effs,
            error=errs,
            path="efficiencies/efficiencies_{}.png".format("__".join([base_analysis, base_tagger])),
        )

    plot_multiple_effs(
        b_tag_centers,
        all_effs,
        errors=all_errs,
        names=all_names,
        path="efficiencies/efficiencies_all.png".format("__".join([base_analysis, base_tagger])),
    )



if __name__ == "__main__":
    main()

