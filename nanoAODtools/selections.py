import awkward as ak
import uproot as ur
import uproot as ur
from functools import reduce
from operator import and_, or_
from coffea.nanoevents.methods import candidate


def reduce_and(*what):
    return reduce(and_, what)


def reduce_or(*what):
    return reduce(or_, what)


def qcd_selection(events: ur.models.TTree.Model_TTree_v20) -> ak.Array:
    """
    returns a mask with a trigger path that has approx. QCD
    """
    return events["HLT_PFHT1050"].array() & jet_selection(events)


def ttbar_selection(events: ur.models.TTree.Model_TTree_v20) -> ak.Array:
    """
    returns a mask with a ttbar selection for a b-enriched phase space
    """

    muon = ak.zip(
        {
            "pt": events["Muon_pt"].array(),
            "eta": events["Muon_eta"].array(),
            "phi": events["Muon_phi"].array(),
            "mass": events["Muon_mass"].array(),
            "charge": events["Muon_charge"].array(),
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    electron = ak.zip(
        {
            "pt": events["Electron_pt"].array(),
            "eta": events["Electron_eta"].array(),
            "phi": events["Electron_phi"].array(),
            "mass": events["Electron_mass"].array(),
            "charge": events["Electron_charge"].array(),
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    electron_mask_n = events["nElectron"].array() == 1
    electron_mask_pt = ak.any(events["Electron_pt"].array() >= 10.0, axis=-1)
    electron_mask_eta = ak.any(abs(events["Electron_eta"].array()) <= 2.5, axis=-1)
    electron_mask_dz = ak.any(events["Electron_dz"].array() < 0.2, axis=-1)
    electron_mask_dxy = ak.any(events["Electron_dxy"].array() < 0.1, axis=-1)

    electron_mask = reduce_and(
        electron_mask_n,
        electron_mask_pt,
        electron_mask_eta,
        electron_mask_dz,
        electron_mask_dxy,
    )

    muon_mask_n = events["nMuon"].array() == 1
    muon_mask_pt = ak.any(events["Muon_pt"].array() >= 10.0, axis=-1)
    muon_mask_eta = ak.any(abs(events["Muon_eta"].array()) <= 2.5, axis=-1)
    muon_mask_dz = ak.any(events["Muon_dz"].array() < 0.2, axis=-1)
    muon_mask_dxy = ak.any(events["Muon_dxy"].array() < 0.1, axis=-1)

    muon_mask = reduce_and(
        muon_mask_n,
        muon_mask_pt,
        muon_mask_eta,
        muon_mask_dz,
        muon_mask_dxy,
    )

    jet_mask_n = events["nJet"].array() >= 2
    jet_mask_pt = ak.any(events["Jet_pt"].array() > 30.0, axis=-1)

    jet_mask = reduce_and(jet_mask_n, jet_mask_pt)
    tot_mask = reduce_and(electron_mask, muon_mask, jet_mask)

    charge_mask = ak.any(
        ak.mask(events["Electron_charge"].array(), tot_mask)
        + ak.mask(events["Muon_charge"].array(), tot_mask)
        == 0,
        axis=-1,
    )

    tot_mask = tot_mask & charge_mask

    dilep = ak.mask(electron, tot_mask) + ak.mask(muon, tot_mask)
    mass_mask = dilep.mass > 20

    tot_mask = tot_mask & mass_mask & jet_selection(events)

    return tot_mask


def jet_selection(events: ur.models.TTree.Model_TTree_v20) -> ak.Array:
    """
    returns a mask for a basic jet selection
    """

    jet_mask_n = events["nJet"].array() >= 2
    jet_mask_pt = events["Jet_pt"].array() > 30.0

    jet_mask = ak.any(reduce_and(jet_mask_n, jet_mask_pt), axis=-1)

    return jet_mask


def trigger_path_selection(events: ur.models.TTree.Model_TTree_v20, analysis_path: str) -> ak.Array:
    """
    return a mask with the fired trigger paths
    """
    return events[analysis_path].array()
