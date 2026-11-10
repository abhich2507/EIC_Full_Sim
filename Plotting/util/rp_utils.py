import os
import numpy as np
import awkward as ak
import uproot

# ==================================================================
# 1. Roman Pot cellID Bitfield Decoder & Geometry Auto-Discovery
# Readout: system:8,assembly:3,layer:4,module:8,sensor:8,x:32:-16,y:-16
# ==================================================================
def decode_rp_cellid(cellID):
    """Vectorized extraction of RP bitfields from uint64 cellID array."""
    sys_id   = (cellID >> 0) & 0xFF
    assembly = (cellID >> 8) & 0x7
    layer    = (cellID >> 11) & 0xF
    module   = (cellID >> 15) & 0xFF
    sensor   = (cellID >> 23) & 0xFF
    return sys_id, assembly, layer, module, sensor


def discover_rp_geometry(files, hit_coll="ForwardRomanPotHits"):
    """
    Scans input files to auto-discover the 4 RP (system, layer) pairs ordered
    by physical Z-position, along with the active LGAD sensor ID.
    Returns:
        stations_info: list of tuples [(sys_id, layer_id, mean_z), ...] for the 4 planes
        active_sensor: int, the sensor bitfield value for the sensitive LGAD
    """
    for fpath in files:
        if not os.path.exists(fpath):
            continue
        try:
            tree = uproot.open(f"{fpath}:events")
            if f"{hit_coll}.cellID" not in tree:
                continue

            cids = ak.flatten(tree[f"{hit_coll}.cellID"].array())
            zs   = ak.flatten(tree[f"{hit_coll}.position.z"].array())
            if len(cids) == 0:
                continue

            sys_ids, _, layers, _, sensors = decode_rp_cellid(cids)

            sys_ids = ak.to_numpy(sys_ids)
            layers  = ak.to_numpy(layers)
            sensors = ak.to_numpy(sensors)
            z_vals  = ak.to_numpy(zs)

            # Active LGAD sensor is the component recording the tracking hits
            active_sensor = int(np.bincount(sensors).argmax())

            unique_pairs = np.unique(np.column_stack((sys_ids, layers)), axis=0)

            pairs_with_z = []
            for (s, l) in unique_pairs:
                mask = (sys_ids == s) & (layers == l) & (sensors == active_sensor)
                if np.any(mask):
                    mean_z = float(np.mean(z_vals[mask]))
                    pairs_with_z.append((int(s), int(l), mean_z))

            # Sort from upstream (Station 1 Layer 1) to downstream (Station 2 Layer 2)
            pairs_with_z.sort(key=lambda x: x[2])

            if len(pairs_with_z) >= 4:
                return pairs_with_z[:4], active_sensor
        except Exception:
            pass

    raise RuntimeError("Could not auto-discover 4 RP layers from the input files.")


# ==================================================================
# 2. Hit Association & Filtering Helpers
# ==================================================================
def validate_assoc(tree, hit_coll, assoc_coll, n_mc):
    """Validates hit-to-MCParticle index associations and prevents out-of-bounds access."""
    idx       = tree[f"{assoc_coll}.index"].array()
    n_hits    = ak.num(tree[f"{hit_coll}.position.x"].array())
    n_assc    = ak.num(idx)
    n_bad_len = int(ak.sum(n_hits != n_assc))
    if n_bad_len:
        raise RuntimeError(f"{assoc_coll}: mismatched association length.")
    in_range = (idx >= 0) & (idx < n_mc)
    safe     = ak.where(in_range, idx, 0)
    return idx, in_range, safe


def get_rp_hit_masks(tree, hit_coll, assoc_coll, n_mc, mc_pdg, mc_is_sig_jag, target_pdg, active_sensor):
    """
    Decodes cellID and builds boolean hit masks for active LGAD hits
    matched to signal MC particles.
    """
    _, valid_idx, safe = validate_assoc(tree, hit_coll, assoc_coll, n_mc)
    cellid = tree[f"{hit_coll}.cellID"].array()
    hit_sys, _, hit_layer, _, hit_sensor = decode_rp_cellid(cellid)

    # Require valid MC index AND hit in the active silicon sensor slice
    valid_active = valid_idx & (hit_sensor == active_sensor)

    pdg      = ak.where(valid_active, mc_pdg[safe], 0)
    pdg_mask = (pdg == target_pdg) & valid_active
    sig_mask = mc_is_sig_jag[safe] & valid_active

    return sig_mask, pdg_mask, safe, hit_sys, hit_layer, (hit_sensor == active_sensor)


def get_xy(tree, hit_coll, mask):
    """Extracts X and Y hit positions filtered by a boolean mask."""
    x = tree[f"{hit_coll}.position.x"].array()[mask]
    y = tree[f"{hit_coll}.position.y"].array()[mask]
    return x, y


def particles_with_hit(safe, sel, offsets, n_mc_total):
    """Maps event-level hit masks back to a flat boolean array over all MCParticles."""
    g   = ak.to_numpy(ak.flatten(safe[sel] + offsets))
    out = np.zeros(n_mc_total, dtype=bool)
    if len(g):
        out[np.unique(g)] = True
    return out


# ==================================================================
# 3. Kinematics & Statistics Helpers
# ==================================================================
def compute_beam_kinematics(tree, crossing_angle=0.025, rot_sign=1.0, p_beam=82.0):
    """Computes total momentum, beam-frame polar angle (mrad), and x_L for MCParticles."""
    px = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.x"].array()))
    py = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.y"].array()))
    pz = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.z"].array()))

    p_tot   = np.sqrt(px**2 + py**2 + pz**2)
    ca, sa  = np.cos(rot_sign * crossing_angle), np.sin(rot_sign * crossing_angle)
    pxb     =  ca * px + sa * pz
    pzb     = -sa * px + ca * pz
    th_beam = np.arctan2(np.hypot(pxb, py), pzb) * 1e3
    xL      = p_tot / p_beam
    return p_tot, th_beam, xL


def binom_err(k, n):
    """Computes binomial uncertainty for efficiency/acceptance."""
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(max(p * (1.0 - p), 0.0) / n)