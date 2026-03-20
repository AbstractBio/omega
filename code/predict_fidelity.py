
from typing import Iterable
from itertools import chain
from Bio.Seq import Seq

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SantaLucia 1998 unified nearest-neighbor parameters for DNA/DNA duplexes
# (1 M NaCl).  Keys are 5'->3' sense-strand dinucleotides.
# Values: (delta_H kcal/mol, delta_S cal/mol/K)
# ---------------------------------------------------------------------------
_NN_H_S: dict[str, tuple[float, float]] = {
    'AA': (-7.9, -22.2), 'TT': (-7.9, -22.2),
    'AT': (-7.2, -20.4),
    'TA': (-7.2, -21.3),
    'CA': (-8.5, -22.7), 'TG': (-8.5, -22.7),
    'GT': (-8.4, -22.4), 'AC': (-8.4, -22.4),
    'CT': (-7.8, -21.0), 'AG': (-7.8, -21.0),
    'GA': (-8.2, -22.2), 'TC': (-8.2, -22.2),
    'CG': (-10.6, -27.2),
    'GC': (-9.8,  -24.4),
    'GG': (-8.0, -19.9), 'CC': (-8.0, -19.9),
}

# Initiation penalties for the terminal base pairs.
_INIT_H_S: dict[str, tuple[float, float]] = {
    'A': (2.3,  4.1), 'T': (2.3,  4.1),
    'G': (0.1, -2.8), 'C': (0.1, -2.8),
}


def correct_ligations(site: str, wc_site: str, data_matrix: pd.DataFrame) -> int:
    """Retrieves number of correct ligations observed for a GG site."""

    return data_matrix.loc[site, wc_site]


def total_ligations(site: str, sites: Iterable[str], data_matrix: pd.DataFrame) -> int:
    """Retrieves total number of ligations observed for a set of GG sites
    in reference to a query golden gate site.
    """
    wc_sites = [str(Seq(s).reverse_complement()) for s in sites]

    return data_matrix.loc[site, np.concatenate((sites, wc_sites))].sum()


def site_probability(site: str, sites: Iterable[str], data_matrix: pd.DataFrame) -> float:
    """Calculate probability of site orthogonality in reference to proposed set of sites."""

    wc_site = str(Seq(site).reverse_complement())

    # get correct ligations for the site and it's watson crick pair
    site_correct_ligations = correct_ligations(site=site, wc_site=wc_site, data_matrix=data_matrix)
    wc_correct_ligations = correct_ligations(site=wc_site, wc_site=site, data_matrix=data_matrix)

    # total correct ligations for both the site and WC pair
    correct_total = site_correct_ligations + wc_correct_ligations

    # all liagation events
    total = total_ligations(site, sites, data_matrix) + total_ligations(wc_site, sites, data_matrix)

    return correct_total / total


def predict_fidelity(sites: Iterable[str], data: pd.DataFrame) -> float:
    """Calculate the predicted fidelity for a set of GG sites."""

    return np.prod([site_probability(site, sites, data) for site in sites])

def predict_minimum_site(sites: Iterable[list[str]], data: pd.DataFrame) -> float:
    """Get least orthogonal site from set."""

    return min(site_probability(site, sites, data) for site in sites)

def geneset_fidelity(gene_sites: Iterable[list[str]], data: pd.DataFrame) -> float:
    """Get fidelities for every gene in a set of genes (typically a pool).
    
    Args:
        gene_sites (Iterable): an iterable containing a list-like container of gene sets.

    Returns:
        A list of fidelities whose index matches the index of the gene. 
    """

    pool_sites = list(chain(*gene_sites))

    return [
        np.prod([site_probability(site, pool_sites, data) for site in sets]) for sets in gene_sites
    ]


def predict_minimum(gene_sites: Iterable[np.ndarray], data: pd.DataFrame) -> float:
    """Return minimum fidelity of things in a pool.
    
    Args:
        pool_sites (Iterable[np.ndarray]): iterable of ndarray's containing gg sites.
                                           each array corresponds to a gene.

    Returns:
        Lowest fidelity for a gene in a pool.
    
    """
    return min(geneset_fidelity(gene_sites, data))


def predict_average(gene_sites: Iterable[np.ndarray], data: pd.DataFrame) -> float:
    """Return average fidelity for genes in a pool.

    Args:
        pool_sites (Iterable[np.ndarray]): iterable of ndarray's containing gg sites.
                                           each array corresponds to a gene.

    Returns:
        Average fidelity for a gene in a pool.
    """

    return sum(geneset_fidelity(gene_sites, data)) / len(gene_sites)


def overhang_gc_score(sites: Iterable[str]) -> float:
    """Score a set of GG overhangs on GC balance.

    Each site is scored as 1.0 when GC content is exactly 50% and 0.0 when
    it is 0% or 100% GC.  The returned value is the mean across all sites.

    Args:
        sites: Iterable of GG overhang sequences (equal-length strings).

    Returns:
        Mean GC-balance score in [0, 1].
    """
    scores = [
        1.0 - abs((s.count('G') + s.count('C')) / len(s) - 0.5) * 2
        for s in sites
    ]
    return float(np.mean(scores)) if scores else 0.0


def overhang_dG(site: str, temp_C: float = 37.0) -> float:
    """Estimate ΔG (kcal/mol) for a short DNA overhang duplex.

    Uses SantaLucia 1998 unified nearest-neighbor parameters with terminal
    initiation corrections.  Suitable for 4-mer Golden Gate overhangs.

    Args:
        site: Overhang sequence (e.g. 'ACGT').  Case-insensitive.
        temp_C: Temperature in °C.  Defaults to 37 °C.

    Returns:
        ΔG in kcal/mol (more negative = more stable / stronger binding).
    """
    site = site.upper()
    temp_K = temp_C + 273.15

    dH = sum(_NN_H_S[site[i:i+2]][0] for i in range(len(site) - 1))
    dS = sum(_NN_H_S[site[i:i+2]][1] for i in range(len(site) - 1))

    ih_5, is_5 = _INIT_H_S[site[0]]
    ih_3, is_3 = _INIT_H_S[site[-1]]
    dH += ih_5 + ih_3
    dS += is_5 + is_3

    return dH - temp_K * (dS / 1000.0)


def overhang_uniformity_score(sites: Iterable[str], temp_C: float = 37.0) -> float:
    """Score a set of GG overhangs on thermodynamic uniformity.

    A perfectly uniform set (all overhangs identical ΔG) scores 1.0.
    The score decreases as the standard deviation of ΔG values across sites
    grows, following 1 / (1 + σ).  This penalises overhang sets where some
    sites bind much more strongly than others, which correlates with biased
    final product formation in pooled GG assembly.

    Args:
        sites: Iterable of GG overhang sequences.
        temp_C: Temperature for ΔG calculation (default 37 °C).

    Returns:
        Uniformity score in (0, 1].
    """
    site_list = list(sites)
    if len(site_list) <= 1:
        return 1.0
    dg_values = np.array([overhang_dG(s, temp_C) for s in site_list])
    return 1.0 / (1.0 + float(np.std(dg_values)))


def overhang_thermo_stats(sites: Iterable[str], temp_C: float = 37.0) -> dict[str, float]:
    """Compute thermodynamic summary statistics for a set of GG overhangs.

    Provides metrics useful for correlating thermodynamic properties of an
    optimized junction set with experimental assembly performance.  A large
    range_dG or std_dG indicates highly variable overhang strength, which
    can drive biased product formation.

    Args:
        sites: Iterable of GG overhang sequences.
        temp_C: Temperature for ΔG calculation (default 37 °C).

    Returns:
        Dict with keys:
            mean_dG          – mean ΔG across all sites (kcal/mol)
            std_dG           – std dev of ΔG; the uniformity penalty term
            min_dG           – most stable/strongest site (most negative)
            max_dG           – least stable/weakest site (least negative)
            range_dG         – max_dG - min_dG; total spread in stability
            uniformity_score – 1 / (1 + std_dG); 1.0 = perfectly uniform
    """
    site_list = list(sites)
    if not site_list:
        nan = float('nan')
        return {
            'mean_dG': nan, 'std_dG': nan, 'min_dG': nan,
            'max_dG': nan, 'range_dG': nan, 'uniformity_score': nan
        }
    dg = np.array([overhang_dG(s, temp_C) for s in site_list])
    std = float(np.std(dg))
    return {
        'mean_dG': float(np.mean(dg)),
        'std_dG': std,
        'min_dG': float(np.min(dg)),
        'max_dG': float(np.max(dg)),
        'range_dG': float(np.max(dg) - np.min(dg)),
        'uniformity_score': 1.0 / (1.0 + std),
    }
