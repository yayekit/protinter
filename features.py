from typing import Dict, List
from Bio.SeqUtils import ProtParam
from Bio.Seq import Seq

def extract_features(sequence: Seq) -> Dict[str, float]:
    """Extract features from a protein sequence."""
    analyser = ProtParam.ProteinAnalysis(str(sequence))
    amino_acid_percent = analyser.get_amino_acids_percent()
    return {
        'length': len(sequence),
        'weight': analyser.molecular_weight(),
        'aromaticity': analyser.aromaticity(),
        'instability': analyser.instability_index(),
        'isoelectric_point': analyser.isoelectric_point(),
        'helix_fraction': analyser.secondary_structure_fraction()[0],
        'turn_fraction': analyser.secondary_structure_fraction()[1],
        'sheet_fraction': analyser.secondary_structure_fraction()[2],
        'gravy': analyser.gravy(),
        **{f'{aa}_percent': percent for aa, percent in amino_acid_percent.items()}
    }

def compute_conjoint_triad(sequence: str) -> List[int]:
    """Compute Conjoint Triad features."""
    groups = {'A': 0, 'G': 0, 'V': 0,
              'I': 1, 'L': 1, 'F': 1, 'P': 1,
              'Y': 2, 'M': 2, 'T': 2, 'S': 2,
              'H': 3, 'N': 3, 'Q': 3, 'W': 3,
              'R': 4, 'K': 4,
              'D': 5, 'E': 5,
              'C': 6}
    features = [0] * 343  # 7^3 possible triads

    for i in range(len(sequence) - 2):
        triad = (groups[sequence[i]], groups[sequence[i+1]], groups[sequence[i+2]])
        features[triad[0]*49 + triad[1]*7 + triad[2]] += 1

    return features