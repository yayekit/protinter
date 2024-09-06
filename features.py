from typing import Dict, List
from Bio.SeqUtils import ProtParam
from Bio.Seq import Seq

def extract_features(sequence: Seq) -> Dict[str, float]:
    """Extract features from a protein sequence."""
    analyser = ProtParam.ProteinAnalysis(str(sequence))
    amino_acid_percent = analyser.get_amino_acids_percent()
    amino_acid_count = analyser.count_amino_acids()
    
    # New features
    charge = calculate_charge(sequence)
    hydrophobic_ratio = sum(amino_acid_count[aa] for aa in 'AILMFPWV') / len(sequence)
    
    # Additional features
    aliphatic_index = calculate_aliphatic_index(amino_acid_percent)
    boman_index = calculate_boman_index(sequence)
    hydrophobicity = calculate_hydrophobicity(sequence)
    
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
        'charge': charge,
        'hydrophobic_ratio': hydrophobic_ratio,
        'flexibility': analyser.flexibility(),
        'aliphatic_index': aliphatic_index,  # New feature
        'boman_index': boman_index,  # New feature
        'hydrophobicity': hydrophobicity,  # New feature
        **{f'{aa}_percent': percent for aa, percent in amino_acid_percent.items()}
    }

def calculate_charge(sequence: Seq, pH: float = 7.0) -> float:
    """Calculate the net charge of a protein at a given pH."""
    analyser = ProtParam.ProteinAnalysis(str(sequence))
    return analyser.charge_at_pH(pH)

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

def calculate_aliphatic_index(amino_acid_percent: Dict[str, float]) -> float:
    """Calculate the aliphatic index of a protein sequence."""
    return (100 * amino_acid_percent.get('A', 0) +
            2.9 * amino_acid_percent.get('V', 0) +
            3.9 * (amino_acid_percent.get('I', 0) + amino_acid_percent.get('L', 0)))

def calculate_boman_index(sequence: Seq) -> float:
    """Calculate the Boman (Potential Protein Interaction) index."""
    aa_values = {'L': -4.92, 'I': -4.92, 'V': -4.04, 'F': -2.98, 'M': -2.35,
                 'W': -2.33, 'A': -1.89, 'C': -1.85, 'G': -1.05, 'Y': -0.14,
                 'T': 0.69, 'S': 0.84, 'H': 2.06, 'Q': 2.36, 'K': 2.71,
                 'N': 2.95, 'E': 3.81, 'D': 3.98, 'R': 4.38, 'P': 0}
    return sum(aa_values.get(aa, 0) for aa in str(sequence)) / len(sequence)

def calculate_hydrophobicity(sequence: Seq) -> float:
    """Calculate the overall hydrophobicity of a protein sequence using the Kyte-Doolittle scale."""
    kd_scale = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    return sum(kd_scale.get(aa, 0) for aa in str(sequence)) / len(sequence)