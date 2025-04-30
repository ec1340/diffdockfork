

def coordinates_to_pdb(coordinates, sequence):
    """
    Convert coordinates array to PDB format string
    Args:
        coordinates: Nx37x3 array of atom coordinates
        sequence: protein sequence string
    Returns:
        PDB format string
    """
    pdb_lines = []
    atom_counter = 1
    
    # Standard amino acid 3-letter codes
    aa_codes = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    
    # Main backbone atoms (first 3 atoms for each residue)
    backbone_atoms = ['N', 'CA', 'C']
    
    for residue_idx, residue_coords in enumerate(coordinates):
        residue_num = residue_idx + 1
        residue = sequence[residue_idx]
        residue_name = aa_codes[residue]
        
        # Add backbone atoms
        for atom_idx, atom_name in enumerate(backbone_atoms):
            x, y, z = residue_coords[atom_idx]
            pdb_line = (f"ATOM  {atom_counter:5d}  {atom_name:<3s} {residue_name:3s} "
                       f"A{residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
            pdb_lines.append(pdb_line)
            atom_counter += 1
    
    pdb_lines.append("END")
    return "\n".join(pdb_lines)
