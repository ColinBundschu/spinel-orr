import itertools
from ml.adsorbates_encoding import AdsorbatesEncoding

N_STEPS = 4

ENCODINGS = (
    # A: Whether or not the adsorbate binds to a Tetrahedral site (No, Yes)
    # B: Whether or not the adsorbate binds to an Octahedral site (No, Yes)
    # H: The number of hydrogen atoms in the adsorbate (counting the spectator hydrogen)
    # O: The number of oxygen atoms in the adsorbate
    # a: The attachment configuration of the adsorbates (None, atop, bridge, separate, split)
    # b: Whether or not the adsorbate binds to an Octahedral2 or Octahedral3 site (None, Oct2, Oct3)
    # d: The number of oxygen double bonds in the adsorbate
    # g: Molecular geometry (None, *O, *OH, *OO, *OOH, *O + *O, *OO*, *(O)(O), *(OH)(O), *(OH)(OH))
    # h: Whether or not a spectator hydrogen is present (None, O-*H)
    # i: The index of the adsorbate ignoring spectator hydrogen
    # j: The index of the adsorbate overall
    # s: Binding site (None, Tet, Oct, TetOct, OctOct2, OctOct3, TetO)

    # A  B  H  O  a  b  d  g  h  i   j   s
    ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0), 1, 'clean'),
    ((0, 0, 1, 0, 0, 0, 0, 0, 1, 1,  0,  0), 2, 'O-xH'),
    ((0, 1, 0, 1, 1, 0, 0, 1, 0, 2,  1,  2), 3, 'Oct-xO'),
    ((0, 1, 1, 1, 1, 0, 0, 1, 1, 3,  1,  2), 0, 'O-xH_Oct-xO'),
    ((1, 0, 0, 1, 1, 0, 0, 1, 0, 4,  2,  1), 3, 'Tet-xO'),
    ((1, 0, 1, 1, 1, 0, 0, 1, 1, 5,  2,  1), 0, 'O-xH_Tet-xO'),
    ((0, 1, 1, 1, 1, 0, 0, 2, 0, 6,  3,  2), 0, 'Oct-xOH'),
    ((0, 1, 2, 1, 1, 0, 0, 2, 1, 7,  3,  2), 1, 'O-xH_Oct-xOH'),
    ((1, 0, 1, 1, 1, 0, 0, 2, 0, 8,  4,  1), 0, 'Tet-xOH'),
    ((1, 0, 2, 1, 1, 0, 0, 2, 1, 9,  4,  1), 1, 'O-xH_Tet-xOH'),
    ((1, 0, 1, 1, 2, 0, 0, 2, 0, 10, 5,  6), 0, 'TetO-xOH'),
    ((1, 0, 2, 1, 2, 0, 0, 2, 1, 11, 5,  6), 1, 'O-xH_TetO-xOH'),
    ((0, 1, 0, 2, 1, 0, 0, 3, 0, 12, 6,  2), 1, 'Oct-xOO'),
    ((0, 1, 1, 2, 1, 0, 1, 3, 1, 13, 6,  2), 2, 'O-xH_Oct-xOO'),
    ((1, 0, 0, 2, 1, 0, 1, 3, 0, 14, 7,  1), 1, 'Tet-xOO'),
    ((1, 0, 1, 2, 1, 0, 1, 3, 1, 15, 7,  1), 2, 'O-xH_Tet-xOO'),
    ((0, 1, 1, 2, 1, 0, 1, 4, 0, 16, 8,  2), 2, 'Oct-xOOH'),
    ((0, 1, 2, 2, 1, 0, 1, 4, 1, 17, 8,  2), 3, 'O-xH_Oct-xOOH'),
    ((1, 0, 1, 2, 1, 0, 1, 4, 0, 18, 9,  1), 2, 'Tet-xOOH'),
    ((1, 0, 2, 2, 1, 0, 1, 4, 1, 19, 9,  1), 3, 'O-xH_Tet-xOOH'),
    ((1, 1, 0, 2, 2, 0, 1, 6, 0, 20, 10, 3), 1, 'TetOct2B-xOOx'),
    ((1, 1, 1, 2, 2, 0, 1, 6, 1, 21, 10, 3), 2, 'O-xH_TetOct2B-xOOx'),
    ((0, 1, 0, 2, 3, 1, 0, 5, 0, 22, 11, 4), 1, 'Oct-xO_Oct2-xO'),
    ((0, 1, 1, 2, 3, 1, 0, 5, 1, 23, 11, 4), 2, 'O-xH_Oct-xO_Oct2-xO'),
    ((0, 1, 0, 2, 3, 2, 0, 5, 0, 24, 12, 5), 1, 'Oct-xO_Oct3-xO'),
    ((0, 1, 1, 2, 3, 2, 0, 5, 1, 25, 12, 5), 2, 'O-xH_Oct-xO_Oct3-xO'),
    ((1, 1, 0, 2, 3, 0, 0, 5, 0, 26, 13, 3), 1, 'Oct-xO_Tet-xO'),
    ((1, 1, 1, 2, 3, 0, 0, 5, 1, 27, 13, 3), 2, 'O-xH_Oct-xO_Tet-xO'),
    ((0, 1, 0, 2, 2, 1, 1, 6, 0, 28, 14, 4), 1, 'OctOctO2B-xOOx'),
    ((0, 1, 1, 2, 2, 1, 1, 6, 1, 29, 14, 4), 2, 'O-xH_OctOctO2B-xOOx'),
    ((0, 1, 0, 2, 2, 2, 1, 6, 0, 30, 15, 5), 1, 'OctOctT2B-xOOx'),
    ((0, 1, 1, 2, 2, 2, 1, 6, 1, 31, 15, 5), 2, 'O-xH_OctOctT2B-xOOx'),
    ((1, 0, 0, 2, 4, 0, 0, 7, 0, 32, 16, 1), 1, 'TetA-xO_TetB-xO'),
    ((1, 0, 1, 2, 4, 0, 0, 8, 0, 33, 17, 1), 2, 'TetA-xOH_TetB-xO'),
    ((1, 0, 2, 2, 4, 0, 0, 8, 1, 34, 17, 1), 3, 'O-xH_TetA-xOH_TetB-xO'),
    ((1, 0, 2, 2, 4, 0, 0, 9, 0, 35, 18, 1), 3, 'TetA-xOH_TetB-xOH'),
    ((1, 0, 3, 2, 4, 0, 0, 9, 1, 36, 18, 1), 0, 'O-xH_TetA-xOH_TetB-xOH'),
    ((0, 1, 2, 2, 3, 1, 0, 5, 0, 37, 19, 4), 3, 'Oct-xOH_Oct2-xOH'),
    ((0, 1, 3, 2, 3, 1, 0, 5, 1, 38, 19, 4), 0, 'O-xH_Oct-xOH_Oct2-xOH'),
    ((0, 1, 2, 2, 3, 2, 0, 5, 0, 39, 20, 5), 3, 'Oct-xOH_Oct3-xOH'),
    ((0, 1, 3, 2, 3, 2, 0, 5, 1, 40, 20, 5), 0, 'O-xH_Oct-xOH_Oct3-xOH'),
    ((1, 1, 2, 2, 3, 0, 0, 5, 0, 41, 21, 3), 3, 'Oct-xOH_Tet-xOH'),
    ((1, 1, 3, 2, 3, 0, 0, 5, 1, 42, 21, 3), 0, 'O-xH_Oct-xOH_Tet-xOH'),
)

ENCODING_LETTERS = 'ABHOabdghijs'
assert(len(ENCODINGS[0][0]) == len(ENCODING_LETTERS))

def create_encoding(specifier: str) -> AdsorbatesEncoding:
    if not all(letter in ENCODING_LETTERS for letter in specifier):
        raise ValueError(f'One or more letters in "{specifier}" are not defined in the ENCODING_LETTERS.')

    if ''.join(sorted(specifier)) != specifier:
        raise ValueError(f'The letters in "{specifier}" are not sorted.')

    # Collect encodings based on the specifier
    specified_encodings = [
        (tuple([encodings[ENCODING_LETTERS.index(letter)] for letter in specifier]), step_index, ad)
        for encodings, step_index, ad in ENCODINGS
    ]

    # Ensure each adsorbate has a unique encoding tuple
    encodings_only = [encoding for encoding, _, _ in specified_encodings]
    if len(set(encodings_only)) != len(encodings_only):
        raise ValueError('The tuples for adsorbates are not unique.')

    return AdsorbatesEncoding(specifier, N_STEPS, *zip(*specified_encodings))


def encodings_length_n(n: int) -> tuple[AdsorbatesEncoding, ...]:
    smaller_encodings = [encodings_length_n(i) for i in range(1, n)]
    valid_encodings = []
    for specifier in itertools.combinations(ENCODING_LETTERS, n):
        # If a smaller complete specifier subset exists, this is not adding anything new
        if any(all(s in specifier for s in enc.name) for enc_list in smaller_encodings for enc in enc_list):
            continue

        try:
            valid_encodings.append(create_encoding(''.join(specifier)))
        except ValueError as e:
            # Skip the encoding if adsorbates do not have unique encoding tuples.
            if str(e) != 'The tuples for adsorbates are not unique.':
                raise
    return tuple(valid_encodings)
