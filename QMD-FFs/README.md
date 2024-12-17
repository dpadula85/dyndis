A database of Quantum Mechanically Derived Force Fields (QMD-FFs)
parameterised with the Joyce software, for use in GROMACS for research
purposes.

For each molecule we provide:

- a starting geometry in `.xyz` format. The numbering comes from the `.cif`
  file reported in the paper.

- an include topology in `.itp` format, containing the definition of the
  molecule, which follows the numbering in the `.xyz` file, and the parameters
  resulting from the fitting.

- a topology in `.top` format, containing definition of atom types, assignment
  of LJ parameters taken from OPLS.
