1. The real SSE can be obtained using mkdssp.
```bash
mkdssp -i xxx_protein.pdb -o xxx_protein.dssp
```
2. The predicted SSE can be calculated using SCRATCH tool, more details can be referred the website https://download.igb.uci.edu/.
3. The ligand smiles can be obtained using Open Babel.
```bash
babel -i sdf xxx_ligand.sdf -o smi xxx.smi
```
