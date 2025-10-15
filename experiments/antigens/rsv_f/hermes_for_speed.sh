
results_dir='./results_all_sites/'

echo '4jhw_trimer A' > 'pdb_and_chain.txt'

SECONDS=0

python -u ../../../run_hermes_on_pdbfiles.py \
                -m hermes_py_050 \
                -r logits \
                -pd ./pdbs \
                -pn pdb_and_chain.txt \
                -o $results_dir'hermes_py_050.csv' \
                -pp 0

duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."

rm 'pdb_and_chain.txt'
