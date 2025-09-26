
# model_version_list='hermes_py_000 hermes_py_000_ft_cdna117k_relaxed_pred hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_cdna117k_relaxed_pred hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st'
model_version_list='hermes_py_000_ft_mega_thermompnn_ddg_st hermes_py_050_ft_mega_thermompnn_ddg_st hermes_py_000_ft_cdna117k_relaxed_pred_ft_mega_thermompnn_ddg_st hermes_py_050_ft_cdna117k_relaxed_pred_ft_mega_thermompnn_ddg_st'

parallelism=0

results_dir='./results_all_sites/'
mkdir -p $results_dir

# make temporary file with pdb 4jhw and chain F
echo '5WB0_trimer A' > 'pdb_and_chain.txt'

for model_version in $model_version_list
    do

    echo $model_version

    python -u ../../../run_hermes_on_pdbfiles.py \
                    -m $model_version \
                    -r logits \
                    -pd ./pdbs \
                    -o $results_dir$model_version'.csv' \
                    -pp $parallelism
    
    python -u keep_only_desired_mutations.py \
                    -m $model_version

done

# remove file
rm 'pdb_and_chain.txt'
