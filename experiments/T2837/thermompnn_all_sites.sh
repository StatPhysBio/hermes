

for file in ./pdbs/*
    do

    # assuming chain A only for simplicity... by eye most pdbs are monomers with chain A only anyway

    python /gscratch/spe/gvisan01/ThermoMPNN/analysis/custom_inference.py \
                --pdb $file \
                --chain A \
                --model_path /gscratch/spe/gvisan01/ThermoMPNN/models/thermoMPNN_default.pt \
                --out_dir ./results_all_sites/thermompnn

done

