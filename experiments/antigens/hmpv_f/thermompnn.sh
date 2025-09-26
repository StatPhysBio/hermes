

python /gscratch/spe/gvisan01/ThermoMPNN/analysis/custom_inference.py \
            --pdb ./pdbs/5WB0_trimer.pdb \
            --chain A \
            --model_path /gscratch/spe/gvisan01/ThermoMPNN/models/thermoMPNN_default.pt \
            --out_dir ./results_all_sites

python parse_predictions_thermompnn.py
