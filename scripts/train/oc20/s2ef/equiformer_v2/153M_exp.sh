# python main_oc20.py \
#     --mode mpflow_validate \
#     --config-yml 'oc20/configs/s2ef/200k/equiformer_v2/153M_exp.yml' \
#     --run-dir 'models' \
#     --print-every 200 \
#     --amp \
#     --checkpoint 'save_models/eq2_153M_ec4_allmd.pt'


python main_oc20.py \
    --mode mpflow_train \
    --config-yml 'oc20/configs/s2ef/200k/equiformer_v2/153M_exp.yml' \
    --run-dir 'models' \
    --print-every 200 \
    --amp \
    --checkpoint 'save_models/eq2_153M_ec4_allmd.pt'