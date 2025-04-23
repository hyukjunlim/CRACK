# python main_oc20.py \
#     --mode validate \
#     --config-yml 'oc20/configs/s2ef/2M/equiformer_v2/83M_exp.yml' \
#     --run-dir 'models' \
#     --print-every 200 \
#     --amp \
#     --checkpoint 'save_models/eq2_83M_2M.pt'

python main_oc20.py \
    --mode train \
    --config-yml 'oc20/configs/s2ef/2M/equiformer_v2/83M_exp.yml' \
    --run-dir 'models' \
    --print-every 200 \
    --amp \
    --checkpoint 'save_models/eq2_83M_2M.pt'