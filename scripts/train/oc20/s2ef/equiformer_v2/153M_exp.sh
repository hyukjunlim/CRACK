# python main_oc20.py \
#     --mode mpflow_validate \
#     --config-yml 'oc20/configs/s2ef/all_md/equiformer_v2/153M_exp.yml' \
#     --run-dir 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@1x1' \
#     --print-every 200 \
#     --amp \
#     --checkpoint 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@1x1/mpflow_exp2_lr5e-4_wu1.pt'


python main_oc20.py \
    --mode mpflow_train \
    --config-yml 'oc20/configs/s2ef/all_md/equiformer_v2/153M_exp.yml' \
    --run-dir 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@1x1' \
    --print-every 200 \
    --amp \
    --checkpoint 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@1x1/eq2_153M_ec4_allmd.pt'