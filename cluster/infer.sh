chkpt=88417
qsub ./resources/biwi/infer_on_host.sh "/scratch_net/biwidl104/oskopek/mammography/data_out/MaskFalse_BcdrInbreastFilterTrain_NoAugment_ICNR_nnUPSAMPLE_Lam0.0/chook/model.ckpt-${chkpt}" False True nn_upsample_conv 0.0 False "data_out/MaskFalse_BcdrInbreastFilterTrain_NoAugment_ICNR_nnUPSAMPLE_Lam0.0_steps_${chkpt}_inference_eval"

