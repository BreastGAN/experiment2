chkpt=0
./local/infer_on_host.sh "data_out/MaskFalse_BcdrInbreastFilterTrain_NoAugment_ICNR_nnUPSAMPLE_Lam0.0_SpectralNorm/chook/model.ckpt-${chkpt}" False True nn_upsample_conv 0.0 False "data_out/MaskFalse_BcdrInbreastFilterTrain_NoAugment_ICNR_nnUPSAMPLE_Lam0.0_SpectralNorm_steps_${chkpt}_inference_eval"

