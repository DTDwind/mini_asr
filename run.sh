 . tools/env.sh || exit 1;

gpu="0" # CUDA_VISIBLE_DEVICES
infer_gpu="0"
train_config=conf/train_lstm.yaml
stage=1
stop_stage=1
train_set="an4_train_nodev"
dev_set=""
tag="asr_test"


if [ -z ${tag} ]; then
    expname=${train_set}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${tag}
fi

expdir=exp/${expname}


 if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1:  Training ASR Network"
    
    work=$expdir/.work
    mkdir -p $work

    export CUDA_VISIBLE_DEVICES=$gpu
    echo "log at: $work/train.log"
    echo "use GPU: $gpu"

    trainer.py \
        -c $train_config \
        data/${train_set} \
        data/${dev_set} \
        $expdir || exit 1
        # > $work/train.log
fi

echo "Finish !!!"