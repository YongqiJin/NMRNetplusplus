export WANDB_MODE=offline

# Scenario A: before backbone BOS-only additive
SOLVENT_FLAGS="--solvent-embed-before-backbone --bos-only --solvent-max-types 4"
# Scenario B: before backbone broadcast additive
# SOLVENT_FLAGS="--solvent-embed-before-backbone --solvent-max-types 4"
# Scenario C: after backbone concat + linear
# SOLVENT_FLAGS="--solvent-embed-after-backbone --solvent-embed-dim 16 --solv-concat  --solvent-max-types 4"
# Scenario D: after backbone additive
# SOLVENT_FLAGS="--solvent-embed-after-backbone --solvent-max-types 4"

dataset='nmrshiftdb2'
data_path="./data/nmrshiftdb2/All" # replace to your data path

unlabeled_data_path="./data/ShiftDB-Lit/H" # replace to your unlabeled data path
unlabeled_weight=16
bs1=4
bs2=16
ratio1=$bs1
ratio2=$bs2
batch_size=$((bs1 + bs2))

n_gpu=1  
MASTER_PORT=33371
num_classes=1
weight_path='./weight'  # replace to your pre-training ckpt path
weight_name='pretraining_molecular'  # replace to your pre-training ckpt name
lr=0.0004
epoch=10
dropout=0.0
warmup=0.03
update_freq=1

selected_atom='H'   # replace to your labeled atom
loss='atom_regloss_mae'
arch='unimol_large_solv'

GLOBAL_DISTANCE_FLAG=""
GAUSS_FLAG="--gaussian-kernel"
atom_des=0

global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
global_bs1=`expr $bs1 \* $n_gpu \* $update_freq`
global_bs2=`expr $bs2 \* $n_gpu \* $update_freq`
timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="scratch_${selected_atom}_${weight_name}_${arch}_${loss}_lr_${lr}_bs1_${global_bs1}_bs2_${global_bs2}_wu_${warmup}_ep_${epoch}_wgt_${unlabeled_weight}_${timestamp}"

save_dir="./output/unlabel/${dataset}/5cv/${exp_name}"
if [ -d "${save_dir}" ]; then
    rm -rf ${save_dir}
    echo "Folder remove at: ${save_dir}"
fi
mkdir -p ${save_dir}
echo "Folder created at: ${save_dir}"


nfolds=5
maxfolds=5
for fold in $(seq 0 $(($maxfolds - 1)))
    do
    export NCCL_ASYNC_ERROR_HANDLING=1
    export OMP_NUM_THREADS=1
    cv_seed=42
    fold_save_dir="${save_dir}/cv_seed_${cv_seed}_fold_${fold}"
    torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
        --unlabeled-data $unlabeled_data_path --unlabeled-weight $unlabeled_weight --ratios $ratio1 $ratio2 \
        --user-dir ./uninmr  --train-subset train --valid-subset valid \
        --num-workers 8 --ddp-backend=c10d \
        --tensorboard-logdir "${fold_save_dir}/tensorboard" --wandb-project "NMRNet" --wandb-name "${exp_name}_fold_${fold}" \
        --task uninmr_solv --loss $loss --arch $arch \
        $SOLVENT_FLAGS \
        --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
        --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
        --update-freq $update_freq --seed 1 \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --num-classes $num_classes --pooler-dropout $dropout \
        --finetune-from-model "${weight_path}/${weight_name}.pt" --dict-name "../../../weight/dict.txt" \
        --log-interval 1000 --log-format simple \
        --validate-interval 1 --keep-last-epochs 1 --save-interval 1 \
        --save-dir $fold_save_dir \
        --best-checkpoint-metric valid_rmse \
        --selected-atom $selected_atom  --split-mode cross_valid --nfolds $nfolds --fold $fold --cv-seed $cv_seed $GLOBAL_DISTANCE_FLAG $GAUSS_FLAG --atom-descriptor $atom_des
done 2>&1 | tee "${save_dir}/finetune.log"


sh script/infer_with_solv.sh ${save_dir} ${selected_atom} ${arch} ${SOLVENT_FLAGS}

