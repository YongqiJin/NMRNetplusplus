export WANDB_MODE=offline

# Example training script for v2 solvent modes
# This example uses before-backbone BOS-only additive injection.

dataset='nmrshiftdb2_2018'
data_path=""  # replace with your labeled data path
unlabeled_data_path=""  # replace to your unlabeled data path
unlabeled_weight=16
bs1=4
bs2=16
ratio1=$bs1
ratio2=$bs2
batch_size=$((bs1 + bs2))

n_gpu=1
MASTER_PORT=33471
num_classes=1
weight_path='' # replace with your pretrained weight path
weight_name='checkpoint_best'
dict_name='dict'
lr=0.0004
epoch=10
dropout=0.0
warmup=0.03
update_freq=1
selected_atom='C'
loss='atom_regloss_mae'
arch='unimol_large_solv_v2'

GLOBAL_DISTANCE_FLAG=""
GAUSS_FLAG="--gaussian-kernel"
atom_des=0

# Solvent mode configuration (choose one scenario)
# Scenario A: before backbone BOS-only additive
SOLVENT_FLAGS="--solvent-embed-before-backbone True --bos-only --solvent-max-types 4"
# Scenario B: before backbone broadcast additive
# SOLVENT_FLAGS="--solvent-embed-before-backbone True --solvent-max-types 4"
# Scenario C: after backbone concat + linear
# SOLVENT_FLAGS="--solvent-embed-after-backbone True --solvent-embed-dim 16 --solv-concat  --solvent-max-types 4"
# Scenario D: after backbone additive
# SOLVENT_FLAGS="--solvent-embed-after-backbone True  --solvent-max-types 4"


global_batch_size=$((batch_size * n_gpu * update_freq))
global_bs1=$((bs1 * n_gpu * update_freq))
global_bs2=$((bs2 * n_gpu * update_freq))
timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="pretrain_${selected_atom}_${weight_name}_${arch}_${loss}_lr_${lr}_bs1_${global_bs1}_bs2_${global_bs2}_wu_${warmup}_ep_${epoch}_wgt_${unlabeled_weight}_${timestamp}"

save_dir="./output/unlabel/${dataset}/5cv/${exp_name}"
rm -rf "${save_dir}" 2>/dev/null || true
mkdir -p "${save_dir}"
echo "Folder created at: ${save_dir}"

nfolds=5
maxfolds=5
for fold in $(seq 0 $((maxfolds - 1))); do
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
        --finetune-from-model "${weight_path}/cv_seed_42_fold_${fold}/${weight_name}.pt" --dict-name "${dict_name}.txt" \
        --log-interval 1000 --log-format simple \
        --validate-interval 1 --keep-last-epochs 1 --save-interval 1 \
        --save-dir $fold_save_dir \
        --best-checkpoint-metric valid_rmse \
        --selected-atom $selected_atom  --split-mode cross_valid --nfolds $nfolds --fold $fold --cv-seed $cv_seed $GLOBAL_DISTANCE_FLAG $GAUSS_FLAG --atom-descriptor $atom_des

done 2>&1 | tee "${save_dir}/finetune.log"

# Add your inference script invocation if needed
sh ./script/infer_all_with_solv.sh ${save_dir} ${selected_atom} "${arch}" "${SOLVENT_FLAGS}"

