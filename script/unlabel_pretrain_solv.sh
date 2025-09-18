export WANDB_MODE=offline

dataset='nmrshiftdb2_2018'
data_path="./data/nmrshiftdb2_2018/All" # replace to your data path

unlabeled_data_path="./data/NMRexp_v0905/C" # replace to your unlabeled data path
#unlabeled_data_path="/mnt/fs_mol/yongqi/clean-code/forward/unlabeled_data/v0508_H_max_chiral_1_max_atoms_100_warn_True_ele_C_H_O_N_S_P_F_Cl_filtered" # replace to your unlabeled data path
unlabeled_weight=1
bs1=4
bs2=16
ratio1=$bs1
ratio2=$bs2
batch_size=$((bs1 + bs2))  # replace 8 or 16

n_gpu=1  
MASTER_PORT=33371
num_classes=1
weight_path='./weight/C_pretraining_molecular_unimol_large_atom_regloss_mae_lr_0.0001_bs_8_0.03_50'  # replace to your pre-training ckpt path
weight_name='checkpoint_best'  # replace to your pre-training ckpt name
dict_name='dict'
lr=0.0001 
epoch=20   # replace 40, 45, 50, 60, 200 
dropout=0.0
warmup=0.03
update_freq=1

selected_atom='C'   # replace to your labeled atom
loss='atom_regloss_mae'
#arch='unimol_large'
arch='unimol_large_solv'

GLOBAL_DISTANCE_FLAG=""

GAUSS_FLAG="--gaussian-kernel"


atom_des=0

global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
global_bs1=`expr $bs1 \* $n_gpu \* $update_freq`
global_bs2=`expr $bs2 \* $n_gpu \* $update_freq`
timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="pretrain_${selected_atom}_${weight_name}_${arch}_${loss}_lr_${lr}_bs1_${global_bs1}_bs2_${global_bs2}_wu_${warmup}_ep_${epoch}_wgt_${unlabeled_weight}_T_${T}_ratio_${ratio1}_${ratio2}_${timestamp}"
#

save_dir="./output/unlabel/${dataset}/5cv/${exp_name}"
if [ -d "${save_dir}" ]; then
    rm -rf ${save_dir}
    echo "Folder remove at: ${save_dir}"
fi
mkdir -p ${save_dir}
echo "Folder created at: ${save_dir}"


nfolds=5
maxfolds=5  # 1 or 5
#python /mnt/fs_mol/wyc/NMRNet++/modify_checkpoint.py
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
        --task uninmr_solv --loss $loss --arch $arch  \
        --solvent-embed-dim 16  --solvent-embed-after-backbone True --solvent-embed-before-backbone False \
        --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
        --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
        --update-freq $update_freq --seed 1 \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --num-classes $num_classes --pooler-dropout $dropout \
        --finetune-from-model "${weight_path}/cv_seed_42_fold_${fold}/${weight_name}.pt" --dict-name "${dict_name}.txt" \
        --log-interval 1000 --log-format simple \
        --validate-interval 1 --keep-last-epochs 1 --save-interval 1\
        --save-dir $fold_save_dir \
        --best-checkpoint-metric valid_rmse \
        --selected-atom $selected_atom  --split-mode cross_valid --nfolds $nfolds --fold $fold --cv-seed $cv_seed $GLOBAL_DISTANCE_FLAG $GAUSS_FLAG --atom-descriptor $atom_des
    done 2>&1 | tee "${save_dir}/finetune.log"


sh infer_all.sh ${save_dir} ${selected_atom}
#find ${save_dir}/cv_seed_${cv_seed}_fold_${fold} -type f -name "*.pt" -exec rm -f {} \;
