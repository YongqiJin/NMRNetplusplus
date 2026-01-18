data_path="./data/nmrshiftdb2/All" # replace to your data path

#save_dir="./output/unlabel/nmrshiftdb2/5cv/pretrain_H_checkpoint_best_unimol_large_solv_atom_regloss_mae_lr_0.0001_bs1_4_bs2_16_wu_0.03_ep_20_wgt_1_T__ratio_4_16"  # replace to your finetune ckpt path
save_dir="$1"  # replace to your finetune ckpt path
element="$2"   # replace to your labeled atom
arch="$3"      # replace to your arch, e.g., unimol_large_solv
solvent_flags="$4" 
unlabeled_data_path="./data/ShiftDB-Lit/${element}" # replace to your unlabeled data path


mkdir -p ${save_dir}

nfolds=5
for fold in $(seq 0 $(($nfolds - 1)))
    do
    echo "Start infering..."
    cv_seed=42
    python ./uninmr/infer.py --user-dir  ./uninmr   ${data_path}   --valid-subset valid \
        --results-path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}  --saved-dir $save_dir \
        --num-workers 8 --ddp-backend=c10d --batch-size 64 \
        --task uninmr_solv --loss 'atom_regloss_mae' --arch "${arch}" \
        --dict-name "../../../weight/dict.txt" \
        ${solvent_flags} \
        --path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}/checkpoint_last.pt \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --selected-atom ${element}   --gaussian-kernel   --atom-descriptor 0 --split-mode infer 
    done 2>&1 | tee ${save_dir}/infer.log

python ./uninmr/utils/get_result.py --path ${save_dir} --file_end "*valid.out.pkl" --mode cv --dict "./weight/dict.txt" 2>&1 | tee ${save_dir}/result.log
python ./uninmr/utils/get_result_unlabel.py --path ${save_dir} --file_end "*valid.out.pkl" --mode cv --dict "./weight/dict.txt" 2>&1 | tee ${save_dir}/result_unlabel_1.log


nfolds=5
for fold in $(seq 0 $(($nfolds - 1)))
    do
    echo "Start infering..."
    cv_seed=42
    python ./uninmr/infer.py --user-dir  ./uninmr   ${unlabeled_data_path}   --valid-subset valid  --subset_name valid_unlabel \
        --results-path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}  --saved-dir $save_dir \
        --num-workers 8 --ddp-backend=c10d --batch-size 64 \
        --task uninmr_solv --loss 'atom_regloss_mae' --arch "${arch}" \
        --dict-name "../../../weight/dict.txt"\
        ${solvent_flags} \
        --path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}/checkpoint_last.pt \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --selected-atom ${element}   --gaussian-kernel   --atom-descriptor 0 --split-mode infer 

    done 2>&1 | tee ${save_dir}/infer_unlabel.log

python ./uninmr/utils/get_result_unlabel.py --path ${save_dir} --file_end "*valid_unlabel.out.pkl" --mode cv --dict "./weight/dict.txt" 2>&1 | tee ${save_dir}/result_unlabel_2.log 
