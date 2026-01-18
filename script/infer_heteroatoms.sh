data_path="./data/nmrshiftdb2_2024/All" # replace to your data path

save_dir="$1"  # replace to your finetune ckpt path
element="$2"  # replace to your element

data_path_new="./data/ShiftDB-Lit/${element}" # replace to your unlabeled data path


mkdir -p ${save_dir}

nfolds=5
for fold in $(seq 0 $(($nfolds - 1)))
    do
    echo "Start infering..."
    cv_seed=42
    python ./uninmr/infer.py --user-dir  ./uninmr   ${data_path}   --valid-subset valid \
        --results-path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}  --saved-dir $save_dir \
        --num-workers 8 --ddp-backend=c10d --batch-size 64 \
        --task uninmr --loss 'atom_regloss_mae' --arch 'unimol_large' \
        --dict-name "../../../weight/dict.txt" \
        --path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}/checkpoint_last.pt \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --selected-atom ${element}   --gaussian-kernel   --atom-descriptor 0 --split-mode infer 
    done 2>&1 | tee ${save_dir}/infer.log

python ./uninmr/utils/get_result.py --path ${save_dir} --file_end "*valid.out.pkl" --mode cv --dict "./weight/dict.txt" 2>&1 | tee ${save_dir}/result.log


nfolds=5
for fold in $(seq 0 $(($nfolds - 1)))
    do
    echo "Start infering..."
    cv_seed=42
    python ./uninmr/infer.py --user-dir  ./uninmr   ${data_path_new}   --valid-subset valid \
        --results-path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}  --saved-dir $save_dir \
        --num-workers 8 --ddp-backend=c10d --batch-size 64 \
        --task uninmr --loss 'atom_regloss_mae' --arch 'unimol_large' \
        --dict-name "../../../weight/dict.txt" \
        --path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}/checkpoint_last.pt \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --selected-atom ${element}   --gaussian-kernel   --atom-descriptor 0 --split-mode infer 
    done 2>&1 | tee ${save_dir}/infer_new.log

python ./uninmr/utils/get_result.py --path ${save_dir} --file_end "*valid.out.pkl" --mode cv --dict "./weight/dict.txt" 2>&1 | tee ${save_dir}/result_new.log

