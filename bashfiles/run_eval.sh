module load Anaconda3
source activate
conda activate uncertainr_eval

run_eval_checkpoints() {
    local tasks="${1:-gsm8k math_500 minerva olympiad amc23}"       
    local models="${2:-tiiuae/Falcon3-1B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct}"
    local data_list="${3:-gsm8k deepscaler symbolic}"                               
    local start=${4:-100}
    local end=${5:-1000}
    local step=${6:-100}
    local num_shots="${7:-0}"
    local model_seed="${8:-0}"
    
    # Counters
    local total_jobs=0
    local success_jobs=0
    local missing_models=0
    local missing_ckpts=0
    local failed_jobs=0
    local output_dir=""

    for model_full in $models; do
        local family=$(echo "$model_full" | cut -d'/' -f1)
        local model=$(echo "$model_full" | cut -d'/' -f2)

        for data in $data_list; do
            for task in $tasks; do
                for seed in 1025 2025 3025; do
                    local model_base="${output_dir}/${family}/${model}-${data}-GRPO-${num_shots}-seed${seed}-modeno_curr-uncertainmetricNone-T0.0-eta50.0-sensitivity2.0-beta0.5-d_min0.0-d_max100.0"

                    if [[ ! -d "$model_base" ]]; then
                        echo "⚠️  Model path does not exist: $model_base → skip this model/data."
                        ((missing_models++))
                        continue 3
                    fi

                    for (( ckpt=$start; ckpt<=$end; ckpt+=step )); do
                        local ckpt_path="${model_base}/checkpoint-${ckpt}"
                        ((total_jobs++))

                        if [[ ! -d "$ckpt_path" ]]; then
                            echo "⚠️  Missing checkpoint-$ckpt for model=$model_full data=$data task=$task seed=$seed → skip."
                            ((missing_ckpts++))
                            continue
                        fi

                        echo "▶️  Running model=$model_full data=$data task=$task seed=$seed checkpoint-$ckpt..."
                        if python run_eval.py \
                            --task="$task" \
                            --model_name="$ckpt_path" \
                            --model_seed=$model_seed \
                            --num_shots=$num_shots; then
                            ((success_jobs++))
                        else
                            echo "❌  Error task=$task model=$model_full data=$data seed=$seed ckpt=$ckpt → skip."
                            ((failed_jobs++))
                        fi
                    done
                done
            done
        done
    done

    echo ""
    echo "===== SUMMARY ====="
    echo "Total jobs attempted : $total_jobs"
    echo "✅ Successful jobs   : $success_jobs"
    echo "⚠️  Missing models   : $missing_models"
    echo "⚠️  Missing ckpts    : $missing_ckpts"
    echo "❌ Failed jobs       : $failed_jobs"
    echo "===================="
}

run_eval_checkpoints