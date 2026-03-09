CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train_scripts/debug_critic_values_all.py \
  --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_class_bin_11_hl_gauss_075/global_step_1100 \
  --dataset_path /data/shuozhe/saved_dataset/math-500/test-00000-of-00001_verl.parquet \
  --correct_match verl \
  --max_new_tokens 2048 \
  --out_dir /data/shuozhe/verl/critic_debug/05b_vh_class_bin_11_hl_gauss_075_step_1100 \
  # --start_index 3000 \
  # --end_index 3500 \
