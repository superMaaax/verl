# Optional ablations:
#   --ablation reset_each_token
#   --ablation zero_weight_hh
#   --ablation no_direct_carry
#
# Optional range restriction:
#   --start_index 3000
#   --end_index 3500

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 eval_scripts/debug_critic_values_all.py \
#   --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5/global_step_1100 \
#   --dataset_path /data/shuozhe/saved_dataset/verl_math_7500_500_5000/test_5000.parquet \
#   --correct_match verl \
#   --max_new_tokens 2048 \
#   --out_dir /data/shuozhe/verl/critic_debug/05b_vh_init_e5_step_1100_level_4_format_match \
#   --levels "Level 4"


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 eval_scripts/debug_critic_values_all.py \
#   --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_gru/global_step_450 \
#   --dataset_path /data/shuozhe/saved_dataset/verl_math_7500_500_5000/test_5000.parquet \
#   --correct_match verl \
#   --max_new_tokens 2048 \
#   --levels "Level 1" \
#   --out_dir /data/shuozhe/verl/critic_debug/05b_vh_init_e5_gru_step_450_reset_each_token_l1 \
#   --ablation reset_each_token

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 eval_scripts/debug_critic_values_all.py \
#   --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_gru/global_step_450 \
#   --dataset_path /data/shuozhe/saved_dataset/verl_math_7500_500_5000/test_5000.parquet \
#   --correct_match verl \
#   --max_new_tokens 2048 \
#   --levels "Level 1" \
#   --out_dir /data/shuozhe/verl/critic_debug/05b_vh_init_e5_gru_step_450_zero_weight_hh_l1 \
#   --ablation zero_weight_hh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 eval_scripts/debug_critic_values_all.py \
  --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_gru/global_step_450 \
  --dataset_path /data/shuozhe/saved_dataset/verl_math_7500_500_5000/test_5000.parquet \
  --correct_match verl \
  --max_new_tokens 2048 \
  --levels "Level 1" \
  --out_dir /data/shuozhe/verl/critic_debug/05b_vh_init_e5_gru_step_450_no_direct_carry_l1 \
  --ablation no_direct_carry


  # --levels "Level 4" \
