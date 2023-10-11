

dset_name=charades_vgg
ctx_mode=video_tef
train_path=data/charades/charades_sta_train_tvr_format.jsonl
eval_path=data/charades/charades_sta_test_tvr_format.jsonl
eval_split_name=val
# v_feat_dirs=charades_features/rgb_features
v_feat_dim=4096
t_feat_dir=/mnt/workspace/workgroup/multimodal/moment_ret/charades/glove_text_feature
t_feat_dim=300
results_root=results
exp_id=exp
bsz=32


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python momentdiff/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs  /mnt/workspace/workgroup/multimodal/moment_ret/charades/rgb_features  \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--eval_split_name val \
--seed 42 \
--num_workers 12 \
--lr 0.0002 \
--lr_drop 40 \
--clip_length 0.16666 \
--lw_saliency 4.0 \
--max_v_l -1
