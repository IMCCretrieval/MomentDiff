
dset_name=charades
ctx_mode=video_tef
train_path=data/charades/charades_sta_train_tvr_format.jsonl
eval_path=data/charades/charades_sta_test_tvr_format.jsonl
eval_split_name=val
v_feat_dim=2816
t_feat_dir=/mnt/workspace/workgroup/multimodal/moment_ret/charades_clip_features/clip_text_features
t_feat_dim=512
results_root=results
exp_id=exp
bsz=32


PYTHONPATH=$PYTHONPATH:. python momentdiff/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs  /mnt/workspace/workgroup/multimodal/moment_ret/charades_clip_features/clip_features /mnt/workspace/workgroup/multimodal/moment_ret/charades_clip_features/slowfast_features \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--eval_split_name val \
--seed 42 \
--lr 0.0002 \
--num_workers 12 \
--lr_drop 40 \
--clip_length 1 \
--lw_saliency 4.0 \
--giou_loss_coef 1.0 \
--max_v_l -1
