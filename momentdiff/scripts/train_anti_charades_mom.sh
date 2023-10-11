

dset_name=charades_vgg
ctx_mode=video_tef
train_path=data/charades_mom/train_mom_80.jsonl 
eval_path=data/charades_mom/test_mom_80.jsonl 
eval_split_name=val
v_feat_dim=4096
t_feat_dir=/mnt/workspace/workgroup/multimodal/moment_ret/charades/mom_glove_text_feature
t_feat_dim=300
results_root=results
exp_id=exp
bsz=32


PYTHONPATH=$PYTHONPATH:. python momentdiff/train.py \
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
--lw_saliency 2.0 \
--max_v_l -1
# ${@:1}