

dset_name=charades_vgg
ctx_mode=video_tef
train_path=data/charades_cd/charades_train_momentdiff.jsonl
eval_path=data/charades_cd/charades_test_ood_momentdiff.jsonl
eval_split_name=val
# v_feat_dirs=charades_features/rgb_features
v_feat_dim=4096
t_feat_dir=/mnt/workspace/workgroup/multimodal/moment_ret/charades/CD_glove_text_feature
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
--v_feat_dirs  /mnt/workspace/workgroup/multimodal/moment_ret/charades/rgb_features \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--eval_split_name val \
--seed 2018 \
--num_workers 12 \
--lr 0.0002 \
--lr_drop 40 \
--clip_length 0.16666 \
--lw_saliency 4.0 \
--enc_layers 3 \
--dec_layers 3 \
--span_loss_coef 1 \
--giou_loss_coef 8 \
--label_loss_coef 8 \
--set_cost_span 1 \
--set_cost_giou 8 \
--set_cost_class 8 \
--contrastive_align_loss_coef 0.02 \
--max_v_l -1
# ${@:1}