dset_name=activity
ctx_mode=video_tef
train_path=data/anet_CD/anet_train_momentdiff.jsonl
eval_path=data/anet_CD/anet_test_ood_momentdiff.jsonl
eval_split_name=val
v_feat_dim=500
t_feat_dir=/mnt/workspace/workgroup/multimodal/moment_ret/activitynet/CD_glove_text_fea/
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
--v_feat_dirs /mnt/workspace/workgroup/multimodal/moment_ret/activitynet/c3d_npz \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--eval_split_name val \
--seed 2018 \
--lr 0.0002 \
--num_workers 12 \
--lr_drop 20 \
--span_loss_coef 1 \
--giou_loss_coef 8 \
--label_loss_coef 8 \
--set_cost_span 1 \
--set_cost_giou 8 \
--set_cost_class 8 \
--enc_layers 2 \
--dec_layers 2 \
--lw_saliency 4.0 \
--max_v_l -1


