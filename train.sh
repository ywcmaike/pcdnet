#!/usr/bin/env sh

now=$(date +"%Y%m%d_%H%M%S")

python train.py --name psgn_offset \
                --gpu_ids 2 \
                --data_root /home/weicai/dataset/ShapeNetCore.v2 \
                --vote_to_sub True \
                --use_z_weight False \
                --use_symm_edge_aggr False \
                --use_symm_edge_update False \
                --use_offset True \
                --form_batch False \
                --train_category plane \
                --eval_category plane \
                --num_per_model 4 \
                --batch_size 6 \
                --num_workers 3 \
                --lr 1e-4 \
                --weight_decay 5e-6 \
                --obj_file ./dataset/p2m/unit_ball.obj \
                --save_path saves/save_${now} \
                --log_path logs/log_${now} \
                --use_sample True \
                --use_orient_chamfer False \
                --use_new_laploss True \
                --point_loss_weight 1.0 \
                --edge_loss_weight 0.3 \ #0.3 \
                --norm_loss_weight 0.0 \ #1.6e-4 \
                --laplace_loss_weight 0.5 \ #0.0 \
                --move_loss_weight 0.1 \ #0.1 \
                --convex_loss_weight 0.0 \ #0.01 \
                --symm_loss_weight 0.0 \ #0.01 \
                --img_loss_weight 0.0 \ #0.0 \
                --simplify_loss_weight 1.0 \
                --hidden_channel 256 \
                --block_num 6 \
                --increase_level 0 \
                --global_level 0 \
                --use_diff_sub False \
                --use_pcdnet 0 \
                --pcdnet_adain 0 \
#                --decoder 'PC_ResGraphXDec' \ # 'PC_Dec', 'PC_ResDec', 'PC_GraphXDec', 'PC_ResGraphXDec'
