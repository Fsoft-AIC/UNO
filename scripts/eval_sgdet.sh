export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=5
TAG_NAME=test
EPOCH=20
CKPT_PATH=${TAG_NAME}/checkpoint_${EPOCH}.pth
python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port 32502 \
                    --nproc_per_node=1 \
                    --use_env \
                    main.py \
                    --eval \
                    --pretrained ${CKPT_PATH} \
                    --dino_layer 22 \
                    --num_object_slots 40 \
                    --num_relation_slots 24 \
                    --set_cost_obj_class 2 \
                    --obj_loss_coef 2 \
                    --set_cost_rel 1 \
                    --rel_loss_coef 10 \
                    --use_pointer_matching \
                    --set_cost_pointer 20 \
                    --pointer_loss_coef 1 \
                    --output_dir ${TAG_NAME}/ \
                    --batch_size 192 \
                    --dsgg_task sgdet ;\