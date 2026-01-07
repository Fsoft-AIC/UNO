export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3,4,5
TAG_NAME=test
CKPT_PATH=${TAG_NAME}/checkpoint_1.pth
python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port 35502 \
                    --nproc_per_node=4 \
                    --use_env \
                    main.py \
                    --dino_layer 20 \
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
                    --epochs 300000 \
                    --lr 1e-3 \
                    --batch_size 704 \
                    --dsgg_task sgdet ;\
