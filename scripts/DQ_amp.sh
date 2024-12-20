coco_path=/home/lance/workspace/AITOD-Dataset/aitod_small
python /home/lance/workspace/DQ-DETR/main_gpt_amp.py --amp \
  --output_dir logs/DQDETR_small_amp_150epoch -c /home/lance/workspace/DQ-DETR/config/DQ_5scale.py --coco_path $coco_path \
  --pretrain_model_path /home/lance/workspace/DQ-DETR/pretrained/pretrain_model.pth \
  --options dn_scalar=100 embed_init_tgt=False \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0