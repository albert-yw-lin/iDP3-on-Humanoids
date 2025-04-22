# bash scripts/vis_dataset.sh

dataset_path=data/cup

vis_cloud=1
# cd Improved-3D-Diffusion-Policy
python Improved-3D-Diffusion-Policy/vis_dataset.py --dataset_path $dataset_path \
                    --use_img 0 \
                    --vis_cloud ${vis_cloud} \
                    --use_pc_color 1 \
                    --downsample 0 \