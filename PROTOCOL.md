# Convert blender cameras
mkdir -p colmap/hypersim/train/images
cp /data/graphdeco/share/lamberts-world/extraction/hypersim_nerf_data/hypersim_nerf_data/ai_001_001/train/images/* colmap/hypersim/train/images
cp /data/graphdeco/share/lamberts-world/extraction/hypersim_nerf_data/hypersim_nerf_data/ai_001_001/train/transforms_train.json colmap/hypersim
PROJECT_PATH=colmap/hypersim bash colmap_points_from_blender_cameras.sh


# Run
bash run.sh 

# Run with split specular_diffuse
bash run_split.sh