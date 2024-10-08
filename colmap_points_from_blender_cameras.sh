
# Note to self: copy the blender color .pngs into something like colmap/easy_kitchen/train/images
# Then, copy the jsons to colmap/easy_kitchen/train/transforms_{train}.json

set -e

if [ -v SLURM_CLUSTER_NAME ]; then
    module load StdEnv/2020 intel/2020.1.217 cuda/11.0 colmap/3.6
fi

: ${PROJECT_PATH?Must point to something like colmap/my_scene}

for split in test train; do
    python blender_cameras_to_colmap_cameras.py $PROJECT_PATH/transforms_$split.json
    rm -rf $PROJECT_PATH/$split/fake_sparse
    rm -rf $PROJECT_PATH/$split/database.db
    rm -rf $PROJECT_PATH/$split/sparse
    rm -rf $PROJECT_PATH/$split/new_ids.txt
    mv $PROJECT_PATH/fake_sparse $PROJECT_PATH/$split

    colmap feature_extractor --database_path $PROJECT_PATH/$split/database.db --image_path $PROJECT_PATH/$split/images --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1

    colmap exhaustive_matcher --database_path $PROJECT_PATH/$split/database.db

    sqlite3 $PROJECT_PATH/$split/database.db "select image_id from images order by name" > $PROJECT_PATH/$split/new_ids.txt
    python colmap_replace_ids.py $PROJECT_PATH/$split/new_ids.txt $PROJECT_PATH/$split/fake_sparse/images.txt 

    colmap point_triangulator --database_path $PROJECT_PATH/$split/database.db --image_path $PROJECT_PATH/$split/images --input_path $PROJECT_PATH/$split/fake_sparse --output_path $(mkdir -p $PROJECT_PATH/$split/sparse/0; echo $_) 
done 
