
q a6000 -n ours_sparse_baseline <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_sparse/${scene}_lut_sparseness_$k -r 768"

for scene in shiny_kitchen; do 
    for k in 1 2 4 8; do 
        q a6000 -n ours_sparse_$k <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_sparse/${scene}_sparseness_$k -r 768 --sparseness $k"
    done
done


q a6000 -n ours_sparse_lut_baseline <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_lut -s colmap/$scene -m output_sparse/${scene}_lut_sparseness_$k -r 768"

for scene in shiny_kitchen; do 
    for k in 1 2 4 8; do 
        q a6000 -n ours_sparse_lut_$k <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_lut -s colmap/$scene -m output_sparse/${scene}_lut_sparseness_$k -r 768 --sparseness $k"
    done
done

