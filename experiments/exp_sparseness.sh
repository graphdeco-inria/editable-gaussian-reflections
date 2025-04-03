
q a6000 -n sparse_baseline <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v45 -s colmap/$scene -m output_sparse/${scene}_baseline -r 768"

for scene in shiny_kitchen; do 
    for k in 1 2 4 8; do 
        q a6000 -n sparse_$k <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v45 -s colmap/$scene -m output_sparse/${scene}_sparseness_$k -r 768 --sparseness $k"
    done
done
