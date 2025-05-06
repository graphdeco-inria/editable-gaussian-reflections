


for scene in shiny_livingroom; do 
    for k in 1 4 8 20; do 
        q a6000 -n sparse_$k <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 -s renders/$scene -m output_sparse/${scene}_sparseness_$k -r 768 --sparseness $k"
    done
done
