
DISENTANGLEMENT=diffuse,glossy
GEOMETRY=normals,position
BRDF=roughness,base_color,specular,metalness

for scene in shiny_livingroom shiny_bedroom shiny_kitchen shiny_office; do
    for ablations in NONE DISENTANGLEMENT GEOMETRY BRDF DISENTANGLEMENT,GEOMETRY GEOMETRY,BRDF DISENTANGLEMENT,BRDF DISENTANGLEMENT,GEOMETRY,BRDF; do
        IFS=',' read -ra keys <<< "$ablations"
        ablated_passes=""
        for key in "${keys[@]}"; do
            if [[ "$var" != "NONE" ]]; then
                ablated_passes="$ablated_passes${!key},"
            fi
        done
        echo $ablations $ablated_passes
        q a6000 -t 1:00:00 -n ${scene}_ablate_$ablations <<< "ABLATION=$ablated_passes bash run.sh -s priors/real_datasets_v2_filmic/renders_priors/${scene}_256 -m output_network_ablations_v3/${scene}_ablate_${ablations//,/+} -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 ${@:3}"
    done
done

