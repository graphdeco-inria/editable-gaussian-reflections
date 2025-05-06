
DISENTANGLEMENT=diffuse,glossy
GEOMETRY=normals,position
ROUGHNESS=roughness
F0=base_color,specular,metalness

for scene in shiny_livingroom shiny_bedroom shiny_kitchen shiny_office; do
    for ablations in NONE GEOMETRY ROUGHNESS F0 ROUGHNESS,F0 DISENTANGLEMENT,GEOMETRY GEOMETRY,ROUGHNESS,F0 DISENTANGLEMENT,ROUGHNESS,F0 DISENTANGLEMENT,GEOMETRY,ROUGHNESS,F0 DISENTANGLEMENT+GEOMETRY+ROUGHNESS+F0; do
        IFS=',' read -ra keys <<< "$ablations"
        ablated_passes=""
        for key in "${keys[@]}"; do
            if [[ "$var" != "NONE" ]]; then
                ablated_passes="$ablated_passes${!key},"
            fi
        done
        q a6000 -t 1:00:00 -n ${scene}_ablate_$ablations <<< "ABLATION=$ablated_passes bash run.sh -s priors/real_datasets_v1/renders_priors/${scene}_256 -m output_network_ablations_v2/${scene}_ablate_${ablations//,/+} -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 ${@:3}"
    done
done

