#! /bin/bash
start_time=$(date +%s)

mkdir -p results/mappings/

path_config_modules="configs/"

config_modules=(
    "config_mapzero.py:ConfigMapzero"
    "config_smartmap.py:ConfigSmartMap"
)

arch_dims=(
    "4x4"
    "8x8"
)

archs=(
    "OH_TOR_DIAG"
    "ONE_HOP"
    "MESH"
)

for arch_dim in "${arch_dims[@]}"
do
    for arch in "${archs[@]}"
    do
        for module in "${config_modules[@]}"
        do  
            IFS=':' read -r config_file config_class <<< "$module"
               
            ./scripts/train.sh $path_config_modules$config_file $config_class $arch $arch_dim

            ./scripts/map_with_zero_shot.sh $path_config_modules$config_file $config_class $arch $arch_dim
            
            ./scripts/map_with_finetune.sh $path_config_modules$config_file $config_class $arch $arch_dim
            
        done
    done
done

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo "Experiments time: $elapsed_time seconds."