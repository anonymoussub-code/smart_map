#! /bin/bash

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
        done
    done
done
