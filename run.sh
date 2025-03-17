
# 🤷
(
    python train.py "${@}" 
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python train.py "${@}" 
    fi
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python train.py "${@}" 
    fi 
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python train.py "${@}" 
    fi 
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python train.py "${@}" 
    fi 
) && (
    python render.py "${@}" 
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python render.py "${@}" 
    fi
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python render.py "${@}" 
    fi
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python render.py "${@}" 
    fi
    if [ $? -eq 139 ]; then
        echo "Segfault detected, trying again..."
        python render.py "${@}" 
    fi 
) 