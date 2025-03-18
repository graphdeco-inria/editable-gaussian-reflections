(
    python train.py "${@}"
    if [ $? -ne 0 ]; then
        echo "Error detected, trying again..."
        python train.py "${@}"
    fi
    if [ $? -ne 0 ]; then
        echo "Error detected, trying again..."
        python train.py "${@}"
    fi
    if [ $? -ne 0 ]; then
        echo "Error detected, trying again..."
        python train.py "${@}"
    fi
) && (
    python render.py "${@}"
    if [ $? -ne 0 ]; then
        echo "Error detected, trying again..."
        python render.py "${@}"
    fi
    if [ $? -ne 0 ]; then
        echo "Error detected, trying again..."
        python render.py "${@}"
    fi
    if [ $? -ne 0 ]; then
        echo "Error detected, trying again..."
        python render.py "${@}"
    fi
)
