python train.py "${@}" && python render.py --train_views --modes regular "${@}" && python render.py "${@}" 
