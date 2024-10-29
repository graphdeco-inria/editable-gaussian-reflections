```
python train.py -s colmap/chromeball_kitchen --split_spec_diff
```
Don't include /train at the end 
Assumes renders/chromeball_kitchen/train exists, which contains the transform jsons 
The version with split_spec_diff doesn't run anymore


Downsampling can be done by just changing the resolution in arguments. Remember that the slow bvh ran out of memory