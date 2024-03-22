### copy the raw data
### add "aug" to the tile 
### save to data/augmented/Train

import os
import shutil
import re

source_dir = "./data/raw/Train"
destination_dir = "./data/raw/Train"
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for _,dirs,_ in os.walk(source_dir):
    for dir in dirs:
        tile_number = re.findall(r'\d+',dir)[0]
        new_number = int(tile_number) + 60
        new_tile = re.sub(r'\d+', str(new_number),dir)
        destination_root = os.path.join(destination_dir, new_tile)
        destination_root

        if not os.path.exists(destination_root):
            os.makedirs(destination_root)

        for root,_,files in os.walk(os.path.join(source_dir,dir)):
            for index, file_name in enumerate(files):
                source_path = os.path.join(root,file_name)
                destination_path = os.path.join(destination_root, file_name)

                shutil.copy2(source_path,destination_path)

