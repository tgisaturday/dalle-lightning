#from https://github.com/robvanvolt/DALLE-datasets/blob/main/utilities/wds_create_lagacy.py
import webdataset as wds
import os
from pathlib import Path
from collections import Counter
from PIL import Image
import argparse, sys, random, glob


parser = argparse.ArgumentParser("""Generate sharded dataset from image-text-datasets.""")
parser.add_argument(
    "--image_text_keys", 
    type=str, 
    default="img,cap",
    help="Comma separated WebDataset dictionary keys for images (first argument) and texts (second argument). \
          The exact argument has to be provided to train_dalle.py, e.g. python train_dalle.py --wds img,cp --image_text_folder ../shards"
    )
parser.add_argument(
    "--compression", 
    dest="compression", 
    action="store_true",
    help="Creates compressed .tar.gz files instead of uncompressed .tar files."
    )    
parser.add_argument(
    "--output", 
    default="./dataset", 
    help="directory where shards are written"
)
parser.add_argument(
    "--shard_prefix", 
    default="ds_", 
    help="prefix of shards' filenames created in the shards-folder"
)
parser.add_argument(
    "--data",
    default="./data",
    help="directory path containing data suitable for DALLE-pytorch training",
)
args = parser.parse_args()

image_key, caption_key = tuple(args.image_text_keys.split(','))

if not os.path.isdir(os.path.join(args.data)):
    print(f"{args.data}: should be directory containing image-text pairs", file=sys.stderr)
    print(f"or subfolders containing image-text-pairs", file=sys.stderr)
    sys.exit(1)

os.makedirs(Path(args.output), exist_ok=True)

def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()

path = Path(args.data)
text_files = [*path.glob('**/*.txt')]
text_files = {text_file.stem: text_file for text_file in text_files} # str(text_file.parents[0]) + 
text_total = len(text_files)

image_files = [
    *path.glob('**/*.png'), *path.glob('**/*.jpg'),
    *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
]
image_files = {image_file.stem: image_file for image_file in image_files} # str(image_file.parents[0]) +
image_total = len(image_files)

print('Found {:,} textfiles and {:,} images.'.format(text_total, image_total))

keys = (image_files.keys() & text_files.keys())

text_files = {k: v for k, v in text_files.items() if k in keys}
image_files = {k: v for k, v in image_files.items() if k in keys}

for key in image_files:
    img = Image.open(image_files[key])
    try:
        img.verify()
    except Exception:
        print('Invalid image on path {}'.format(key))
        keys.remove(key)

print("Remaining keys after image sanity check: {:,}".format(len(keys)))

total_pairs = len(keys)
keys = list(keys)

indexes = list(range(total_pairs))
random.shuffle(indexes)

### (3) Create compressed Webdataset tar file
if args.compression:
    sink = wds.TarWriter(f'{args.output}/ds.tar.gz', encoder=False)    
else:
    sink = wds.TarWriter(f'{args.output}/ds.tar', encoder=False)

for i in indexes:
    with open(image_files[keys[i]], "rb") as imgstream:
        image = imgstream.read()
    with open(text_files[keys[i]], "rb") as txtstream:
        text = txtstream.read()
    ds_key = "%09d" % i
    sample = {
        "__key__": ds_key,
        "img": image,
        "cap": text
    }
    sink.write(sample)
sink.close()
