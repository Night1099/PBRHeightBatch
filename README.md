# PBRHeightBatch


## Prerequisites

Python <br>
Git

## Install

1. Install [Cuda Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive).

2. Open Terminal in folder you want to install into (Shift Rightclick in file explorer)<br>
Run

```
git clone https://github.com/Night1099/PBRHeightBatch
cd PBRHeightBatch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U requirements.txt
```

## Running

Example command
```
python height.py --input_dir input --output_dir output
```

extra args
```
parser.add_argument("--seed", type=int, default=random.randint(0, 100000), help="Seed for inference")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images")
parser.add_argument("--append", type=str, default='', help="String to append to the end of original file's basename")
parser.add_argument("--resolution", type=int, nargs=2, default=None, help="Custom resolution (width, height). If provided, no resizing will be done.")
parser.add_argument("--keep_res_resize", action="store_true", help="If set, the output image will be resized to the original input resolution.")
```
By default the output file will be same name as input but with --append you can add any string onto end of name

ie <br>
--append _height <br>
will turn panel.png into panel_height.png
