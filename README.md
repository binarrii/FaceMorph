# FaceMorph

### Run

```bash

git clone https://github.com/binarrii/FaceMorph.git

cd FaceMorph
git submodule update --init --recursive

```

``` bash

# Install uv if needed: (See https://docs.astral.sh/uv/getting-started/installation)

uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt

python main.py --workdir="/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试" \
               --refface=female \
               --workers=2

```

```bash

# Notice: Ensure the following directory structure

/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试
├── female/
├── female_morphed_face/
├── female.png
├── female_predict_face/
├── female_skipped.txt
└── male/

```