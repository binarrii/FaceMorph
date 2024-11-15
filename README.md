# FaceMorph

### Run

```bash

git clone https://github.com/binarrii/FaceMorph.git

cd FaceMorph
git submodule update --init --recursive

```

``` bash

# Install uv if needed : (https://docs.astral.sh/uv/getting-started/installation)

uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt

python main.py --workdir "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试" --refface female

```
