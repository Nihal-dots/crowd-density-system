# Crowd Density System

Crowd counting and density estimation with [CSRNet](https://arxiv.org/abs/1802.10062)
(Congested Scene Recognition Network). The project covers model training, an inference
backend with alerting, and a Streamlit dashboard.

## Project structure

```
crowd-density-system/
├── model/               # CSRNet architecture
├── training/            # Training and evaluation scripts
├── backend/
│   ├── inference/       # Density map prediction and counting
│   ├── alerts/          # Threshold-based crowd alerts
│   ├── database/        # Alert storage (alerts.db)
│   └── video/           # Video / camera stream handling
├── app/
│   └── components/      # Streamlit UI components
├── experiments/         # Experiment configs and logs
├── data/
│   ├── raw/             # Original datasets (not tracked by git)
│   ├── processed/       # Generated density maps
│   └── samples/         # Small sample images / clips for demos
├── research/
│   └── results/         # Metrics, plots, comparison tables
├── notebooks/           # Jupyter / Colab notebooks
├── requirements.txt
└── test_env.py          # Environment check
```

## Setup on Windows (venv)

Requires Python 3.10 or newer. Run these in PowerShell from the project folder.

1. Create the virtual environment:

   ```powershell
   python -m venv venv
   ```

2. Activate it:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks the script, allow it for the current session and try again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   ```

   In Command Prompt, use `venv\Scripts\activate.bat` instead.

3. Upgrade pip and install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) GPU support.** `pip install torch` on Windows installs the CPU-only
   build. If you have an NVIDIA GPU, reinstall PyTorch with the command for your CUDA
   version from <https://pytorch.org/get-started/locally/>, for example:

   ```powershell
   pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

5. Verify the environment:

   ```powershell
   python test_env.py
   ```

   It prints the version of every library and whether CUDA is available.

## Setup on Google Colab

Colab already ships with PyTorch, torchvision, NumPy, SciPy, OpenCV, Pillow, Matplotlib,
h5py, pandas and tqdm, so only a few packages need installing.

1. Open a new notebook and enable the GPU:
   **Runtime → Change runtime type → Hardware accelerator → GPU**.

2. Get the project into the runtime. Either clone it:

   ```python
   !git clone <your-repo-url> crowd-density-system
   %cd crowd-density-system
   ```

   or mount Google Drive and work from a copy stored there (this also keeps checkpoints
   after the runtime disconnects):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/crowd-density-system
   ```

3. Install the dependencies:

   ```python
   !pip install -q -r requirements.txt
   ```

4. Verify the environment:

   ```python
   !python test_env.py
   ```

   `CUDA available : True` confirms the GPU runtime is active.

## Training

Run these from the project root, in order. On Colab, prefix each with `!` and mount
Google Drive first so checkpoints survive a disconnect.

```
python training/inspect_dataset.py     # validate the dataset
python training/density_map.py         # precompute density maps (about 2 minutes, 1.5 GB)
python training/train.py --part B --batch 8 --resume
```

`--resume` continues from `last.pth` if it exists and starts from scratch otherwise, so
the same command can be rerun after a disconnect. On Colab, checkpoints are written to
`MyDrive/crowd-density-system/checkpoints/part_<A|B>/`; elsewhere they go to
`experiments/checkpoints/`. Each checkpoint is about 190 MB and one is kept every 10
epochs (`--save-every`), so a 200-epoch run uses roughly 4 GB of Drive space.

The log and curves are written to `research/results/train_log_<part>.csv` and
`training_curves_<part>.png` after every epoch.

### Roboflow crowd dataset

The Roboflow export in `data/raw/crowd density tensorflow/` (640x640 images, one
head box per CSV row) is trained separately, fine-tuning from a ShanghaiTech checkpoint:

```
python training/prepare_roboflow.py    # box centres -> density maps in data/processed/roboflow/
python training/train.py --dataset roboflow --epochs 50 --resume --init-from <checkpoints>/part_A/best_model.pth
```

It validates on the `valid` split; checkpoints go to `checkpoints/roboflow/` and the log to
`research/results/train_log_roboflow.csv`.

## Data

Place datasets (e.g. ShanghaiTech Part A / Part B) under `data/raw/`. That folder,
model weights (`*.pth`) and `alerts.db` are ignored by git — keep checkpoints on Google
Drive or another store when training on Colab.
# crowd-density-system
