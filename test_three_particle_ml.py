import numpy as np
import subprocess
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from time import sleep
from time import perf_counter

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def get_gpu_temperature_c():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_temp = result.stdout.strip().splitlines()[0]
    return float(first_temp)


def wait_for_gpu_temperature(target_temp_c, sleep_seconds=15):
    if target_temp_c is None:
        return None

    temp_c = get_gpu_temperature_c()
    while temp_c is not None and temp_c > target_temp_c:
        print(
            f"GPU temp {temp_c:.0f}C exceeds target {target_temp_c:.0f}C; "
            f"pausing {sleep_seconds}s.",
            flush=True,
        )
        sleep(sleep_seconds)
        temp_c = get_gpu_temperature_c()

    return temp_c

import numpy as np
import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------- USER CONFIG ----------------

PHOTON1_FILE = "../photon1_test_50000.root"
PHOTON2_FILE = "../photon2_test_50000.root"
NEUTRON_FILE = "../neutron_test_50000.root"

TREE_NAME = "events"

# Geometry assumptions
ZDC_FACE_Z_MM = 35500.0
ZDC_HALF_WIDTH_X_MM = 300.0
ZDC_HALF_WIDTH_Y_MM = 300.0

# If files are still in the unrotated meson-structure frame, keep this true.
# If they are already in the ZDC/proton-beam frame, set ROTATE_TO_ZDC_FRAME = False.
ROTATE_TO_ZDC_FRAME = True
ROTATION_THETA_RAD = 0.025


def make_wsi_image_stack(hit_x, hit_y, hit_layer, hit_E,
                         n_layers=20,
                         nx=64,
                         ny=64,
                         x_min=-300.0,
                         x_max=300.0,
                         y_min=-300.0,
                         y_max=300.0):
    """
    Converts sparse WSi hits for one particle in one event into
    a dense image stack of shape [n_layers, nx, ny].
    """

    img = np.zeros((n_layers, nx, ny), dtype=np.float32)

    # Convert x,y positions to pixel indices
    ix = ((hit_x - x_min) / (x_max - x_min) * nx).astype(int)
    iy = ((hit_y - y_min) / (y_max - y_min) * ny).astype(int)

    # Keep only hits inside the image and valid layers
    mask = (
        (hit_layer >= 0) & (hit_layer < n_layers) &
        (ix >= 0) & (ix < nx) &
        (iy >= 0) & (iy < ny)
    )

    ix = ix[mask]
    iy = iy[mask]
    layers = hit_layer[mask]
    energies = hit_E[mask]

    # Accumulate energy into pixels
    np.add.at(img, (layers, ix, iy), energies)

    return img

class LambdaWSIDataset(Dataset):
    def __init__(self, events, z_mean=None, z_std=None, cache_images=False, progress_desc=None):
        """
        events should be a list of dictionaries.

        Each event should look like:

        {
            "gamma1": {"x": ..., "y": ..., "layer": ..., "E": ...},
            "gamma2": {"x": ..., "y": ..., "layer": ..., "E": ...},
            "neutron": {"x": ..., "y": ..., "layer": ..., "E": ...},
            "z_vertex": ...
        }
        """

        self.events = events
        self.cache_images = cache_images
        self._x_cache = None
        self._z_cache = None

        z_values = np.array([ev["z_vertex"] for ev in events], dtype=np.float32)

        if z_mean is None:
            z_mean = z_values.mean()
        if z_std is None:
            z_std = z_values.std()

        self.z_mean = z_mean
        self.z_std = z_std
        self._z_cache = ((z_values - self.z_mean) / self.z_std).astype(np.float32)

        if self.cache_images:
            iterator = range(len(self.events))
            if progress_desc and tqdm is not None:
                iterator = tqdm(iterator, desc=progress_desc, leave=False)

            self._x_cache = np.empty((len(self.events), 3, 20, 64, 64), dtype=np.float32)
            for i in iterator:
                self._x_cache[i] = self._encode_event(self.events[i])

    def __len__(self):
        return len(self.events)

    def _encode_event(self, ev):
        imgs = []

        for name in ["gamma1", "gamma2", "neutron"]:
            h = ev[name]

            img = make_wsi_image_stack(
                hit_x=np.asarray(h["x"]),
                hit_y=np.asarray(h["y"]),
                hit_layer=np.asarray(h["layer"], dtype=int),
                hit_E=np.asarray(h["E"]),
            )

            imgs.append(img)

        # Shape: [3, 20, 64, 64]
        x = np.stack(imgs, axis=0)

        # Optional but usually useful: compress dynamic range
        x = np.log1p(x)
        return x

    def __getitem__(self, idx):
        if self._x_cache is not None:
            z = self._z_cache[idx]
            return torch.from_numpy(self._x_cache[idx]), torch.tensor([z], dtype=torch.float32)

        ev = self.events[idx]
        x = self._encode_event(ev)

        # Normalize target
        z = self._z_cache[idx]

        return torch.from_numpy(x), torch.tensor([z], dtype=torch.float32)

class SimpleWSICNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(60, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: [batch, 3, 20, 64, 64]

        batch_size = x.shape[0]

        # Merge particle and layer dimensions:
        # [batch, 3, 20, 64, 64] -> [batch, 60, 64, 64]
        x = x.view(batch_size, 3 * 20, 64, 64)

        features = self.cnn(x)
        z_pred = self.regressor(features)

        return z_pred

def train_model(
    train_events,
    val_events,
    n_epochs=20,
    batch_size=32,
    lr=1e-3,
    show_progress=True,
    cache_images=True,
    num_workers=0,
    use_amp=True,
    target_gpu_temp_c=80,
    temp_check_interval=1,
    temp_cooldown_sleep=30,
):
    train_dataset = LambdaWSIDataset(
        train_events,
        cache_images=cache_images,
        progress_desc="Precomputing train images" if show_progress else None,
    )

    val_dataset = LambdaWSIDataset(
        val_events,
        z_mean=train_dataset.z_mean,
        z_std=train_dataset.z_std,
        cache_images=cache_images,
        progress_desc="Precomputing val images" if show_progress else None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = SimpleWSICNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    amp_enabled = use_amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"Training on {device}", flush=True)
    if show_progress and tqdm is None:
        print("tqdm is not installed; falling back to epoch-only logging.", flush=True)
    if target_gpu_temp_c is not None:
        current_temp_c = get_gpu_temperature_c()
        if current_temp_c is None:
            print("GPU temperature guard requested, but nvidia-smi temperature query failed.", flush=True)
        else:
            print(
                f"GPU temperature guard active: target {target_gpu_temp_c:.0f}C, "
                f"current {current_temp_c:.0f}C.",
                flush=True,
            )

    for epoch in range(n_epochs):
        epoch_start = perf_counter()
        model.train()
        train_loss = 0.0

        train_iter = train_loader
        if show_progress and tqdm is not None:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{n_epochs:03d} train", leave=False)

        for batch_idx, (x, z_true) in enumerate(train_iter):
            if batch_idx % max(1, temp_check_interval) == 0:
                wait_for_gpu_temperature(target_gpu_temp_c, temp_cooldown_sleep)

            x = x.to(device, non_blocking=pin_memory)
            z_true = z_true.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                z_pred = model(x)
                loss = loss_fn(z_pred, z_true)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)

            if show_progress and tqdm is not None:
                train_iter.set_postfix(loss=f"{loss.item():.5f}")

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_iter = val_loader
            if show_progress and tqdm is not None:
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{n_epochs:03d} val", leave=False)

            for batch_idx, (x, z_true) in enumerate(val_iter):
                if batch_idx % max(1, temp_check_interval) == 0:
                    wait_for_gpu_temperature(target_gpu_temp_c, temp_cooldown_sleep)

                x = x.to(device, non_blocking=pin_memory)
                z_true = z_true.to(device, non_blocking=pin_memory)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    z_pred = model(x)
                    loss = loss_fn(z_pred, z_true)

                val_loss += loss.item() * x.size(0)

                if show_progress and tqdm is not None:
                    val_iter.set_postfix(loss=f"{loss.item():.5f}")

        val_loss /= len(val_dataset)
        epoch_seconds = perf_counter() - epoch_start

        print(
            f"Epoch {epoch+1:03d} | train loss = {train_loss:.5f} | "
            f"val loss = {val_loss:.5f} | time = {epoch_seconds:.1f}s",
            flush=True,
        )

    return model, train_dataset.z_mean, train_dataset.z_std

def rotate_vectors_numpy(x, y, z, theta=0.025):
    c = np.cos(theta)
    s = np.sin(theta)

    x_rot =  c * x + s * z
    y_rot =  y
    z_rot = -s * x + c * z

    return x_rot, y_rot, z_rot


def first_per_event(arr):
    return np.array([x[1] if len(x) > 0 else np.nan for x in arr])


def load_first_mc_particle(filename, mass_assumption):
    tree = up.open(filename)[TREE_NAME]

    vx = first_per_event(tree["MCParticles.vertex.x"].array(library="np"))
    vy = first_per_event(tree["MCParticles.vertex.y"].array(library="np"))
    vz = first_per_event(tree["MCParticles.vertex.z"].array(library="np"))

    px = first_per_event(tree["MCParticles.momentum.x"].array(library="np"))
    py = first_per_event(tree["MCParticles.momentum.y"].array(library="np"))
    pz = first_per_event(tree["MCParticles.momentum.z"].array(library="np"))

    mass = np.full_like(px, mass_assumption, dtype=float)

    if ROTATE_TO_ZDC_FRAME:
        vx, vy, vz = rotate_vectors_numpy(vx, vy, vz, ROTATION_THETA_RAD)
        px, py, pz = rotate_vectors_numpy(px, py, pz, ROTATION_THETA_RAD)

    p = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p**2 + mass**2)

    return {
        "vx": vx, "vy": vy, "vz": vz,
        "px": px, "py": py, "pz": pz,
        "p": p,
        "E": E,
        "m": mass,
    }


def project_to_zdc(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = ZDC_FACE_Z_MM - p["vz"]
        x_zdc = p["vx"] + (p["px"] / p["pz"]) * dz
        y_zdc = p["vy"] + (p["py"] / p["pz"]) * dz
    return x_zdc, y_zdc


def intersects_zdc(x, y):
    return (
        np.isfinite(x) &
        np.isfinite(y) &
        (np.abs(x) < ZDC_HALF_WIDTH_X_MM) &
        (np.abs(y) < ZDC_HALF_WIDTH_Y_MM)
    )


def draw_zdc_box(ax):
    x0, x1 = -ZDC_HALF_WIDTH_X_MM, ZDC_HALF_WIDTH_X_MM
    y0, y1 = -ZDC_HALF_WIDTH_Y_MM, ZDC_HALF_WIDTH_Y_MM

    ax.plot([x0, x1], [y0, y0], "k--", linewidth=1)
    ax.plot([x1, x1], [y0, y1], "k--", linewidth=1)
    ax.plot([x1, x0], [y1, y1], "k--", linewidth=1)
    ax.plot([x0, x0], [y1, y0], "k--", linewidth=1)
