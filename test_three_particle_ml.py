import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, events, z_mean=None, z_std=None):
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

        z_values = np.array([ev["z_vertex"] for ev in events], dtype=np.float32)

        if z_mean is None:
            z_mean = z_values.mean()
        if z_std is None:
            z_std = z_values.std()

        self.z_mean = z_mean
        self.z_std = z_std

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        ev = self.events[idx]

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

        # Normalize target
        z = np.float32((ev["z_vertex"] - self.z_mean) / self.z_std)

        return torch.tensor(x, dtype=torch.float32), torch.tensor([z], dtype=torch.float32)

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

def train_model(train_events, val_events, n_epochs=20, batch_size=32, lr=1e-3):
    train_dataset = LambdaWSIDataset(train_events)

    val_dataset = LambdaWSIDataset(
        val_events,
        z_mean=train_dataset.z_mean,
        z_std=train_dataset.z_std,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleWSICNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for x, z_true in train_loader:
            x = x.to(device)
            z_true = z_true.to(device)

            optimizer.zero_grad()

            z_pred = model(x)
            loss = loss_fn(z_pred, z_true)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, z_true in val_loader:
                x = x.to(device)
                z_true = z_true.to(device)

                z_pred = model(x)
                loss = loss_fn(z_pred, z_true)

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1:03d} | train loss = {train_loss:.5f} | val loss = {val_loss:.5f}")

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

