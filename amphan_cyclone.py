# ================================================================
#  AMPHAN CYCLONE — Focused Wind (cyclone-centered, not global)
#  Google Colab | MP4 + GIF | 22 seconds
# ================================================================

# ── CELL 1: Run once then Runtime → Restart & Run All ───────────
# !pip install cartopy matplotlib numpy pillow --quiet
# !apt-get install -y ffmpeg libgeos-dev libproj-dev --quiet

# ── CELL 2: Main code ───────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
import warnings
warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.feature import NaturalEarthFeature
    HAS_CARTOPY = True
    print("✅ Cartopy available")
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Run Cell 1 first!")

# ════════════════════════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════════════════════════
N_PARTICLES     = 3500
TRAIL_LEN       = 35
STEPS_PER_FRAME = 6
FPS             = 24
TOTAL_SECONDS   = 22
DPI             = 160

LON_MIN, LON_MAX = 76.0, 97.0
LAT_MIN, LAT_MAX =  6.0, 28.0

# Radius beyond which particles fade to near-zero (degrees)
# Particles outside this ring from the eye are invisible
ACTIVE_RADIUS   = 9.0    # strong wind band
FADE_START      = 6.0    # alpha starts dropping here
SPAWN_RADIUS    = 8.5    # spawn particles within this ring

# ════════════════════════════════════════════════════════════════
#  TRACK  (lon, lat, label, intensity)
# ════════════════════════════════════════════════════════════════
RAW_TRACK = [
    (88.6, 11.5, "18 May 06:00", 0.30),
    (88.5, 12.5, "18 May 12:00", 0.42),
    (88.3, 13.8, "18 May 18:00", 0.54),
    (88.1, 15.0, "19 May 00:00", 0.64),
    (87.9, 16.2, "19 May 06:00", 0.74),
    (87.6, 17.4, "19 May 12:00", 0.83),
    (87.3, 18.5, "19 May 18:00", 0.89),
    (87.0, 19.6, "20 May 00:00", 0.94),
    (86.8, 20.5, "20 May 06:00", 0.98),
    (86.6, 21.4, "20 May 12:00", 1.00),
    (86.4, 22.1, "20 May 15:00", 1.00),
    (86.2, 22.7, "20 May 18:00", 0.97),
    (86.0, 23.2, "20 May 21:00", 0.90),
    (85.8, 23.8, "21 May 00:00", 0.75),
    (85.6, 24.3, "21 May 03:00", 0.58),
    (85.3, 24.9, "21 May 06:00", 0.40),
    (85.0, 25.4, "21 May 12:00", 0.25),
]
N_FRAMES = TOTAL_SECONDS * FPS   # 528 frames

def interp_track(t_norm):
    n = len(RAW_TRACK)
    idx_f = t_norm * (n - 1)
    i0 = int(np.clip(idx_f, 0, n - 2))
    frac = idx_f - i0
    i1 = i0 + 1
    lon   = RAW_TRACK[i0][0] + frac * (RAW_TRACK[i1][0] - RAW_TRACK[i0][0])
    lat   = RAW_TRACK[i0][1] + frac * (RAW_TRACK[i1][1] - RAW_TRACK[i0][1])
    label = RAW_TRACK[i0][2]
    inten = RAW_TRACK[i0][3] + frac * (RAW_TRACK[i1][3] - RAW_TRACK[i0][3])
    return lon, lat, label, inten

# ════════════════════════════════════════════════════════════════
#  WIND FIELD  — only strong near the eye
# ════════════════════════════════════════════════════════════════
def wind_uv(px, py, eye_lon, eye_lat, intensity):
    dx = px - eye_lon
    dy = py - eye_lat
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6
    Rmax = 2.0
    # Rankine vortex — zero far from eye
    core  = np.where(dist < Rmax, dist / Rmax, (Rmax / dist) ** 1.8)
    # Hard cut-off: no wind beyond ACTIVE_RADIUS
    cutoff = np.clip(1.0 - (dist - FADE_START) / (ACTIVE_RADIUS - FADE_START), 0, 1) ** 2
    Vmag  = core * intensity * 2.4 * cutoff
    angle = np.arctan2(dy, dx)
    u = -np.sin(angle) * Vmag   # CCW
    v =  np.cos(angle) * Vmag
    return u, v

# ════════════════════════════════════════════════════════════════
#  COLOR MAP  (matches screenshot: deep blue → purple → pink)
# ════════════════════════════════════════════════════════════════
WIND_CMAP = mcolors.LinearSegmentedColormap.from_list('wind', [
    (0.00, '#1a2870'),
    (0.30, '#2244cc'),
    (0.55, '#5522aa'),
    (0.75, '#cc2288'),
    (0.90, '#ff66bb'),
    (1.00, '#ffffff'),
], N=512)
SPEED_NORM = mcolors.Normalize(vmin=0.0, vmax=2.2)

# ════════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM — spawn inside SPAWN_RADIUS ring around eye
# ════════════════════════════════════════════════════════════════
class Particles:
    def __init__(self, n):
        self.n = n
        # Start clustered — will be overwritten in pre-warm
        self.x    = np.random.uniform(LON_MIN, LON_MAX, n)
        self.y    = np.random.uniform(LAT_MIN, LAT_MAX, n)
        self.age  = np.random.randint(0, TRAIL_LEN, n).astype(float)
        self.life = np.random.randint(TRAIL_LEN, TRAIL_LEN * 3, n).astype(float)
        self.spd  = np.zeros(n)
        self.hx   = np.full((TRAIL_LEN, n), np.nan)
        self.hy   = np.full((TRAIL_LEN, n), np.nan)
        self.dist_eye = np.ones(n) * 999

    def _spawn_around_eye(self, indices, eye_lon, eye_lat):
        """Spawn new particles in a ring around the current eye position."""
        nd = len(indices)
        if nd == 0:
            return
        # Random angle + random radius within spawn ring
        angles = np.random.uniform(0, 2 * np.pi, nd)
        radii  = np.random.uniform(0.5, SPAWN_RADIUS, nd)
        self.x[indices] = eye_lon + radii * np.cos(angles)
        self.y[indices] = eye_lat + radii * np.sin(angles)
        # Clip to map bounds
        self.x[indices] = np.clip(self.x[indices], LON_MIN + 0.1, LON_MAX - 0.1)
        self.y[indices] = np.clip(self.y[indices], LAT_MIN + 0.1, LAT_MAX - 0.1)
        self.age[indices]  = 0
        self.life[indices] = np.random.randint(TRAIL_LEN, TRAIL_LEN * 3, nd)
        self.hx[:, indices] = np.nan
        self.hy[:, indices] = np.nan

    def step(self, eye_lon, eye_lat, intensity):
        u, v = wind_uv(self.x, self.y, eye_lon, eye_lat, intensity)
        spd   = np.sqrt(u*u + v*v)
        self.spd = spd
        dt = 0.18 / (spd + 0.35)
        self.x += u * dt
        self.y += v * dt
        # Distance from eye
        dx = self.x - eye_lon
        dy = self.y - eye_lat
        self.dist_eye = np.sqrt(dx*dx + dy*dy)
        # Roll trail
        self.hx = np.roll(self.hx, 1, axis=0)
        self.hy = np.roll(self.hy, 1, axis=0)
        self.hx[0] = self.x
        self.hy[0] = self.y
        self.age += 1
        # Respawn if: too old, out of bounds, or drifted too far from eye
        dead = (
            (self.age  > self.life) |
            (self.x    < LON_MIN)   | (self.x > LON_MAX) |
            (self.y    < LAT_MIN)   | (self.y > LAT_MAX) |
            (self.dist_eye > ACTIVE_RADIUS + 1.0)
        )
        idx = np.where(dead)[0]
        self._spawn_around_eye(idx, eye_lon, eye_lat)

# ════════════════════════════════════════════════════════════════
#  FIGURE & MAP
# ════════════════════════════════════════════════════════════════
OCEAN_COL = '#0d1825'
LAND_COL  = '#1e2420'
COAST_COL = '#3a4a3a'
BORDER_COL= '#2e3e2e'
GRID_COL  = '#1a2820'

if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig  = plt.figure(figsize=(12, 9.5), facecolor=OCEAN_COL)
    ax   = fig.add_axes([0.0, 0.06, 1.0, 0.92], projection=proj)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    ax.set_facecolor(OCEAN_COL)
    land50 = NaturalEarthFeature('physical', 'land', '50m',
                                 facecolor=LAND_COL,
                                 edgecolor=COAST_COL,
                                 linewidth=0.6, zorder=1)
    ax.add_feature(land50)
    ax.add_feature(cfeature.BORDERS,
                   edgecolor=BORDER_COL, linewidth=0.5,
                   linestyle='--', zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'),
                   edgecolor='#162030', linewidth=0.35, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25,
                      color=GRID_COL, alpha=0.8, linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'color': '#445544', 'size': 7}
    gl.ylabel_style = {'color': '#445544', 'size': 7}
    TR = {'transform': proj}
else:
    fig, ax = plt.subplots(figsize=(12, 9.5), facecolor=OCEAN_COL)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0.06)
    ax.set_facecolor(OCEAN_COL)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    for s in ax.spines.values(): s.set_edgecolor(GRID_COL)
    ax.tick_params(colors='#445544', labelsize=7)
    TR = {}

# Particle overlay axes
if HAS_CARTOPY:
    ax2 = fig.add_axes(ax.get_position(), frameon=False, projection=proj)
    ax2.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
else:
    ax2 = fig.add_axes(ax.get_position(), frameon=False)
    ax2.set_xlim(LON_MIN, LON_MAX)
    ax2.set_ylim(LAT_MIN, LAT_MAX)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.patch.set_visible(False)

# ── Colorbar ─────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=WIND_CMAP, norm=SPEED_NORM)
sm.set_array([])
cax = fig.add_axes([0.94, 0.1, 0.02, 0.7]) # Changed position and dimensions for vertical
cb  = fig.colorbar(sm, cax=cax, orientation='vertical') # Changed orientation to vertical
cb.set_label('Wind Speed  (High → Low)', color='#555555', fontsize=7.5, rotation=270) # Added rotation for vertical label
cb.ax.yaxis.set_tick_params(labelsize=7, color='#555555') # Changed to yaxis and enabled labels
cb.outline.set_edgecolor('#333333')

# ── Static labels ────────────────────────────────────────────
CITIES = [
    ('Kolkata',         88.36, 22.57),
    ('Bhubaneswar',     85.83, 20.30),
    ('Dhaka',           90.41, 23.81),
    ('Visakhapatnam',   83.30, 17.70),
    ('Chennai',         80.27, 13.08),
    ('Port Blair',      92.75, 11.67),
]
REGIONS = [
    ('Bay of\nBengal',  88.5, 16.5, 13, '#3a5a7a'),
    ('India',           79.5, 20.5, 15, '#4a5a4a'),
    ('Bangladesh',      90.8, 24.3,  8, '#5a6a5a'),
]
for name, lon, lat, sz, col in REGIONS:
    ax.text(lon, lat, name, color=col, fontsize=sz,
            ha='center', va='center', alpha=0.45,
            style='italic', zorder=3, **TR)

for name, lon, lat in CITIES:
    ax.plot(lon, lat, 'o', color='#99aabb',
            markersize=3, zorder=8, **TR)
    ax.text(lon + 0.25, lat + 0.22, name,
            color='#99aabb', fontsize=7.5, zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5,
                                        foreground=OCEAN_COL)], **TR)

# ════════════════════════════════════════════════════════════════
#  ANIMATION STATE
# ════════════════════════════════════════════════════════════════
parts      = Particles(N_PARTICLES)
trail_cols = []
eye_arts   = []
title_obj  = [None]
time_obj   = [None]
track_obj  = [None]

def clear_artists(lst):
    for a in lst:
        try: a.remove()
        except: pass
    lst.clear()

# Pre-warm around the starting eye
print("Pre-warming particles around cyclone eye...")
el0, ea0, _, ei0 = interp_track(0.0)
# First place all particles near the eye
parts._spawn_around_eye(np.arange(N_PARTICLES), el0, ea0)
for _ in range(TRAIL_LEN * STEPS_PER_FRAME):
    parts.step(el0, ea0, ei0)
print("Ready. Starting render...\n")

# ════════════════════════════════════════════════════════════════
#  UPDATE FUNCTION
# ════════════════════════════════════════════════════════════════
def update(fi):
    t_norm = fi / max(N_FRAMES - 1, 1)
    eye_lon, eye_lat, label, intensity = interp_track(t_norm)

    for _ in range(STEPS_PER_FRAME):
        parts.step(eye_lon, eye_lat, intensity)

    # ── Trails ──────────────────────────────────────────────
    clear_artists(trail_cols)
    dist_from_eye = parts.dist_eye

    for seg in range(TRAIL_LEN - 1):
        x0 = parts.hx[seg + 1]
        y0 = parts.hy[seg + 1]
        x1 = parts.hx[seg]
        y1 = parts.hy[seg]
        ok = ~(np.isnan(x0) | np.isnan(y0) |
               np.isnan(x1) | np.isnan(y1))
        if not ok.any():
            continue

        # Trail fade (tail is transparent, head is bright)
        trail_fade = (1.0 - seg / TRAIL_LEN) ** 1.5

        # Distance-based alpha: particles far from eye are dim
        dist = dist_from_eye[ok]
        dist_alpha = np.clip(
            1.0 - (dist - FADE_START) / (ACTIVE_RADIUS - FADE_START),
            0.0, 1.0) ** 1.5

        alpha_arr = trail_fade * dist_alpha * 0.88

        spd  = parts.spd[ok]
        rgba = WIND_CMAP(SPEED_NORM(spd))
        rgba[:, 3] = alpha_arr

        segs = np.stack(
            [np.stack([x0[ok], x1[ok]], axis=1),
             np.stack([y0[ok], y1[ok]], axis=1)], axis=2)
        lc = LineCollection(segs, colors=rgba,
                            linewidths=0.9,
                            antialiaseds=True, zorder=5)
        if HAS_CARTOPY:
            lc.set_transform(proj)
        ax2.add_collection(lc)
        trail_cols.append(lc)

    # ── Eye wall ─────────────────────────────────────────────
    clear_artists(eye_arts)
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, alp, lw, dash in [
        (0.5, 0.70, 1.4, False),
        (1.2, 0.40, 1.1, True),
        (2.2, 0.22, 0.9, True),
        (3.5, 0.12, 0.7, True),
    ]:
        ln, = ax.plot(
            eye_lon + r * np.cos(theta),
            eye_lat + r * np.sin(theta),
            '--' if dash else '-',
            color='white', alpha=alp * intensity,
            linewidth=lw, zorder=7, **TR)
        eye_arts.append(ln)
    dot, = ax.plot(eye_lon, eye_lat, 'o',
                   color='white', markersize=6, zorder=9,
                   path_effects=[pe.withStroke(linewidth=7,
                                               foreground='#ee1166')],
                   **TR)
    eye_arts.append(dot)

    # ── Past track ───────────────────────────────────────────
    clear_artists([] if track_obj[0] is None else [track_obj[0]])
    n_past = max(2, int(t_norm * len(RAW_TRACK)))
    tl, = ax.plot(
        [RAW_TRACK[i][0] for i in range(n_past)],
        [RAW_TRACK[i][1] for i in range(n_past)],
        'o--', color='#ffcc44', linewidth=0.9,
        markersize=2.5, alpha=0.70, zorder=8, **TR)
    track_obj[0] = tl

    # ── Title & time ─────────────────────────────────────────
    if title_obj[0]: title_obj[0].remove()
    if time_obj[0]:  time_obj[0].remove()
    title_obj[0] = fig.text(
        0.50, 0.978, 'Amphan Cyclone',
        ha='center', va='top', color='white',
        fontsize=15, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=4, foreground=OCEAN_COL)])
    time_obj[0] = fig.text(
        0.50, 0.954, f'{label} IST',
        ha='center', va='top', color='#aaccff', fontsize=10,
        path_effects=[pe.withStroke(linewidth=3, foreground=OCEAN_COL)])

    # Attribution
    fig.text(0.01, 0.004,
             '© Synthetic ERA5-style wind  |  Bay of Bengal',
             color='#334433', fontsize=6.5)

    if (fi + 1) % FPS == 0 or fi == 0:
        pct = (fi + 1) / N_FRAMES * 100
        print(f"  Frame {fi+1:4d}/{N_FRAMES} [{pct:5.1f}%]  "
              f"eye=({eye_lon:.2f}°E, {eye_lat:.2f}°N)  I={intensity:.2f}")

# ════════════════════════════════════════════════════════════════
#  RENDER
# ════════════════════════════════════════════════════════════════
anim = FuncAnimation(fig, update, frames=N_FRAMES,
                     interval=1000 // FPS, blit=False)

# ── MP4 ──────────────────────────────────────────────────────
print(f"\nSaving amphan_cyclone.mp4  ({N_FRAMES} frames, {TOTAL_SECONDS}s @ {FPS}fps)...")
writer = FFMpegWriter(fps=FPS, bitrate=4000,
                      extra_args=['-vcodec', 'libx264',
                                  '-pix_fmt', 'yuv420p', '-crf', '17'])
anim.save('amphan_cyclone.mp4', writer=writer, dpi=DPI,
          savefig_kwargs={'facecolor': OCEAN_COL})
print("✅  amphan_cyclone.mp4 done!")

# ── GIF ──────────────────────────────────────────────────────
print(f"\nSaving amphan_cyclone.gif  (every 4th frame, DPI=90)...")
anim.save('amphan_cyclone.gif', writer='pillow',
          fps=FPS // 4, dpi=90,
          savefig_kwargs={'facecolor': OCEAN_COL})
print("✅  amphan_cyclone.gif done!")

# ── Display ──────────────────────────────────────────────────
from IPython.display import Video, Image, display
print("\n── MP4 preview ──")
display(Video('amphan_cyclone.mp4', embed=True, width=780))
print("\n── GIF preview ──")
display(Image('amphan_cyclone.gif', width=780))
print(f"\n📁 Both files in Files panel (left sidebar). Duration={TOTAL_SECONDS}s")
