# Multi-View 3D Object Reconstruction and Visualization

A student project that reconstructs 3D objects from multiple overlapping photographs using **COLMAP** (Structure from Motion / Multi-View Stereo), **OpenCV** (image validation and optional feature-matching experiments), **Open3D** (optional mesh processing), and **Streamlit** (web interface).

## What this application does

- Accepts multiple JPG/JPEG/PNG images.
- Validates image files and reports dimensions and a basic blur score.
- Runs a CPU-friendly COLMAP sparse Structure-from-Motion (SfM) workflow by default.
- Exports the sparse model to `output/sparse.ply` and previews/downloads it in the Results tab (including common binary PLY files without Open3D).
- Optionally runs image undistortion, CUDA-dependent PatchMatch stereo, stereo fusion, Poisson meshing, and texture generation when dense reconstruction is enabled.
- Displays sparse/dense point clouds and meshes in the browser using Plotly.
- Provides an independent SIFT/ORB feature-matching experiment using OpenCV.
- Can create an additional Poisson mesh from the dense point cloud using Open3D.
- Exports PLY assets and can convert an Open3D mesh to GLB.

> COLMAP is a separate native application. Installing the Python requirements does **not** install COLMAP.

## Requirements

- Python 3.10–3.12 recommended for broad package compatibility.
- Core sparse workflow dependencies are in `requirements.txt`; Open3D and Trimesh are optional and listed in `requirements-optional.txt`.
- COLMAP installed separately and available on PATH, or its executable path entered in the app sidebar.
- 16 GB RAM is a useful target for student-scale projects. Sparse reconstruction can run without CUDA when SIFT extraction/matching are set to CPU mode. Dense PatchMatch stereo generally needs a CUDA-enabled COLMAP build and compatible GPU/driver setup.
- 20–50 sharp, overlapping images are a practical starting range for a small object, not a guarantee of success.

## Setup on Windows (PowerShell)

```powershell
cd 3D_Object_Reconstruction
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional: needed only for Open3D mesh generation/preview and GLB conversion
pip install -r requirements-optional.txt
```

Start the app:

```powershell
streamlit run app.py
```

In the sidebar, enter the path to your COLMAP batch file, for example:

```text
C:\COLMAP\COLMAP.bat
```

The actual path depends on where COLMAP was extracted/installed. Alternatively, add COLMAP to PATH and leave the default `colmap` value.

## Setup on Linux / macOS

```bash
cd 3D_Object_Reconstruction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional mesh / GLB functionality:
pip install -r requirements-optional.txt
streamlit run app.py
```

Make sure the `colmap` executable is available on PATH, or enter its full path in the sidebar.

## Check the environment

```bash
python scripts/check_environment.py
```

## How to use

1. Start Streamlit and open the local URL it prints (usually `http://localhost:8501`).
2. Enter a project name in the sidebar.
3. Open **Images**, upload your object photographs, and save valid images.
4. Open **Feature matching** to compare two images with SIFT or ORB (optional educational module).
5. Open **Reconstruction** and leave **Run dense reconstruction (CUDA required)** unchecked for the recommended CPU-friendly sparse workflow.
6. After the mapper completes, the app exports `output/sparse.ply`. Open **3D results** to preview and download the sparse point cloud.
7. Enable dense reconstruction only on a machine with a compatible CUDA-enabled COLMAP build/GPU. After dense stereo succeeds, the app can show the fused point cloud, mesh, and textured output.
8. Optionally create an Open3D Poisson mesh from the dense fused point cloud. A sparse SfM cloud is not a complete surface and is generally unsuitable for reliable meshing.
9. Download the PLY or GLB files. A textured COLMAP OBJ, when successfully generated, must be kept together with its MTL and texture image files.

## COLMAP stages used

The application invokes these commands in order:

1. `feature_extractor`
2. `exhaustive_matcher`
3. `mapper`
4. `image_undistorter` (dense mode)
5. `patch_match_stereo` (dense mode)
6. `stereo_fusion` (dense mode)
7. `poisson_mesher` (best-effort meshing)
8. `mesh_texturer` (best-effort texturing)

The command-line flags follow COLMAP's documented CLI. Use `colmap <command> -h` (or `COLMAP.bat <command> -h` on Windows) if your installed version differs.

## Image capture guidance

- Photograph a **stationary** object while moving the camera around it.
- Use a matte, textured, opaque object for the first experiment.
- Capture from multiple sides and at more than one height.
- Keep focus, zoom, and lighting consistent.
- Avoid motion blur, reflections, transparent materials, and featureless surfaces.
- Ensure neighboring images overlap substantially; each visible surface region should appear in multiple views.
- Keep the object still relative to its surroundings. Moving the object between shots can confuse standard SfM when the background is also reconstructed.

## Important technical notes

- The sparse PLY preview uses a lightweight built-in PLY reader; Open3D is optional for the sparse workflow. Open3D is used for Poisson mesh generation and mesh previews, while Trimesh is used for GLB conversion.
- The OpenCV SIFT/ORB tab is a **separate pairwise experiment**. COLMAP performs its own internal extraction and matching; this app does not inject custom OpenCV matches into COLMAP's database.
- The sparse point cloud is an initial estimate containing triangulated feature points and is not a complete surface. `output/sparse.ply` is exported immediately after the mapper stage. Dense reconstruction requires successful undistortion and stereo stages; PatchMatch stereo is CUDA-dependent in the standard COLMAP pipeline.
- Reconstruction scale is generally arbitrary unless a known physical distance or calibrated reference is supplied.
- Poisson meshing can create extra surfaces in unsupported regions. Inspect and trim generated geometry before using it as a measurement or 3D-printing asset.
- COLMAP texture output is a group of files (OBJ, MTL, image textures); keep them together.
- This starter project is a working pipeline scaffold, not a guarantee that every arbitrary photo set will reconstruct. Photogrammetry quality depends strongly on image coverage, texture, focus, camera motion, and hardware.

## Troubleshooting

### COLMAP is not found
Install COLMAP separately and enter its complete path in the sidebar, such as `C:\COLMAP\COLMAP.bat`, or add it to PATH.

### `COLMAP mapper did not produce a sparse model`
The app now reports database counts for images, keypoints, putative matches, and geometrically verified pairs. Use those values to locate the failure: zero keypoints usually points to image/feature extraction issues; zero putative matches points to insufficient overlap or weak visual texture; zero verified pairs means the tentative matches did not form consistent camera geometry. Three images are a very small set for object photogrammetry; try 20–50 sharp, overlapping photos from multiple heights. The sparse folder listing in the diagnostic also helps identify whether COLMAP wrote a model in an unexpected format/location.

### Sparse reconstruction finds very few registered images
Use more overlap, add photos around the object at different elevations, avoid blur, and choose a surface with distinct texture. Avoid repetitive patterns and changing the camera zoom.

### Dense PatchMatch stereo fails
Your COLMAP build may lack CUDA support, or the GPU driver/runtime may be incompatible. First run with **Run dense reconstruction (CUDA required)** disabled. The app retains and displays the sparse model even if an enabled dense stage fails. Use a compatible CUDA-enabled COLMAP build or another supported machine for MVS.

### Mesh is noisy or has holes
Improve source coverage and image focus. Try changing Poisson depth, inspect low-density regions, and remove outliers. Do not assume a visually complete mesh is metrically accurate.

### Textured model is missing
The app treats meshing and texturing as best-effort stages. Check the COLMAP command output and run `mesh_texturer -h` for your installed version. The dense point cloud can still be used if texturing fails.

## Project structure

```text
3D_Object_Reconstruction/
├── app.py
├── requirements.txt
├── requirements-optional.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── colmap_runner.py
│   ├── feature_matching.py
│   ├── image_utils.py
│   └── open3d_utils.py
├── scripts/
│   └── check_environment.py
└── projects/                 # generated at runtime
    └── <project>/output/sparse.ply  # automatically exported sparse cloud
```

## References

- COLMAP CLI: https://colmap.github.io/cli.html
- COLMAP tutorial: https://colmap.github.io/tutorial.html
- OpenCV feature matching: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
- Open3D documentation: https://www.open3d.org/docs/latest/
- Streamlit documentation: https://docs.streamlit.io/
