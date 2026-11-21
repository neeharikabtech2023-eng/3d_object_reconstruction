from __future__ import annotations

import os
import re
import io
import zipfile
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st

from modules.image_utils import inspect_image_bytes, save_uploaded_images
from modules.feature_matching import create_feature_match_visualization
from modules.colmap_runner import (
    find_sparse_model,
    run_colmap_pipeline,
    resolve_colmap_executable,
)
from modules.open3d_utils import (
    get_point_cloud_stats,
    create_poisson_mesh,
    load_point_cloud_for_plot,
    load_mesh_for_plot,
    export_mesh_to_glb,
)

ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Multi-View 3D Reconstruction",
    page_icon="🧊",
    layout="wide",
)


def safe_project_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_-")
    return cleaned[:60] or "reconstruction_project"


def project_paths(project_name: str) -> dict[str, Path]:
    project_dir = PROJECTS_DIR / safe_project_name(project_name)
    return {
        "root": project_dir,
        "images": project_dir / "images",
        "sparse": project_dir / "sparse",
        "dense": project_dir / "dense",
        "output": project_dir / "output",
        "database": project_dir / "database.db",
    }


def discover_assets(paths: dict[str, Path]) -> dict[str, Path | None]:
    output = paths["output"]
    dense = paths["dense"]
    sparse_point_cloud = next(
        (p for p in (output / "sparse.ply", paths["sparse"] / "sparse.ply") if p.is_file()),
        None,
    )
    sparse_model = find_sparse_model(paths["sparse"])
    candidates = [
        dense / "fused.ply",
        output / "fused.ply",
        dense / "meshed-poisson.ply",
        output / "meshed-poisson.ply",
        output / "open3d_mesh.ply",
    ]
    point_cloud = next((p for p in candidates[:2] if p.is_file()), None)
    mesh = next((p for p in candidates[2:] if p.is_file()), None)

    textured_obj = None
    for base in (dense / "textured", output / "textured"):
        possible = [base, Path(str(base) + ".obj"), base / "mesh.obj"]
        for candidate in possible:
            if candidate.is_file() and candidate.suffix.lower() == ".obj":
                textured_obj = candidate
                break
        if textured_obj:
            break
        if base.is_dir():
            obj_files = list(base.glob("*.obj"))
            if obj_files:
                textured_obj = obj_files[0]
                break
    if textured_obj is None:
        for candidate in list(dense.glob("*.obj")) + list(output.glob("*.obj")):
            if "textur" in candidate.name.lower():
                textured_obj = candidate
                break

    return {
        "sparse_model": sparse_model,
        "sparse_point_cloud": sparse_point_cloud,
        "point_cloud": point_cloud,
        "mesh": mesh,
        "textured_obj": textured_obj,
    }


def render_asset_preview(asset: Path, kind: str, max_points: int = 50000) -> None:
    try:
        import plotly.graph_objects as go

        if kind == "point_cloud":
            data = load_point_cloud_for_plot(asset, max_points=max_points)
            if data is None:
                st.warning("Could not read this point cloud. Check that the PLY file is valid and uses a supported format.")
                return
            x, y, z, colors = data
            marker: dict[str, Any] = {"size": 2, "opacity": 0.85}
            if colors is not None:
                marker["color"] = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in colors]
            else:
                marker["color"] = z
                marker["colorscale"] = "Viridis"
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker)])
            preview_title = "Sparse point cloud" if asset.name.lower() == "sparse.ply" else "Dense point cloud"
            fig.update_layout(title=preview_title, scene_aspectmode="data", height=620, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            data = load_mesh_for_plot(asset, max_faces=100000)
            if data is None:
                st.warning("Could not read this mesh. Check that Open3D is installed and the mesh file is valid.")
                return
            vertices, faces, vertex_colors = data
            mesh_args: dict[str, Any] = {
                "x": vertices[:, 0], "y": vertices[:, 1], "z": vertices[:, 2],
                "i": faces[:, 0], "j": faces[:, 1], "k": faces[:, 2],
                "opacity": 1.0,
                "flatshading": False,
                "name": "Reconstructed mesh",
            }
            if vertex_colors is not None:
                mesh_args["vertexcolor"] = [
                    f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in vertex_colors
                ]
                mesh_args["color"] = "lightblue"
            else:
                mesh_args["color"] = "lightblue"
            fig = go.Figure(data=[go.Mesh3d(**mesh_args)])
            fig.update_layout(title="3D mesh", scene_aspectmode="data", height=620, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Preview error: {exc}")


# Sidebar configuration
st.sidebar.title("Project settings")
project_name_input = st.sidebar.text_input("Project name", value="my_first_reconstruction")
project_name = safe_project_name(project_name_input)
paths = project_paths(project_name)
paths["root"].mkdir(parents=True, exist_ok=True)
paths["images"].mkdir(parents=True, exist_ok=True)
paths["output"].mkdir(parents=True, exist_ok=True)

colmap_default = os.environ.get("COLMAP_EXE", "colmap")
colmap_input = st.sidebar.text_input(
    "COLMAP executable or full path",
    value=colmap_default,
    help="Examples: colmap (if on PATH), C:/COLMAP/COLMAP.bat, or /usr/bin/colmap",
)
run_dense = st.sidebar.checkbox(
    "Run dense reconstruction (CUDA required)",
    value=False,
    help="Recommended: leave off for the CPU-friendly sparse workflow. Dense PatchMatch stereo needs a compatible CUDA-enabled COLMAP build and GPU.",
)
use_gpu_for_sift = st.sidebar.checkbox("Use GPU for SIFT extraction/matching", value=False)
max_image_size = st.sidebar.selectbox("Maximum image size for dense stage", [1200, 1600, 2000, 2400, 3000], index=2)

st.title("🧊 Multi-View 3D Object Reconstruction")
st.caption("Build a CPU-friendly sparse 3D reconstruction first. Dense point clouds, meshes, and texturing are optional CUDA-dependent stages.")

st.info(
    "Capture a stationary object from many overlapping viewpoints. Start with a textured, matte object such as a toy, shoe, or ceramic mug. "
    "For a first attempt, use 25–50 sharp photos with consistent lighting."
)

tab_upload, tab_reconstruct, tab_features, tab_results, tab_help = st.tabs(
    ["1 · Images", "2 · Reconstruction", "3 · Feature matching", "4 · 3D results", "5 · Help"]
)

with tab_upload:
    st.subheader("Upload object photographs")
    uploaded_files = st.file_uploader(
        "Choose multiple JPG, JPEG, or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{project_name}",
    )
    if uploaded_files:
        previews = []
        inspection_rows = []
        for item in uploaded_files:
            try:
                info = inspect_image_bytes(item.getvalue(), item.name)
                inspection_rows.append(info)
                previews.append(item)
            except Exception as exc:
                inspection_rows.append({"filename": item.name, "valid": False, "error": str(exc)})
        st.dataframe(pd.DataFrame(inspection_rows), use_container_width=True, hide_index=True)
        if st.button("Save valid images to this project", type="primary"):
            try:
                saved, rejected = save_uploaded_images(uploaded_files, paths["images"])
                st.success(f"Saved {len(saved)} image(s) to {paths['images']}")
                if rejected:
                    st.warning("Some files were skipped: " + "; ".join(rejected))
            except Exception as exc:
                st.error(f"Could not save uploads: {exc}")

    current_images = sorted(
        p for p in paths["images"].iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    st.divider()
    st.metric("Images currently saved", len(current_images))
    if current_images:
        st.write("Saved images:")
        st.write(" · ".join(p.name for p in current_images[:20]))
        if len(current_images) > 20:
            st.caption(f"And {len(current_images) - 20} more...")
        if st.button("Remove all saved images", type="secondary"):
            for image_path in current_images:
                image_path.unlink(missing_ok=True)
            st.rerun()
    else:
        st.warning("No images saved yet. Upload images and save them to begin.")

with tab_reconstruct:
    st.subheader("Run COLMAP reconstruction")
    existing_images = sorted(
        p for p in paths["images"].iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    st.write(f"**Project:** `{project_name}`")
    st.write(f"**Input images:** {len(existing_images)}")
    if len(existing_images) < 2:
        st.warning("Upload at least two images. For a meaningful object reconstruction, use 20–50 overlapping images.")
    elif len(existing_images) < 8:
        st.warning(
            f"Only {len(existing_images)} images are saved. COLMAP may work with a small set, "
            "but 20–50 sharp, overlapping photos from multiple heights are strongly recommended."
        )
    st.info(
        "CPU-friendly mode: SIFT extraction, feature matching, sparse mapping, and sparse PLY export run without CUDA. "
        "Leave dense reconstruction off unless your COLMAP build can access a compatible CUDA GPU."
    )
    if run_dense:
        st.warning("Dense PatchMatch stereo is enabled. If CUDA is unavailable, the app will keep the sparse result and report the dense-stage error.")

    if st.button("Start reconstruction", type="primary", disabled=len(existing_images) < 2):
        # Do not let an earlier successful run be mistaken for the result of this attempt.
        st.session_state.pop(f"result_{project_name}", None)
        try:
            executable = resolve_colmap_executable(colmap_input)
            log_box = st.empty()
            progress = st.progress(0)
            log_lines: list[str] = []

            def on_log(message: str, fraction: float | None = None) -> None:
                log_lines.append(message)
                log_box.code("\n".join(log_lines[-100:]), language="text")
                if fraction is not None:
                    progress.progress(max(0.0, min(1.0, fraction)))

            with st.spinner("COLMAP is processing the image set. This may take several minutes or longer..."):
                result = run_colmap_pipeline(
                    project_dir=paths["root"],
                    image_dir=paths["images"],
                    colmap_executable=executable,
                    run_dense=run_dense,
                    max_image_size=int(max_image_size),
                    use_gpu_for_sift=use_gpu_for_sift,
                    log_callback=on_log,
                )
            st.session_state[f"result_{project_name}"] = result
            if result.get("status") == "sparse_completed_dense_failed":
                st.warning(
                    "Sparse reconstruction succeeded, but the optional dense stage failed. "
                    "Your sparse point cloud remains available in the 3D results tab."
                )
            elif result.get("status") == "sparse_completed":
                st.success("Sparse reconstruction completed. The sparse point cloud is ready in 3D results.")
            else:
                st.success("Reconstruction pipeline finished.")
            st.json(result)
        except Exception as exc:
            st.error(str(exc))
            st.caption("Open the Help tab for common installation and reconstruction issues.")

    result = st.session_state.get(f"result_{project_name}")
    if result:
        st.divider()
        st.subheader("Latest run summary")
        st.json(result)

    st.divider()
    st.subheader("Optional: create a mesh from the dense point cloud with Open3D")
    fused_path = paths["dense"] / "fused.ply"
    if fused_path.exists():
        depth = st.slider("Poisson reconstruction depth", min_value=5, max_value=11, value=8)
        if st.button("Generate Open3D Poisson mesh"):
            try:
                with st.spinner("Cleaning point cloud and generating mesh..."):
                    mesh_path, stats = create_poisson_mesh(
                        fused_path,
                        paths["output"] / "open3d_mesh.ply",
                        depth=int(depth),
                    )
                st.success(f"Mesh saved to {mesh_path}")
                st.json(stats)
            except Exception as exc:
                st.error(f"Mesh generation failed: {exc}")
    else:
        st.caption(
            "Mesh generation here uses the dense fused point cloud (dense/fused.ply). "
            "A sparse SfM cloud is useful for visualization but is generally too incomplete for reliable surface meshing."
        )

with tab_features:
    st.subheader("OpenCV feature matching experiment")
    st.caption("This is an independent educational experiment. COLMAP performs its own feature extraction and matching in the main reconstruction pipeline.")
    available_images = sorted(
        p for p in paths["images"].iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(available_images) >= 2:
        filenames = [p.name for p in available_images]
        left_col, right_col = st.columns(2)
        with left_col:
            image_a = st.selectbox("First image", filenames, index=0)
        with right_col:
            image_b = st.selectbox("Second image", filenames, index=1)
        algorithm = st.selectbox("Feature detector", ["SIFT", "ORB"])
        ratio = st.slider("Lowe ratio threshold", 0.5, 0.95, 0.75, 0.05)
        if st.button("Find and visualize matching features"):
            try:
                visual, stats = create_feature_match_visualization(
                    paths["images"] / image_a,
                    paths["images"] / image_b,
                    method=algorithm,
                    ratio=float(ratio),
                )
                st.image(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB), caption="Feature matches", use_container_width=True)
                st.json(stats)
            except Exception as exc:
                st.error(f"Feature matching failed: {exc}")
    else:
        st.info("Save at least two images in the Images tab to run this experiment.")

with tab_results:
    st.subheader("Reconstructed 3D assets")
    assets = discover_assets(paths)
    sparse_model = assets["sparse_model"]
    sparse_point_cloud = assets["sparse_point_cloud"]
    point_cloud = assets["point_cloud"]
    mesh = assets["mesh"]
    textured_obj = assets["textured_obj"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Sparse model", "Available" if sparse_model else "Not found")
    metric_cols[1].metric("Sparse point cloud", "Available" if sparse_point_cloud else "Not exported")
    metric_cols[2].metric("Dense point cloud", "Available" if point_cloud else "Not found")
    metric_cols[3].metric("Mesh", "Available" if mesh else "Not found")

    latest_result = st.session_state.get(f"result_{project_name}")
    if latest_result and latest_result.get("dense_error"):
        st.warning(
            "The sparse stage completed, but dense reconstruction did not. "
            "This does not invalidate the sparse result."
        )
        with st.expander("Dense-stage error details"):
            st.code(latest_result["dense_error"], language="text")

    if sparse_model:
        st.markdown("### Sparse COLMAP model")
        st.success(f"Sparse model found: `{sparse_model}`")
        st.caption(
            "The sparse model contains registered camera poses and triangulated feature points. "
            "It is an initial 3D reconstruction, not a complete solid surface."
        )
        if sparse_point_cloud:
            try:
                st.json(get_point_cloud_stats(sparse_point_cloud))
            except Exception as exc:
                st.warning(f"Could not read sparse point-cloud statistics: {exc}")
            render_asset_preview(sparse_point_cloud, "point_cloud")
            with open(sparse_point_cloud, "rb") as f:
                st.download_button(
                    "Download sparse point cloud (.ply)",
                    f,
                    file_name="sparse.ply",
                    mime="application/octet-stream",
                    key=f"download_sparse_{project_name}",
                )
        else:
            st.warning(
                "The sparse model exists, but `output/sparse.ply` was not exported. "
                "Check the latest run log for a `model_converter` error, then rerun reconstruction to retry the export."
            )
    elif sparse_point_cloud:
        st.markdown("### Sparse point cloud")
        st.caption("A sparse PLY was found, but the corresponding COLMAP model folder was not detected.")
        render_asset_preview(sparse_point_cloud, "point_cloud")
        with open(sparse_point_cloud, "rb") as f:
            st.download_button("Download sparse point cloud (.ply)", f, file_name="sparse.ply", mime="application/octet-stream")

    if point_cloud:
        st.markdown("### Dense point cloud")
        try:
            st.json(get_point_cloud_stats(point_cloud))
        except Exception as exc:
            st.warning(f"Could not read dense point-cloud statistics: {exc}")
        render_asset_preview(point_cloud, "point_cloud")
        with open(point_cloud, "rb") as f:
            st.download_button("Download dense point cloud (.ply)", f, file_name="fused.ply", mime="application/octet-stream")

    if mesh:
        st.markdown("### Mesh")
        render_asset_preview(mesh, "mesh")
        with open(mesh, "rb") as f:
            st.download_button("Download mesh (.ply)", f, file_name=mesh.name, mime="application/octet-stream")
        if st.button("Convert this mesh to GLB"):
            try:
                glb_path = export_mesh_to_glb(mesh, paths["output"] / "reconstructed_model.glb")
                st.success(f"Created {glb_path}")
                with open(glb_path, "rb") as f:
                    st.download_button("Download GLB", f, file_name=glb_path.name, mime="model/gltf-binary")
            except Exception as exc:
                st.error(f"GLB conversion failed: {exc}")

    if textured_obj:
        st.markdown("### Textured mesh")
        st.success(f"Textured OBJ found: {textured_obj}")
        st.caption("Keep the associated .mtl and texture image files together with the OBJ file when moving or sharing the model.")
        with open(textured_obj, "rb") as f:
            st.download_button("Download textured OBJ only", f, file_name=textured_obj.name, mime="text/plain")
        if textured_obj.parent.name.lower() == "textured":
            companion_files = [p for p in textured_obj.parent.rglob("*") if p.is_file()]
        else:
            companion_files = [p for p in textured_obj.parent.glob(textured_obj.stem + "*") if p.is_file()]
            if textured_obj not in companion_files:
                companion_files.append(textured_obj)
        if companion_files:
            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for companion in companion_files:
                    archive.write(companion, arcname=companion.name)
            st.download_button(
                "Download textured model package (.zip)",
                data=archive_buffer.getvalue(),
                file_name="textured_model_package.zip",
                mime="application/zip",
            )
            st.write("Files included in textured package:")
            for companion in companion_files:
                st.write(f"- {companion.name}")

    if not any([sparse_model, sparse_point_cloud, point_cloud, mesh, textured_obj]):
        st.info("No 3D outputs found yet. Run the sparse reconstruction from the Reconstruction tab first.")

with tab_help:
    st.subheader("Capture guidelines")
    st.markdown(
        """
- Use a **stationary object** and move the camera around it.
- Capture roughly **20–50 sharp images** for a first object. Three images are usually too few for a robust object model.
- Aim for strong overlap between neighboring photos (roughly 60–80% is a useful starting target, not a hard rule).
- Capture several height levels so the top and sides are visible.
- Prefer matte, textured, opaque objects.
- Use soft, consistent lighting and avoid motion blur.
- Do not change zoom or focal length during a capture sequence.
- Keep the background and object stationary where possible.
        """
    )
    st.subheader("CPU-only sparse reconstruction")
    st.markdown(
        """
The recommended starting workflow does **not** require CUDA:

1. Upload and save the object photographs.
2. Run COLMAP feature extraction and exhaustive matching with GPU SIFT disabled (CPU mode).
3. Run the COLMAP mapper to estimate camera poses and a **sparse 3D point cloud**.
4. The app exports `output/sparse.ply` and displays/downloads it in **3D results**.

The sparse result is a cloud of triangulated image features. It is useful for checking camera coverage and basic 3D structure, but it is not a complete surface and is not usually suitable for Poisson meshing on its own.
        """
    )
    st.subheader("Optional dense reconstruction")
    st.markdown(
        """
Enable **Run dense reconstruction (CUDA required)** only when your COLMAP build can access a compatible CUDA GPU and its drivers. The optional stages are image undistortion, PatchMatch stereo, stereo fusion, meshing, and texturing. In the standard COLMAP pipeline, PatchMatch stereo is the CUDA-dependent stage.

If CUDA is unavailable, leave dense reconstruction disabled. If it is enabled and fails, the app keeps the completed sparse model and reports the dense-stage error separately.
        """
    )
    st.subheader("COLMAP requirements")
    st.markdown(
        """
- Install COLMAP separately; Python package dependencies do not install the COLMAP executable.
- Put `colmap` on your PATH or enter the full path to `COLMAP.bat` / `colmap` in the sidebar.
- The app uses CPU mode for SIFT extraction and matching by default (`--SiftExtraction.use_gpu 0` and `--SiftMatching.use_gpu 0`).
- Dense PatchMatch stereo may require a CUDA-enabled COLMAP build and a compatible NVIDIA GPU/driver setup; the presence of `COLMAP.bat` alone does not guarantee CUDA availability.
- Keep the Streamlit process open while COLMAP runs.
        """
    )
    st.subheader("Troubleshooting")
    st.markdown(
        """
| Problem | What to try |
|---|---|
| `COLMAP executable not found` | Install COLMAP and set its full executable path in the sidebar. |
| `COLMAP mapper did not produce a sparse model` | Read the diagnostic counts in the error. If keypoints are zero, check image focus/texture. If putative matches are zero, improve overlap. If geometric pairs are zero, capture closer neighboring views. Prefer 20–50 images; three images are a very small set. |
| Mapper runs but Results says no output | Check that `sparse\0` contains `cameras.bin`, `images.bin`, and `points3D.bin`. The app now searches nested sparse folders and exports `output/sparse.ply`. |
| CUDA / `num_cuda_devices` / PatchMatch error | Disable dense reconstruction. The sparse model and `output/sparse.ply` should still be retained. |
| Sparse PLY preview unavailable | The built-in PLY preview supports common ASCII and binary PLY point clouds without Open3D. Install project requirements if a different PLY variant or mesh preview needs Open3D. |
| Very noisy point cloud | Remove blurry images, improve overlap, and reduce background clutter. |
| Mesh has bulges or extra surfaces | Improve dense coverage, crop the point cloud, and tune Poisson depth / density filtering. |
| Texturing fails | Ensure undistortion succeeded and check `mesh_texturer -h` for your installed COLMAP version. |
        """
    )
    st.subheader("Project output folders")
    st.code(
        f"{paths['root']}\n"
        "├── images/                 source photos\n"
        "├── database.db            COLMAP features and matches\n"
        "├── sparse/0/              sparse COLMAP camera/point model\n"
        "├── output/sparse.ply      exported sparse point cloud\n"
        "└── dense/                 optional dense workspace and outputs",
        language="text",
    )
    st.caption("Each project keeps its source images, COLMAP database, sparse model, exported sparse cloud, and optional dense assets in its own folder.")
