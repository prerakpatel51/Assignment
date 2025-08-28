# app_gradio.py
# Gradio UI that uses calibration_utils.py

import os, json, glob, shutil
import numpy as np
import gradio as gr
import cv2

from calibration_utils import IO, Board, Img, Calib, Overlay, Render

# Paths that work in Colab OR local
IS_COLAB = os.path.exists("/content")
BASE_DIR = "/content" if IS_COLAB else os.getcwd()
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_JSON = os.path.join(BASE_DIR, "calibration.json")
IO.ensure_dir(IMG_DIR)

def save_uploads(files):
    # Clean previous images
    for f in glob.glob(os.path.join(IMG_DIR, "*")):
        try: os.remove(f)
        except: pass

    saved = 0
    if files:
        for i, f in enumerate(files):
            if getattr(f, "name", None):
                base = os.path.splitext(os.path.basename(f.name))[0] + ".jpg"
                out_path = os.path.join(IMG_DIR, f"calib_{i:03d}_{base}")
                shutil.copy(f.name, out_path)
                saved += 1
    return f"Saved {saved} image(s) to {IMG_DIR}"

def run_calibration(cols, rows, square_size_m):
    paths = IO.list_images(IMG_DIR)
    if len(paths) < 5:
        return f"Need at least ~5 images; found {len(paths)} in {IMG_DIR}. Add more and retry."
    pattern = (int(cols), int(rows))
    try:
        result = Calib.calibrate(paths, pattern_size=pattern, square_size=float(square_size_m))
    except Exception as e:
        return f"Calibration failed: {e}"
    IO.save_json(result, OUT_JSON)

    K = np.array(result["K"])
    D = np.array(result["D"])
    msg = [
        f"Saved calibration to: {OUT_JSON}",
        f"Views used: {result['num_views']}",
        f"Image size: {result['image_size']}",
        f"RMS reprojection error: {result['ret_rms']:.4f}",
        f"Mean per-view error: {result['mean_reproj_error']:.4f}",
        f"K (intrinsics):\n{K}",
        f"D (distortion): {D}",
    ]
    return "\n".join(msg)

def show_poses_and_overlays(max_overlays, use_plotly=True):
    if not os.path.exists(OUT_JSON):
        return None, [], None
    calib = IO.load_json(OUT_JSON)

    # Unpack
    K = np.array(calib["K"], dtype=np.float64)
    D = np.array(calib["D"], dtype=np.float64)
    cols = int(calib["pattern_size"]["cols"]); rows = int(calib["pattern_size"]["rows"])
    square_size = float(calib["square_size_m"])
    extrinsics = calib.get("extrinsics", [])
    valid_paths = calib.get("valid_paths", [])
    if not valid_paths:
        valid_paths = IO.list_images(IMG_DIR)

    # 3D plot
    try:
        fig = Render.plot_poses_plotly(extrinsics, square_size=square_size, board_size=(cols, rows)) \
              if use_plotly else \
              Render.plot_poses_matplotlib(extrinsics, square_size=square_size, board_size=(cols, rows))
    except Exception as e:
        print("Pose plot error:", e)
        fig = None

    # Overlays
    overlays = Overlay.make_sample_overlays(valid_paths, K, D, (cols, rows), square_size, max_images=int(max_overlays))
    gallery = [img for (_p, img) in overlays]  # list of RGB numpy arrays

    # Undistort preview
    und_preview = None
    if valid_paths:
        sample = IO.imread_rgb(valid_paths[0])
        if sample is not None:
            und = Calib.undistort(sample, K, D, keep_size=True)
            # side-by-side
            pad = np.ones((sample.shape[0], 10, 3), dtype=np.uint8)*255
            side = np.hstack([sample, pad, und])
            und_preview = side

    return fig, gallery, und_preview

def create_interface():
    with gr.Blocks(title="Camera Calibration (OpenCV + Gradio)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Camera Calibration (OpenCV + Gradio)\nUpload chessboard images, run calibration, and visualize results.\n\n- Images saved to **/content/images** (Colab) or `./images`\n- Results saved to **calibration.json** in the project root\n- Pattern size = INNER corners (e.g., 9×6)\n")

        with gr.Row():
            uploader = gr.Files(label="Upload .jpg/.jpeg/.png images", file_types=["image"])
            upload_btn = gr.Button("Save to images/")

        upload_msg = gr.Textbox(label="Upload Log", interactive=False)

        with gr.Accordion("Calibration Parameters", open=True):
            with gr.Row():
                cols = gr.Number(value=9, label="Inner corners (cols)", precision=0)
                rows = gr.Number(value=6, label="Inner corners (rows)", precision=0)
                sq = gr.Number(value=0.025, label="Square size (meters)")

            run_btn = gr.Button("Run Calibration")
            calib_out = gr.Textbox(label="Calibration Output", lines=12)

        with gr.Accordion("Visualizations", open=True):
            with gr.Row():
                max_ov = gr.Slider(3, 12, value=6, step=1, label="Max overlay images")
                use_plotly = gr.Checkbox(value=True, label="Interactive 3D (Plotly)")

            pose_plot = gr.Plot(label="Camera Poses (3D)")
            gallery = gr.Gallery(label="Sample images with axes", columns=3, height=300)
            und_preview = gr.Image(label="Undistort Preview (left=original, right=undistorted)")

            viz_btn = gr.Button("Generate Visualizations")

        # Wiring
        upload_btn.click(fn=save_uploads, inputs=[uploader], outputs=[upload_msg])
        run_btn.click(fn=run_calibration, inputs=[cols, rows, sq], outputs=[calib_out])
        viz_btn.click(fn=show_poses_and_overlays, inputs=[max_ov, use_plotly],
                      outputs=[pose_plot, gallery, und_preview])
    return demo

if __name__ == "__main__":
    demo = create_interface()
    # Local default: http://127.0.0.1:7860
    demo.launch(share=False, debug=False)
