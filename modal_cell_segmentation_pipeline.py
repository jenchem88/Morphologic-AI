import modal
import os
import modal.parallel_map
import numpy as np

s3_bucket_name = "cell-segmentation-data-a05c1c99b4410510" 
os.environ["AWS_REGION"] = "us-west-2"
CELLPOSE_MODELS_DIR = "~/.cellpose/models"
# cell segmentation for the rxrx dataset (https://www.rxrx.ai/rxrx19a#Download) powered by Modal 

# Define the Modal image
def create_image():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install(
            "git+https://github.com/neibla/cell-segmentation@e3976600663bef70354eae361938b4f2b1b6390b"
        )
        .pip_install(
            "git+https://github.com/MouseLand/cellpose@52f75f9636250979dcff48da18d24aaff41c2c86"
        )
    )

    # Download and save the Cellpose model
    image = image.run_commands(
        f"mkdir -p {CELLPOSE_MODELS_DIR}",
        f"python3 -c \"from cellpose import models; models.CellposeModel(model_type='cyto3', pretrained_model='{CELLPOSE_MODELS_DIR}')\"",
    )
    return image


# Create a Modal App
app = modal.App("cell-segmentation")
image = create_image()


bucket_creds = modal.Secret.from_name("aws-secret")
volume = modal.CloudBucketMount(s3_bucket_name, secret=bucket_creds)

BATCH_SIZE = 512


@app.function(
    volumes={"/s3-bucket": volume}, image=image, secrets=[bucket_creds], timeout=21600
)
def load_data_from_s3(input_dir: str) -> list[str]:
    from cell_segmentor.utils.s3_utils import list_s3_files
    from cell_segmentor.utils.image_utils import is_image_file

    files = list_s3_files(input_dir)
    image_files = [file for file in files if is_image_file(file)]
    return image_files


@app.function(
    gpu="A10G", image=image, secrets=[bucket_creds], concurrency_limit=5, timeout=21600
)
def segment_images(images: list):
    from cellpose import models
    import torch
    import logging

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda")
    logging.info(f"Loading modal")
    model = models.CellposeModel(
        model_type="cyto3",
        gpu=True,
        device=device,
        pretrained_model=CELLPOSE_MODELS_DIR,
    )

    logging.info(f"Running model")
    config = {"channels": [0, 0], "niter": 100, "batch_size": 256}
    masks, _, _ = model.eval(images, **config)
    return masks


@app.function(
    # volumes={"/s3-bucket": volume},
    image=image,
    secrets=[bucket_creds],
    concurrency_limit=50,
    timeout=21600,
)
def extract_and_upload_cells_for_result(img_src, mask, image, output_dir):
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    import numpy as np
    from cell_segmentor.utils.image_utils import save_image, save_mask
    from numba import njit

    @njit
    def find_true_indices(mask):
        rows, cols = [], []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    rows.append(i)
                    cols.append(j)
        return np.array(rows), np.array(cols)

    def process_cell(
        cell_id: int,
        cell_mask: np.ndarray,
        image_np: np.ndarray,
        output_dir: str,
    ) -> None:
        rows, cols = find_true_indices(cell_mask)
        top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()

        cell_image = image_np[top : bottom + 1, left : right + 1]
        cell_mask_cropped = cell_mask[top : bottom + 1, left : right + 1]

        if len(cell_image.shape) == 3 and len(cell_mask_cropped.shape) == 2:
            cell_mask_cropped = cell_mask_cropped[:, :, np.newaxis]

        masked_cell = cell_image * cell_mask_cropped

        # Optimize normalization using vectorized operations
        min_val = np.min(masked_cell)
        max_val = np.max(masked_cell)
        if max_val > min_val:
            masked_cell = np.clip(
                (masked_cell - min_val) * (255.0 / (max_val - min_val)), 0, 255
            ).astype(np.uint8)
        else:
            masked_cell = np.zeros_like(masked_cell, dtype=np.uint8)
        cell_filename = (
            f"{output_dir}/cell_{cell_id}_bbox_{top}_{bottom}_{left}_{right}.png"
        )
        save_image(masked_cell, cell_filename)

    def get_output_path(path: str, output_dir: str) -> str:
        file_name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(output_dir, file_name)
        return output_path

    def save_detected_cells(masks_pred, image_np, output_dir) -> None:
        # save the mask
        logging.info(f"Saving mask for {output_dir}")
        save_mask(masks_pred, f"{output_dir}/mask.npz")
        cells_output_dir = os.path.join(output_dir, "cells")
        unique_cells = np.unique(masks_pred)[1:]
        cell_masks = {cell_id: masks_pred == cell_id for cell_id in unique_cells}
        logging.info(f"Extracting segments for {output_dir}")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_cell, i, cell_masks[i], image_np, cells_output_dir) for i in unique_cells]
                errors = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        errors.append(f"Error processing cell: {str(e)}")
                
                if errors:
                    logging.error(f"Encountered {len(errors)} errors while processing cells:")
                    for error in errors:
                        logging.error(error)
        except Exception as e:
            import traceback

            logging.error(f"An error occurred: {e}")
            logging.error(f"Stack trace:\n{traceback.format_exc()}")
        logging.info(f"Saved {len(unique_cells)} detected cells to {output_dir}")

    def load_and_extract(image, path, mask, output_dir):
        logging.info(f"Loading and extracting {path}")
        output_dir = get_output_path(path, output_dir)
        logging.info(f"Saving results to {output_dir} for {path}")
        save_detected_cells(mask, image, output_dir)

    import concurrent.futures

    logging.info(f"Saving {img_src} results to root: {output_dir}")
    load_and_extract(image, img_src, mask, output_dir)
    return img_src

@app.function(
    volumes={"/s3-bucket": volume},
    image=image,
    secrets=[bucket_creds],
    concurrency_limit=10,
    timeout=21600,
)
def process_batch(batch):
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    (batch, output_dir, i) = batch
    from cell_segmentor.utils.image_utils import load_image
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        images = list(executor.map(load_image, batch))
    logging.info(f"Processing batch of {len(batch)} images {i}")
    segmentation_results = segment_images.remote(images)
    logging.info(f"Completed segmentation for {len(batch)} images {i}")
    

    for src in extract_and_upload_cells_for_result.map(
        (img_src for img_src in batch),
        (mask for mask in segmentation_results),
        (image for image in images),
        (output_dir for _ in batch),
    ):
        logging.info(f"Completed processing batch {src}")
    logging.info(f"Completed processing batch of {len(batch)} images {i}")
    return i


@app.function(concurrency_limit=5, timeout=30600)
def run_pipeline():
    import logging

    logging.basicConfig(level=logging.INFO)
    plates = [
        # "Plate11",
        "Plate12",
        "Plate13",
        "Plate14",
        "Plate19",
        "Plate20",
        "Plate24",
    ]
    # plates = ["Plate26"]
    for plate in plates:
        input_dir = f"s3://cell-segmentation-data-a05c1c99b4410510/data/RxRx19a/images/HRCE-1/{plate}"
        output_dir = f"s3://cell-segmentation-data-a05c1c99b4410510/outputs/RxRx19a/HRCE-1/{plate}"
        image_files = load_data_from_s3.remote(input_dir)
        logging.info(f"Begin processing {input_dir}")
        image_files = image_files
        batches = [
            (image_files[i : i + BATCH_SIZE], output_dir, i)
            for i in range(0, len(image_files), BATCH_SIZE)
        ]
        for i in process_batch.map(batches):
            logging.info(f"Completed processing batch {i}")
        logging.info(f"Completed processing {input_dir}")

    logging.info("Completed processing all batches")


@app.local_entrypoint()
def main():
    run_pipeline.remote()
