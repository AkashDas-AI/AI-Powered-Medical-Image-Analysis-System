"""Unit tests for the image preprocessing module.

Verifies PIL conversions, target dimensions, MobileNetV2 scaling bounds [-1.0, 1.0],
float32 data types, and defensive error handling on invalid inputs.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import deprocess_image, preprocess_for_model, preprocess_image


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Generate a synthetic 300x400 RGB test image with distinct color gradients."""
    arr = np.linspace(0, 255, num=300 * 400 * 3, dtype=np.uint8).reshape((300, 400, 3))
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Generate a synthetic 256x256 grayscale (mode L) image resembling an X-ray."""
    arr = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Generate a synthetic 150x150 RGBA image."""
    arr = np.zeros((150, 150, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # Fully opaque
    return Image.fromarray(arr, mode="RGBA")


def test_preprocessing_rgb_shape_and_dtype(sample_rgb_image: Image.Image) -> None:
    """Verify preprocessing transforms RGB images to (224, 224, 3) float32."""
    processed = preprocess_image(sample_rgb_image, target_size=(224, 224))
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32


def test_preprocessing_grayscale_converts_to_3_channels(
    sample_grayscale_image: Image.Image,
) -> None:
    """Verify grayscale single-channel images are duplicated to 3 identical RGB channels."""
    processed = preprocess_image(sample_grayscale_image, target_size=(224, 224))
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    # Verify channel equality for converted grayscale
    np.testing.assert_allclose(processed[:, :, 0], processed[:, :, 1], rtol=1e-5)
    np.testing.assert_allclose(processed[:, :, 1], processed[:, :, 2], rtol=1e-5)


def test_preprocessing_rgba_converts_to_rgb(sample_rgba_image: Image.Image) -> None:
    """Verify 4-channel RGBA images drop alpha and yield 3-channel RGB output."""
    processed = preprocess_image(sample_rgba_image, target_size=(224, 224))
    assert processed.shape == (224, 224, 3)


def test_mobilenet_v2_pixel_scaling_bounds() -> None:
    """Verify MobileNetV2 normalization scales exactly to [-1.0, 1.0]."""
    # Black image (0) -> (0 / 127.5) - 1.0 = -1.0
    black_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    proc_black = preprocess_image(black_img)
    np.testing.assert_allclose(proc_black, -1.0, atol=1e-5)

    # White image (255) -> (255 / 127.5) - 1.0 = 1.0
    white_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    proc_white = preprocess_image(white_img)
    np.testing.assert_allclose(proc_white, 1.0, atol=1e-5)

    # Mid-gray image (128) -> (128 / 127.5) - 1.0 approx 0.0039
    gray_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    proc_gray = preprocess_image(gray_img)
    assert np.all(proc_gray >= -1.0) and np.all(proc_gray <= 1.0)
    assert np.isclose(proc_gray.mean(), 0.0, atol=0.01)


def test_preprocessing_from_numpy_array() -> None:
    """Verify preprocessing accepts raw 2D and 3D NumPy arrays."""
    # 2D array
    arr_2d = np.full((128, 128), fill_value=200, dtype=np.uint8)
    proc_2d = preprocess_image(arr_2d)
    assert proc_2d.shape == (224, 224, 3)

    # 3D array
    arr_3d = np.full((100, 100, 3), fill_value=50, dtype=np.uint8)
    proc_3d = preprocess_image(arr_3d)
    assert proc_3d.shape == (224, 224, 3)


def test_preprocess_for_model_batch_dimension(sample_rgb_image: Image.Image) -> None:
    """Verify preprocess_for_model adds batch dimension: (1, 224, 224, 3)."""
    batch_tensor = preprocess_for_model(sample_rgb_image)
    assert batch_tensor.shape == (1, 224, 224, 3)
    assert batch_tensor.dtype == np.float32


def test_deprocess_image_inversion() -> None:
    """Verify deprocess_image converts [-1.0, 1.0] float array back to [0, 255] uint8."""
    tensor = np.array([[-1.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    # Expand to 3 channels and H, W
    tensor_3d = np.repeat(tensor[:, :, np.newaxis], 3, axis=2)
    deprocessed = deprocess_image(tensor_3d)

    assert deprocessed.dtype == np.uint8
    assert deprocessed[0, 0, 0] == 0
    assert np.isclose(deprocessed[0, 1, 0], 127, atol=1)
    assert deprocessed[1, 0, 0] == 255


def test_invalid_input_error_handling() -> None:
    """Verify appropriate exceptions are raised for invalid inputs."""
    with pytest.raises(TypeError, match="Input image cannot be None"):
        preprocess_image(None)  # type: ignore

    with pytest.raises(TypeError, match="Unsupported image input type"):
        preprocess_image(12345)  # type: ignore

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        preprocess_image("non_existent_image_path_12345.png")

    with pytest.raises(ValueError, match="Input numpy array is empty"):
        preprocess_image(np.array([], dtype=np.uint8))


def test_dicom_preprocessing_parity_and_extensions(tmp_path: Path) -> None:
    """Verify exact parity between load_dicom_as_array, preprocess_image, and preprocess_for_model.

    Also tests both .dcm and .dicom extensions for complete format coverage.
    """
    try:
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    except ImportError:
        pytest.skip("pydicom is required for DICOM parity testing.")

    # Create synthetic test DICOM
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dcm_file = tmp_path / "parity_test.dcm"
    ds = FileDataset(str(dcm_file), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = "PARITY_PID_999"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PhotometricInterpretation = "MONOCHROME2"

    raw_grid = np.linspace(10, 240, 64 * 64, dtype=np.uint8).reshape((64, 64))
    ds.Rows, ds.Columns = raw_grid.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PixelData = raw_grid.tobytes()
    ds.save_as(str(dcm_file))

    # Also create a .dicom copy
    dicom_ext_file = tmp_path / "parity_test.dicom"
    ds.save_as(str(dicom_ext_file))

    from src.rsna_dataset import load_dicom_as_array

    # 1. Direct path (as used in RSNABatchGenerator)
    direct_uint8 = load_dicom_as_array(dcm_file, target_size=(224, 224))
    direct_norm = (direct_uint8.astype(np.float32) / 127.5) - 1.0

    # 2. Unified preprocess_image path (.dcm)
    prep_norm = preprocess_image(dcm_file, target_size=(224, 224))

    # 3. Unified preprocess_image path (.dicom)
    prep_dicom_norm = preprocess_image(dicom_ext_file, target_size=(224, 224))

    # 4. Model input tensor path (preprocess_for_model)
    model_tensor = preprocess_for_model(dcm_file, target_size=(224, 224))

    # Exact bitwise parity assertions
    np.testing.assert_array_equal(prep_norm, direct_norm)
    np.testing.assert_array_equal(prep_dicom_norm, direct_norm)
    np.testing.assert_array_equal(model_tensor[0], direct_norm)

    assert model_tensor.shape == (1, 224, 224, 3)
    assert model_tensor.dtype == np.float32
    assert np.all(model_tensor >= -1.0) and np.all(model_tensor <= 1.0)


