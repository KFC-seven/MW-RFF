"""Source-fitted spectral/PCA frontend and canonical XFR frequency rows."""
import numpy as np
import pywt


def spectral_bank(physical, rms, reference_z):
    iq = np.ascontiguousarray(physical.reshape(-1, 160, 2) / np.float32(rms))
    bank = np.zeros_like(iq)
    for first in range(0, len(iq), 512):
        part = iq[first:first + 512]
        z = part[..., 0] + 1j * part[..., 1]
        spectrum = np.fft.fftshift(np.fft.fft(z, n=160, axis=-1), axes=-1) / np.sqrt(160)
        logarithm = np.log(np.abs(spectrum) + 1e-8 / np.sqrt(160))
        coefficients = pywt.wavedec(logarithm, "db6", level=3, axis=-1, mode="symmetric")
        coefficients[0] = np.zeros_like(coefficients[0])
        bank[first:first + 512, :, 0] = pywt.waverec(coefficients, "db6", axis=-1, mode="symmetric")[..., :160]
    if reference_z:
        values = bank[:, :, 0].astype(np.float64)
        values = (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-8)
        bank[:, :, 0] = values.astype(np.float32)
    if not np.isfinite(bank).all():
        raise ValueError("Nonfinite receiver-disjoint spectral features")
    return bank


def project(bank, pca):
    projection = np.empty((len(bank), 32), np.float32)
    for first in range(0, len(bank), 512):
        projection[first:first + 512] = ((bank[first:first + 512, :, 0].astype(np.float64) - pca["mean"])
                                         @ pca["components"]).astype(np.float32)
    return projection


def row_inputs(bank, projection, row_ids):
    """Construct selected canonical [row,160,52] inputs without a full row cache."""
    row_ids = np.asarray(row_ids, np.int64)
    if row_ids.ndim != 1 or np.any(row_ids < 0) or np.any(row_ids >= len(bank)):
        raise ValueError("Invalid canonical frequency-row IDs")
    frequencies = row_ids % 160
    reference_ids = (row_ids // 160)[:, None] * 160 + np.arange(160)[None, :]
    output = np.empty((len(row_ids), 160, 52), np.float32)
    output[..., :2] = bank[reference_ids, frequencies[:, None], :]
    for channel, offset in enumerate((*range(-8, 0), *range(1, 9)), 2):
        output[..., channel] = bank[reference_ids, np.clip(frequencies + offset, 0, 159)[:, None], 0]
    coordinates = np.stack((np.sin(2 * np.pi * np.arange(160) / 160),
                            np.cos(2 * np.pi * np.arange(160) / 160)), -1).astype(np.float32)
    output[..., 18:20] = coordinates[frequencies, None, :]
    output[..., 20:] = projection[reference_ids]
    return output


def batches(bank, projection):
    # Match the historical 256-row evaluation batches, including block crossings.
    for first in range(0, len(bank), 256):
        yield row_inputs(bank, projection, np.arange(first, min(first + 256, len(bank))))
