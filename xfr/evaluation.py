"""Position-level classification and matched-budget block decisions."""

import numpy as np
import torch


@torch.inference_mode()
def evaluate_inputs(model, inputs, labels, block_ids, device, batch_size=512):
    """Classify rows in block order and fuse the logits within each physical block."""
    ids = np.asarray(block_ids)
    y = np.asarray(labels)
    if len(inputs) == 0 or y.shape != (len(inputs),) or ids.shape != y.shape:
        raise ValueError("Expected nonempty rows with one label and block ID per row.")
    if batch_size < 1:
        raise ValueError("The evaluation batch size must be positive.")
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(ids)])
    if len(np.unique(ids[starts])) != len(starts) or not (sizes == sizes[0]).all():
        raise ValueError("Each block must supply the same number of consecutive rows.")
    truth = y[starts]
    if not (y.reshape(len(starts), -1) == truth[:, None]).all():
        raise ValueError("Every row of a block must share its transmitter label.")
    model.eval()
    scores = []
    for begin in range(0, len(inputs), batch_size):
        batch = torch.as_tensor(inputs[begin:begin+batch_size], dtype=torch.float32, device=device)
        scores.append(model(batch).cpu())
    outputs = torch.cat(scores).reshape(len(starts), int(sizes[0]), -1)
    means = outputs.mean(dim=1)
    row_preds = outputs.argmax(-1).numpy()
    block_preds = means.argmax(-1).numpy()
    row_correct = int(np.count_nonzero(row_preds == truth[:, None]))
    block_correct = int(np.count_nonzero(block_preds == truth))
    metrics = {
        "rows": len(inputs), "row_correct": row_correct, "row_accuracy_pct": 100 * row_correct / len(inputs),
        "blocks": len(truth), "block_correct": block_correct, "block_accuracy_pct": 100 * block_correct / len(truth),
    }
    predictions = {
        "block_ids": ids[starts], "labels": truth, "model_outputs": outputs.numpy(),
        "primary_scores": means.numpy(), "elementary_predictions": row_preds,
    }
    return metrics, predictions
