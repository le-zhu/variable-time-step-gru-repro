import torch


def _select_step_dt(dt, t, batch_size):
    if dt.ndim == 0:
        return dt.expand(batch_size).view(-1, 1)
    if dt.ndim == 1:
        return dt.view(-1, 1)
    if dt.ndim == 2 and dt.shape[1] == 1:
        return dt
    if dt.ndim == 2:
        return dt[:, t].view(-1, 1)
    raise ValueError(f"Unsupported dt shape: {tuple(dt.shape)}")


def rollout_sequence(model, input_seq, h_gt, dt, *, teacher_forcing_boundary, device, lengths=None):
    batch_size, seq_len, _feature_count = input_seq.shape
    h_curr = h_gt[:, 0, :]
    hs = torch.zeros(batch_size, model.hidden_size, device=device)
    preds = [h_curr]

    if lengths is None:
        lengths_tensor = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    else:
        lengths_tensor = lengths.to(device=device, dtype=torch.long)

    for t in range(max(seq_len - 1, 0)):
        input_t = input_seq[:, t, :]
        gt_t = h_gt[:, t, :]
        valid_step = t < (lengths_tensor - 1)
        teacher_mask = valid_step.unsqueeze(1) if t < teacher_forcing_boundary else torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=device
        )
        h_input = torch.where(
            teacher_mask,
            gt_t,
            h_curr,
        )
        h_next, hs_next = model.forward_step(input_t, h_input, hs, dt=dt)
        valid_step_exp = valid_step.unsqueeze(1)
        hs = torch.where(valid_step_exp, hs_next, hs)
        h_curr = torch.where(valid_step_exp, h_next, h_curr)
        preds.append(h_curr)

    pred = torch.stack(preds, dim=1)
    time_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    target_mask = (time_idx > 0) & (time_idx < lengths_tensor.unsqueeze(1))
    return pred, target_mask


def rollout_sequence_original_grid(
    model,
    input_seq,
    h_gt,
    dt,
    original_indices,
    *,
    warmup_original_steps,
    device,
    lengths=None,
):
    """Roll out on a resampled grid while defining warmup on the original grid.

    The current ground-truth H is used only when that input point's original-grid
    coordinate is strictly smaller than ``warmup_original_steps``.  Evaluation
    starts at the output produced by the first autoregressive H input.
    """
    batch_size, seq_len, _feature_count = input_seq.shape
    if tuple(original_indices.shape) != (batch_size, seq_len):
        raise ValueError(
            "original_indices must have shape "
            f"{(batch_size, seq_len)}, got {tuple(original_indices.shape)}"
        )

    h_curr = h_gt[:, 0, :]
    hs = torch.zeros(batch_size, model.hidden_size, device=device)
    preds = [h_curr]

    if lengths is None:
        lengths_tensor = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    else:
        lengths_tensor = lengths.to(device=device, dtype=torch.long)
    original_indices = original_indices.to(device=device, dtype=torch.long)

    for t in range(max(seq_len - 1, 0)):
        input_t = input_seq[:, t, :]
        gt_t = h_gt[:, t, :]
        valid_step = t < (lengths_tensor - 1)
        use_teacher = valid_step & (original_indices[:, t] < warmup_original_steps)
        h_input = torch.where(use_teacher.unsqueeze(1), gt_t, h_curr)

        dt_t = _select_step_dt(dt, t, batch_size)
        h_next, hs_next = model.forward_step(input_t, h_input, hs, dt=dt_t)
        valid_step_expanded = valid_step.unsqueeze(1)
        hs = torch.where(valid_step_expanded, hs_next, hs)
        h_curr = torch.where(valid_step_expanded, h_next, h_curr)
        preds.append(h_curr)

    pred = torch.stack(preds, dim=1)
    time_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    target_mask = (time_idx > 0) & (time_idx < lengths_tensor.unsqueeze(1))
    eval_mask = torch.zeros_like(target_mask)
    if seq_len > 1:
        source_is_autoregressive = original_indices[:, :-1] >= warmup_original_steps
        eval_mask[:, 1:] = target_mask[:, 1:] & source_is_autoregressive
    return pred, target_mask, eval_mask


def sequence_relative_rms(pred, target, target_mask, row_weights=None):
    diff = (pred - target).squeeze(-1)
    target_values = target.squeeze(-1)
    mask = target_mask.float()
    sum_sq_err = ((diff ** 2) * mask).sum(dim=1)
    sum_sq_tgt = ((target_values ** 2) * mask).sum(dim=1)
    count = target_mask.sum(dim=1)

    row_err = torch.full_like(sum_sq_err, float("nan"))
    valid = count > 0
    if valid.any():
        rms_err = torch.sqrt(sum_sq_err[valid] / count[valid])
        rms_tgt = torch.sqrt(sum_sq_tgt[valid] / count[valid])
        row_err[valid] = rms_err / rms_tgt
        valid_row_err = row_err[valid]
        if row_weights is None:
            loss = valid_row_err.mean()
        else:
            weights = row_weights.to(device=pred.device, dtype=pred.dtype).view(-1)[valid]
            weight_sum = weights.sum()
            if bool(weight_sum.detach() > 0):
                loss = (valid_row_err * weights).sum() / weight_sum
            else:
                loss = valid_row_err.mean()
    else:
        loss = pred.sum() * 0.0
    return loss, row_err
