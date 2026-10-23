import torch
from tqdm import tqdm

"""
We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""

_EPS = 1e-8


def _interval_mean(patch2image, topmin_min=0, topmin_max=0.3):
    # interval average (same logic as the original implementation)
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1] * k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1] * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max

    # Be robust for small N.
    k_max = max(1, min(int(k_max), int(patch2image.shape[1])))
    k_min = max(0, min(int(k_min), int(k_max - 1)))

    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    if k_max - k_min > 0:
        vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)
    return torch.mean(vals, dim=1)


def _interval_mean_batched(patch2image, topmin_min=0, topmin_max=0.3):
    """Batched interval mean for patch2image with shape (B, patch_num, ref_num)."""
    if patch2image.dim() != 3:
        raise ValueError(f"Expected 3D patch2image (B, P, R), got {tuple(patch2image.shape)}")

    k_max = topmin_max
    k_min = topmin_min
    ref_num = int(patch2image.shape[2])
    if ref_num <= 0:
        b = int(patch2image.shape[0])
        p = int(patch2image.shape[1])
        return torch.zeros((b, p), device=patch2image.device, dtype=torch.float32)

    if k_max < 1:
        k_max = int(ref_num * k_max)
    if k_min < 1:
        k_min = int(ref_num * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max

    k_max = max(1, min(int(k_max), ref_num))
    k_min = max(0, min(int(k_min), int(k_max - 1)))

    vals, _ = torch.topk(patch2image.float(), k_max, dim=2, largest=False, sorted=True)
    if k_max - k_min > 0:
        vals, _ = torch.topk(vals.float(), k_max - k_min, dim=2, largest=True, sorted=True)
    return vals.mean(dim=2)


def _min_cdist_chunked(Z, i, device, chunk_size=8):
    """Compute patch2image (patch_num, image_num-1) with chunked cdist to reduce peak memory."""
    image_num, patch_num, c = Z.shape
    if image_num <= 1:
        return torch.zeros((patch_num, 0), device=device, dtype=torch.float32)

    chunk_size = int(chunk_size)
    chunk_size = max(1, min(chunk_size, image_num))
    target = Z[i]  # (patch_num, c)
    patch2image = torch.empty((patch_num, image_num - 1), device=device, dtype=torch.float32)

    col = 0

    def _process_chunk(z_chunk):
        nonlocal col
        if z_chunk.numel() == 0:
            return
        b = z_chunk.shape[0]
        dist = torch.cdist(target.unsqueeze(0), z_chunk.reshape(-1, c))
        dist = dist.squeeze(0).reshape(patch_num, b, patch_num)
        patch2image[:, col : col + b] = torch.min(dist, dim=2).values
        col += b

    for start in range(0, image_num, chunk_size):
        end = min(image_num, start + chunk_size)
        if start <= i < end:
            _process_chunk(Z[start:i])
            _process_chunk(Z[i + 1 : end])
        else:
            _process_chunk(Z[start:end])

    if col != image_num - 1:
        raise RuntimeError(f"Chunked MSM fill mismatch: col={col}, expected={image_num - 1}")
    return patch2image


def _min_cdist_chunked_refs(target, refs, device, chunk_size=8):
    """Compute patch2image (patch_num, ref_num) with chunked cdist over reference images.

    Args:
        target: (patch_num, c)
        refs: (ref_num, patch_num, c)
        device: torch.device or device string
        chunk_size: number of reference images per chunk

    Returns:
        patch2image: (patch_num, ref_num)
    """
    if refs is None or refs.numel() == 0:
        patch_num = int(target.shape[0])
        return torch.zeros((patch_num, 0), device=device, dtype=torch.float32)

    ref_num, patch_num, c = refs.shape
    chunk_size = int(chunk_size)
    chunk_size = max(1, min(chunk_size, int(ref_num)))
    patch2image = torch.empty((patch_num, ref_num), device=device, dtype=torch.float32)

    col = 0

    for start in range(0, ref_num, chunk_size):
        end = min(ref_num, start + chunk_size)
        z_chunk = refs[start:end]
        b = int(z_chunk.shape[0])
        if b <= 0:
            continue
        dist = torch.cdist(target.unsqueeze(0), z_chunk.reshape(-1, c))
        dist = dist.squeeze(0).reshape(patch_num, b, patch_num)
        patch2image[:, col : col + b] = torch.min(dist, dim=2).values
        col += b

    if col != ref_num:
        raise RuntimeError(f"Chunked MSM(refs) fill mismatch: col={col}, expected={ref_num}")
    return patch2image


def compute_scores_fast_with_refs(
    Z,
    i,
    device,
    ref_indices,
    topmin_min=0,
    topmin_max=0.3,
    cdist_chunk_size=0,
):
    """Compute MSM scores for image i using a restricted reference set.

    Args:
        Z: (image_num, patch_num, c)
        i: target image index
        device: torch.device
        ref_indices: 1D LongTensor of reference image indices
        topmin_min/topmin_max: interval mean params
        cdist_chunk_size: if >0, chunk over reference images for lower peak memory
    """
    image_num, patch_num, c = Z.shape
    if image_num <= 1:
        return torch.zeros((patch_num,), device=device, dtype=torch.float32)

    if ref_indices is None:
        # Fallback to full MSM refs (all images except i)
        return compute_scores_fast(Z, i, device, topmin_min, topmin_max, cdist_chunk_size=cdist_chunk_size)

    refs = ref_indices.to(device=device, dtype=torch.long)
    # Keep only valid indices.
    refs = refs[(refs >= 0) & (refs < image_num)]
    # Exclude self.
    refs = refs[refs != int(i)]

    # If empty (e.g., only self), fall back to any other image.
    if refs.numel() == 0:
        refs = torch.arange(image_num, device=device, dtype=torch.long)
        refs = refs[refs != int(i)]
        refs = refs[:1]

    # Remove duplicates (keeps deterministic behavior for 1D bank indices).
    if refs.numel() > 1:
        refs = torch.unique(refs)

    target = Z[i]  # (patch_num, c)
    Z_ref = Z.index_select(0, refs)  # (ref_num, patch_num, c)
    ref_num = int(Z_ref.shape[0])
    if ref_num <= 0:
        return torch.zeros((patch_num,), device=device, dtype=torch.float32)

    if cdist_chunk_size and int(cdist_chunk_size) > 0 and ref_num > int(cdist_chunk_size):
        patch2image = _min_cdist_chunked_refs(target, Z_ref, device, chunk_size=int(cdist_chunk_size))
        return _interval_mean(patch2image, topmin_min, topmin_max)

    dist = torch.cdist(target.unsqueeze(0), Z_ref.reshape(-1, c))
    dist = dist.squeeze(0).reshape(patch_num, ref_num, patch_num)
    patch2image = torch.min(dist, dim=2).values
    return _interval_mean(patch2image, topmin_min, topmin_max)


def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3, cdist_chunk_size=0):
    # speed fast but space large (unless using chunked cdist)
    image_num, patch_num, c = Z.shape

    if cdist_chunk_size and int(cdist_chunk_size) > 0:
        patch2image = _min_cdist_chunked(Z, i, device, chunk_size=int(cdist_chunk_size))
        return _interval_mean(patch2image, topmin_min, topmin_max)

    try:
        Z_ref = torch.cat((Z[:i], Z[i + 1 :]), dim=0)
        patch2image = torch.cdist(Z[i : i + 1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num - 1, patch_num)
        patch2image = torch.min(patch2image, -1)[0]
        return _interval_mean(patch2image, topmin_min, topmin_max)
    except torch.OutOfMemoryError:
        # OOM-safe fallback: progressively reduce cdist chunk size.
        torch.cuda.empty_cache()
        chunk = 16
        while True:
            try:
                patch2image = _min_cdist_chunked(Z, i, device, chunk_size=chunk)
                return _interval_mean(patch2image, topmin_min, topmin_max)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk <= 1:
                    raise
                chunk = max(1, chunk // 2)

def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def MSM(Z, device, topmin_min=0, topmin_max=0.3, cdist_chunk_size=0):
    image_num, patch_num, _ = Z.shape
    anomaly_scores_matrix = torch.empty((image_num, patch_num), device=device, dtype=torch.float32)
    for i in tqdm(range(image_num)):
        anomaly_scores_matrix[i] = compute_scores_fast(
            Z, i, device, topmin_min, topmin_max, cdist_chunk_size=cdist_chunk_size
        ).float()
    return anomaly_scores_matrix.double()


def MSM_with_refs(
    Z,
    device,
    topmin_min=0,
    topmin_max=0.3,
    ref_indices=None,
    cdist_chunk_size=0,
):
    """MSM with restricted reference images (transductive refinement).

    This function preserves the baseline behavior when ref_indices is None.

    Args:
        Z: (image_num, patch_num, c)
        device: torch.device
        topmin_min/topmin_max: interval mean params
        ref_indices:
            - None: use all images except self (same as MSM)
            - 1D indices: global bank indices shared across all queries
            - 2D indices: per-query retrieved indices, shape (image_num, k)
        cdist_chunk_size: if >0, chunk over reference images to reduce peak memory

    Returns:
        (image_num, patch_num) double tensor
    """
    if ref_indices is None:
        return MSM(Z, device, topmin_min=topmin_min, topmin_max=topmin_max, cdist_chunk_size=cdist_chunk_size)

    image_num, patch_num, _ = Z.shape
    refs = torch.as_tensor(ref_indices, dtype=torch.long)
    if refs.ndim not in (1, 2):
        raise ValueError(f"ref_indices must be 1D or 2D, got shape {tuple(refs.shape)}")
    refs = refs.to(device=device)

    anomaly_scores_matrix = torch.empty((image_num, patch_num), device=device, dtype=torch.float32)
    for i in tqdm(range(image_num)):
        if refs.ndim == 1:
            ref_i = refs
        else:
            ref_i = refs[i]
        anomaly_scores_matrix[i] = compute_scores_fast_with_refs(
            Z,
            i,
            device,
            ref_i,
            topmin_min=topmin_min,
            topmin_max=topmin_max,
            cdist_chunk_size=cdist_chunk_size,
        ).float()
    return anomaly_scores_matrix.double()


def MSM_with_bank(
    Z_query,
    device,
    Z_bank,
    bank_indices=None,
    topmin_min=0,
    topmin_max=0.3,
    query_chunk_size=1,
    cdist_chunk_size=0,
):
    """MSM with an external reference bank (e.g., train-split normal images).

    This is an inductive, train-free alternative to transductive mutual scoring.

    Args:
        Z_query: (N, patch_num, c)
        device: torch.device
        Z_bank: (R, patch_num, c)
        bank_indices:
            - None: use all bank images
            - 1D indices: global subset shared across all queries
            - 2D indices: per-query retrieved indices, shape (N, k)
        topmin_min/topmin_max: interval mean params
        query_chunk_size: number of query images per chunk (OOM control)
        cdist_chunk_size: number of reference images per chunk

    Returns:
        (N, patch_num) double tensor on `device`.
    """
    if Z_bank is None:
        return MSM(Z_query, device, topmin_min=topmin_min, topmin_max=topmin_max, cdist_chunk_size=cdist_chunk_size)
    if not isinstance(Z_bank, torch.Tensor) or Z_bank.numel() == 0:
        return MSM(Z_query, device, topmin_min=topmin_min, topmin_max=topmin_max, cdist_chunk_size=cdist_chunk_size)
    if Z_query is None or (isinstance(Z_query, torch.Tensor) and Z_query.numel() == 0):
        return Z_query

    Zq = torch.as_tensor(Z_query)
    bank = torch.as_tensor(Z_bank)
    if Zq.dim() != 3 or bank.dim() != 3:
        return MSM(Z_query, device, topmin_min=topmin_min, topmin_max=topmin_max, cdist_chunk_size=cdist_chunk_size)
    if int(Zq.shape[2]) != int(bank.shape[2]):
        return MSM(Z_query, device, topmin_min=topmin_min, topmin_max=topmin_max, cdist_chunk_size=cdist_chunk_size)

    # Query features live on GPU; keep the bank on CPU to avoid duplicating memory.
    Zq = Zq.to(device=device, dtype=torch.float32)
    bank_cpu_full = bank.detach().cpu()

    num_q, patch_q, c = Zq.shape
    num_ref_full, patch_r, _ = bank_cpu_full.shape
    if int(num_ref_full) <= 0:
        return MSM(Z_query, device, topmin_min=topmin_min, topmin_max=topmin_max, cdist_chunk_size=cdist_chunk_size)

    def _msm_for_bank_subset(q_tensor, bank_cpu_subset):
        num_ref = int(bank_cpu_subset.shape[0])
        if num_ref <= 0:
            return torch.zeros((int(q_tensor.shape[0]), patch_q), device=device, dtype=torch.float32)

        q_chunk = int(query_chunk_size) if query_chunk_size else 1
        q_chunk = max(1, min(q_chunk, int(q_tensor.shape[0])))

        ref_chunk = int(cdist_chunk_size) if cdist_chunk_size and int(cdist_chunk_size) > 0 else num_ref
        ref_chunk = max(1, min(ref_chunk, int(num_ref)))

        scores_local = torch.empty((int(q_tensor.shape[0]), patch_q), device=device, dtype=torch.float32)

        for qs in range(0, int(q_tensor.shape[0]), q_chunk):
            qe = min(int(q_tensor.shape[0]), qs + q_chunk)
            q = q_tensor[qs:qe]
            bq = int(q.shape[0])

            patch2image = torch.empty((bq, patch_q, num_ref), device=device, dtype=torch.float32)
            for rs in range(0, num_ref, ref_chunk):
                re = min(num_ref, rs + ref_chunk)
                refs_chunk = bank_cpu_subset[rs:re].to(device=device, dtype=torch.float32, non_blocking=True)
                br = int(re - rs)
                if br <= 0:
                    continue
                dist = torch.cdist(q.reshape(-1, c), refs_chunk.reshape(-1, c))
                dist = dist.view(bq, patch_q, br, patch_r)
                patch2image[:, :, rs:re] = dist.min(dim=3).values

            scores_local[qs:qe] = _interval_mean_batched(
                patch2image,
                topmin_min=topmin_min,
                topmin_max=topmin_max,
            )

        return scores_local

    if bank_indices is None:
        scores = _msm_for_bank_subset(Zq, bank_cpu_full)
        return scores.double()

    refs = torch.as_tensor(bank_indices, dtype=torch.long)
    if refs.ndim not in (1, 2):
        raise ValueError(f"bank_indices must be 1D or 2D, got shape {tuple(refs.shape)}")

    if refs.ndim == 1:
        ref_i = refs[(refs >= 0) & (refs < int(num_ref_full))]
        if ref_i.numel() == 0:
            scores = _msm_for_bank_subset(Zq, bank_cpu_full)
            return scores.double()
        if ref_i.numel() > 1:
            ref_i = torch.unique(ref_i)
        bank_cpu = bank_cpu_full.index_select(0, ref_i.cpu())
        scores = _msm_for_bank_subset(Zq, bank_cpu)
        return scores.double()

    if int(refs.shape[0]) != int(num_q):
        raise ValueError(
            f"bank_indices 2D expects shape (N, k) with N={num_q}, got {tuple(refs.shape)}"
        )

    # Per-query retrieved references.
    scores = torch.empty((num_q, patch_q), device=device, dtype=torch.float32)
    for i in range(int(num_q)):
        ref_i = refs[i]
        ref_i = ref_i[(ref_i >= 0) & (ref_i < int(num_ref_full))]
        if ref_i.numel() == 0:
            ref_i = torch.arange(min(1, int(num_ref_full)), dtype=torch.long)
        if ref_i.numel() > 1:
            ref_i = torch.unique(ref_i)
        bank_cpu = bank_cpu_full.index_select(0, ref_i.cpu())
        # Use query_chunk_size=1 for per-query refs.
        scores[i : i + 1] = _msm_for_bank_subset(Zq[i : i + 1], bank_cpu)

    return scores.double()

if __name__ == "__main__":
    device = 'cuda:0'
    import time
    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time-s_time)*1000)
    
