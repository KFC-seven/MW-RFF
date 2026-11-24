"""Build balanced cross-domain blocks for a single transmitter and a single split."""
import numpy as np


def cd_blocks(domains, m, strategy="round_robin", *, chunk_size=5):
    """Construct [block, frame, ...] from ordered arrays, one array per domain.

    round_robin takes one frame per visit; short_runs takes chunk_size;
    domain_groups takes a whole domain's share of each block in one visit.
    Regions beyond the last complete balanced block are discarded.
    """
    domains = [np.asarray(pool) for pool in domains]
    if len(domains) < 2 or any(p.ndim < 1 for p in domains):
        raise ValueError("Supply at least two domain arrays with a frame axis.")
    if any(p.shape[1:] != domains[0].shape[1:] for p in domains):
        raise ValueError("All domains need matching reference dimensions.")
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m <= 0 or m % len(domains):
        raise ValueError("m must be a positive multiple of the domain count.")
    per_domain = m // len(domains)
    if strategy not in ("round_robin", "short_runs", "domain_groups"):
        raise ValueError("Choose round_robin, short_runs, or domain_groups.")
    take = {"round_robin": 1, "short_runs": chunk_size, "domain_groups": per_domain}[strategy]
    if isinstance(take, bool) or not isinstance(take, (int, np.integer)) or take < 1 or per_domain % take:
        raise ValueError("Each domain's share must be divisible by the run length.")
    n = min(len(p) for p in domains) // per_domain
    shape = domains[0].shape[1:]
    pieces = [p[:n*per_domain].reshape(n, per_domain//take, take, *shape) for p in domains]
    return np.stack(pieces, axis=2).reshape(n, m, *shape)
