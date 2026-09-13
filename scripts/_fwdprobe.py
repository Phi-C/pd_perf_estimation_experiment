"""Record ATOM forward timing and batch metadata (rank 0 only).

Install probe_gpu on ModelRunner.run_model and probe_wall on the outermost
ModelRunner.forward. Enable before import with ATOM_FWDPROBE=/path/probe.jsonl;
ATOM_FWDPROBE_LAG defaults to 8. Disabled decorators return the original callable.

GPU events measure run_model; perf_counter measures the host call and the
interval between consecutive forward returns. Event reads wait until query()
reports completion. Tail records may need flush() after the events complete.

PyTorch is supplied by the target engine. Adapt _batch_meta() when the engine
batch schema changes. See docs/forward_probe.md for setup and interpretation.
"""

import json
import os
import time

_PATH = os.environ.get("ATOM_FWDPROBE", "")
ENABLED = bool(_PATH)

# How many forwards to let go by before reading a pair of events. Must be large
# enough that the events have certainly completed, small enough that the ring
# does not grow without bound.
DRAIN_LAG = int(os.environ.get("ATOM_FWDPROBE_LAG", "8"))

_pending = []      # [(seq, start_evt, end_evt, meta)]
_records = {}      # seq -> meta (awaiting fwd_gpu_ms)
_fh = None
_seq = 0
_prev_forward_end = None   # perf_counter of previous forward's exit


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _out():
    global _fh
    if _fh is None:
        _fh = open(_PATH, "a", buffering=1)
    return _fh


def _batch_meta(batch):
    """(c, q) per seq, in the same convention the bench harness uses:
    ctx = num_cached_tokens, q = num_scheduled_tokens."""
    m = {}
    if batch is None:
        return m
    try:
        q = [int(x) for x in batch.num_scheduled_tokens]
        m["q"] = q
        m["b"] = len(q)
        m["tok"] = int(sum(q))
        cl = getattr(batch, "context_lens", None)
        if cl is not None:
            m["ctx"] = [int(a) - int(bq) for a, bq in zip(cl, q)]
        m["prefill"] = int(getattr(batch, "total_tokens_num_prefill", 0))
        m["decode"] = int(getattr(batch, "total_tokens_num_decode", 0))
        m["dummy"] = bool(getattr(batch, "is_dummy_run", False))
    except Exception as e:  # never let the probe break a forward
        m["meta_err"] = repr(e)
    return m


def _drain(force=False):
    """Read elapsed_time for events old enough to have certainly completed."""
    import torch

    keep = []
    for seq, s, e, in_flight in _pending:
        if force or (_seq - seq) >= DRAIN_LAG:
            try:
                if e.query():
                    rec = _records.pop(seq, {})
                    rec["fwd_gpu_ms"] = s.elapsed_time(e)
                    _out().write(json.dumps(rec) + "\n")
                    continue
            except Exception:
                _records.pop(seq, None)
                continue
        keep.append((seq, s, e, in_flight))
    _pending[:] = keep


def probe_gpu(fn):
    """Wrap ModelRunner.run_model: GPU time of model forward + compute_logits."""
    if not ENABLED:
        return fn

    def wrapper(self, input_ids, batch=None, *a, **kw):
        import torch

        global _seq
        if _rank() != 0:
            return fn(self, input_ids, batch, *a, **kw)
        _seq += 1
        seq = _seq
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return fn(self, input_ids, batch, *a, **kw)
        finally:
            end.record()
            rec = _records.setdefault(seq, {})
            rec["seq"] = seq
            rec["t"] = time.time()
            rec.update(_batch_meta(batch))
            _pending.append((seq, start, end, True))
            _drain()

    wrapper.__name__ = getattr(fn, "__name__", "run_model")
    wrapper.__wrapped__ = fn
    return wrapper


def probe_wall(fn):
    """Wrap ModelRunner.forward: host wall time of one whole forward, and the
    period between consecutive forwards (= step_period when saturated)."""
    if not ENABLED:
        return fn

    def wrapper(self, batch, *a, **kw):
        global _prev_forward_end
        if _rank() != 0:
            return fn(self, batch, *a, **kw)
        t0 = time.perf_counter()
        try:
            return fn(self, batch, *a, **kw)
        finally:
            t1 = time.perf_counter()
            # Only annotate a record run_model actually created inside THIS
            # forward; a forward that never reached run_model must not attach
            # its wall time to the previous forward's record (nor create an
            # orphan that no event will ever drain).
            rec = _records.get(_seq)
            if rec is not None:
                rec["step_wall_ms"] = (t1 - t0) * 1e3
                if _prev_forward_end is not None:
                    rec["step_period_ms"] = (t1 - _prev_forward_end) * 1e3
            _prev_forward_end = t1

    wrapper.__name__ = getattr(fn, "__name__", "forward")
    wrapper.__wrapped__ = fn
    return wrapper


def flush():
    if ENABLED:
        _drain(force=True)
