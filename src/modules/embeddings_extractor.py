"""
embeddings_extractor.py
-----------------------
Extract AlphaGenome's internal embeddings.

Architecture context
--------------------
AlphaGenome.__call__ returns (predictions_dict, embeddings) where the
Embeddings dataclass holds three arrays:

  embeddings_128bp  : shape (B, 1024, 3072)
      TransformerTower trunk output passed through OutputEmbedder.
      Same representation used by the CHIP_TF and CHIP_HISTONE output heads.
      128 bp resolution over the 131 kb window.

  embeddings_1bp    : shape (B, 131072, 1536)
      SequenceDecoder output (at 1 bp resolution).  Very large — always
      aggregate before storing.  Captures fine-grained sequence features
      not present in the coarser trunk.

  embeddings_pair   : shape (B, 64, 64, 128)
      Pairwise attention (contact-map style).  Compact enough to keep full.

The standard AlphaGenomeModel._predict path discards all embeddings via
extract_predictions().  This module builds a parallel apply function that
keeps all three.

Key function
------------
build_embedding_apply_fn(metadata)
    Returns a jax.jit'd function:
        (params, state, dna_sequence, organism_index)
        -> (predictions_dict, Embeddings)

    Haiku parameter keys are fully determined by the module name hierarchy.
    AlphaGenome is always instantiated with name='alphagenome', so the
    params/state from dna_model.create() are 100% compatible — no weight
    copying or re-initialisation needed.

Convenience wrappers
--------------------
extract_embeddings_128bp(ag_model, sequence, aggregate="mean")
    -> np.ndarray  shape (3072,)  [aggregate="mean"/"max"]
                         (1024, 3072) [aggregate=None]

extract_embeddings_1bp(ag_model, sequence, aggregate="mean")
    -> np.ndarray  shape (1536,)  [aggregate="mean"/"max"]
                         (131072, 1536) [aggregate=None — very large!]

Usage example
-------------
    import sys
    sys.path.insert(0, "/beagle3/haky/data/alpha_genome/src")

    from alphagenome_research.model import dna_model
    from alphagenome.models import dna_model as dna_model_base

    model = dna_model.create(
        model_path,
        organism_settings={dna_model_base.Organism.HOMO_SAPIENS: ...},
        device=device,
    )

    from modules.embeddings_extractor import build_embedding_apply_fn
    apply_fn_emb = build_embedding_apply_fn(model._metadata)

    # For a batch of one sequence:
    import numpy as np, jax
    seq_enc  = model._one_hot_encoder.encode(sequence_str)
    seq_arr  = jax.device_put(np.asarray(seq_enc)[np.newaxis], device)
    org_arr  = jax.device_put(np.full((1,), 0, dtype=np.int32), device)
    _, emb   = apply_fn_emb(model._params, model._state, seq_arr, org_arr)
    emb_128  = np.array(jax.device_get(emb.embeddings_128bp[0]))   # (1024, 3072)
    emb_1bp  = np.array(jax.device_get(emb.embeddings_1bp[0]))     # (131072, 1536)
"""

from __future__ import annotations

import numpy as np
import jax


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------

def build_embedding_apply_fn(metadata, *, num_splice_sites: int = 512,
                              splice_site_threshold: float = 0.1):
    """
    Build a JIT-compiled apply function that returns embeddings_128bp.

    This creates a *new* Haiku transform wrapping ``AlphaGenome.__call__``.
    Because Haiku parameter names are fully determined by the module name
    hierarchy (and AlphaGenome always uses name='alphagenome'), the params /
    state loaded via ``dna_model.create()`` work without modification.

    Parameters
    ----------
    metadata : Mapping[Organism, AlphaGenomeOutputMetadata]
        The organism → metadata mapping from the loaded model
        (``model._metadata``).
    num_splice_sites : int, default 512
    splice_site_threshold : float, default 0.1

    Returns
    -------
    apply_fn_emb : callable  (jax.jit'd)
        Signature::

            apply_fn_emb(params, state, dna_sequence, organism_index)
            -> (predictions_dict, Embeddings)

        ``predictions_dict`` is the raw dict produced by AlphaGenome (same as
        what the standard apply_fn returns before ``extract_predictions``).
        ``embeddings.embeddings_128bp`` has shape ``(B, 1024, 3072)``.
    """
    import haiku as hk
    import jmp
    from alphagenome_research.model import model as ag_model_module

    jmp_policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

    @hk.transform_with_state
    def _forward_with_embeddings(dna_sequence, organism_index):
        """Same forward pass as the default, but also returns the Embeddings."""
        with hk.mixed_precision.push_policy(ag_model_module.AlphaGenome, jmp_policy):
            predictions, embeddings = ag_model_module.AlphaGenome(
                metadata,
                num_splice_sites=num_splice_sites,
                splice_site_threshold=splice_site_threshold,
            )(dna_sequence, organism_index)
        return predictions, embeddings

    def _apply(params, state, dna_sequence, organism_index):
        (predictions, embeddings), _ = _forward_with_embeddings.apply(
            params, state, None, dna_sequence, organism_index
        )
        return predictions, embeddings

    return jax.jit(_apply)


# ---------------------------------------------------------------------------
# Single-sequence convenience wrapper
# ---------------------------------------------------------------------------

def _run_embedding_forward(ag_model, sequence: str, organism_index: int = 0):
    """
    Shared helper: one-hot encode, run model, return the Embeddings dataclass.

    Results are still on-device when returned.  The caller is responsible for
    transferring needed arrays with ``jax.device_get``.

    Caches the JIT-compiled apply fn on ``ag_model._embedding_apply_fn``.
    """
    if not hasattr(ag_model, "_embedding_apply_fn"):
        ag_model._embedding_apply_fn = build_embedding_apply_fn(ag_model._metadata)

    encoder = ag_model._one_hot_encoder

    with ag_model._device_context as dev, jax.transfer_guard("disallow"):
        seq_arr = jax.device_put(
            np.asarray(encoder.encode(sequence))[np.newaxis], dev
        )
        org_arr = jax.device_put(
            np.full((1,), organism_index, dtype=np.int32), dev
        )
        _, embeddings = ag_model._embedding_apply_fn(
            ag_model._params, ag_model._state, seq_arr, org_arr
        )
    return embeddings


def _apply_aggregate(arr: np.ndarray, aggregate: str | None) -> np.ndarray:
    """Aggregate over axis 0 (spatial bins). arr is a 2-D (bins, dim) numpy array."""
    if aggregate == "mean":
        return arr.mean(axis=0).astype(np.float32)
    elif aggregate == "max":
        return arr.max(axis=0).astype(np.float32)
    elif aggregate is None:
        return arr.astype(np.float32)
    raise ValueError(f"aggregate must be 'mean', 'max', or None; got {aggregate!r}")


def extract_embeddings_128bp(
    ag_model,
    sequence: str,
    *,
    aggregate: str | None = "mean",
    organism_index: int = 0,
) -> np.ndarray:
    """
    Extract 128bp trunk embeddings for a single DNA sequence.

    The apply function is built once and cached on the model object so
    repeated calls within a session pay no extra compilation cost.

    Parameters
    ----------
    ag_model : AlphaGenomeModel
    sequence : str
        DNA sequence of length ``window_size`` (typically 131 072 bp).
    aggregate : {"mean", "max", None}, default "mean"
        Aggregation over the 1 024 spatial bins.
        "mean" / "max" → shape (3072,);  None → shape (1024, 3072).
    organism_index : int, default 0

    Returns
    -------
    np.ndarray  float32,  shape (3072,) or (1024, 3072).
    """
    embeddings = _run_embedding_forward(ag_model, sequence, organism_index)
    emb = np.array(jax.device_get(embeddings.embeddings_128bp[0]))  # (1024, 3072)
    return _apply_aggregate(emb, aggregate)


def extract_embeddings_1bp(
    ag_model,
    sequence: str,
    *,
    aggregate: str | None = "mean",
    organism_index: int = 0,
) -> np.ndarray:
    """
    Extract 1bp-resolution decoder embeddings for a single DNA sequence.

    **Memory warning**: full shape is (131072, 1536) ≈ 768 MB as float32.
    Use ``aggregate="mean"`` or ``aggregate="max"`` (default "mean") to get a
    compact (1536,) vector instead.

    Parameters
    ----------
    ag_model : AlphaGenomeModel
    sequence : str
        DNA sequence of length ``window_size`` (typically 131 072 bp).
    aggregate : {"mean", "max", None}, default "mean"
        Aggregation over the 131 072 spatial positions.
        "mean" / "max" → shape (1536,);  None → shape (131072, 1536).
    organism_index : int, default 0

    Returns
    -------
    np.ndarray  float32,  shape (1536,) or (131072, 1536).
    """
    embeddings = _run_embedding_forward(ag_model, sequence, organism_index)
    emb = np.array(jax.device_get(embeddings.embeddings_1bp[0]))  # (131072, 1536)
    return _apply_aggregate(emb, aggregate)


def extract_all_embeddings(
    ag_model,
    sequence: str,
    *,
    aggregate: str | None = "mean",
    organism_index: int = 0,
) -> dict:
    """
    Run a single forward pass and return all three embeddings.

    Returns
    -------
    dict with keys:
        "embeddings_128bp" : np.ndarray  shape (3072,) or (1024, 3072)
        "embeddings_1bp"   : np.ndarray  shape (1536,) or (131072, 1536)
        "embeddings_pair"  : np.ndarray  shape (64, 64, 128)  [always full]
    """
    embeddings = _run_embedding_forward(ag_model, sequence, organism_index)
    return {
        "embeddings_128bp": _apply_aggregate(
            np.array(jax.device_get(embeddings.embeddings_128bp[0])), aggregate
        ),
        "embeddings_1bp": _apply_aggregate(
            np.array(jax.device_get(embeddings.embeddings_1bp[0])), aggregate
        ),
        "embeddings_pair": np.array(
            jax.device_get(embeddings.embeddings_pair[0])
        ).astype(np.float32),  # (64, 64, 128) — always full
    }
