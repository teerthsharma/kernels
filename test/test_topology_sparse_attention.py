import pytest
import torch

import kernels.topology_sparse_attention as topology_sparse_attention
from kernels.topology_sparse_attention import (
    build_dense_causal_block_schedule,
    build_topology_block_schedule,
    dense_masked_attention,
    scheduled_attention,
)


def test_scheduled_attention_is_exported_from_kernels_package():
    from kernels import scheduled_attention

    assert scheduled_attention is topology_sparse_attention.scheduled_attention


def test_dense_causal_block_schedule_is_lower_triangular_csr():
    offsets, indices = build_dense_causal_block_schedule(4)

    assert offsets.tolist() == [0, 1, 3, 6, 10]
    assert indices.tolist() == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]


def test_topology_schedule_adds_persistent_block_outside_local_window():
    centroids = torch.tensor(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [101.0, 0.0],
            [102.0, 0.0],
        ]
    )
    keys = centroids.repeat_interleave(2, dim=0)

    local_offsets, local_indices = build_topology_block_schedule(
        keys,
        block_size=2,
        local_radius_blocks=0,
        sink_blocks=0,
        topk_topology_blocks=0,
    )
    topology_offsets, topology_indices = build_topology_block_schedule(
        keys,
        block_size=2,
        local_radius_blocks=0,
        sink_blocks=0,
        topk_topology_blocks=1,
    )

    assert local_indices[local_offsets[3] : local_offsets[4]].tolist() == [3]
    assert topology_indices[topology_offsets[3] : topology_offsets[4]].tolist() == [
        0,
        3,
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scheduled_attention_matches_dense_masked_reference():
    torch.manual_seed(0)
    seq = 128
    head_dim = 32
    block_size = 32
    q = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    offsets, indices = build_topology_block_schedule(
        k.float(),
        block_size=block_size,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=1,
    )

    expected = dense_masked_attention(
        q.float(), k.float(), v.float(), offsets, indices, block_size
    )
    actual = scheduled_attention(
        q,
        k,
        v,
        offsets.to(device="cuda"),
        indices.to(device="cuda"),
        block_size,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scheduled_attention_supports_batch_and_heads():
    torch.manual_seed(1)
    batch = 2
    heads = 3
    seq = 128
    head_dim = 32
    block_size = 32
    q = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
    offsets, indices = build_topology_block_schedule(
        k[0, 0].float(),
        block_size=block_size,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=1,
    )

    expected = torch.empty_like(q, dtype=torch.float32)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            expected[batch_idx, head_idx] = dense_masked_attention(
                q[batch_idx, head_idx].float(),
                k[batch_idx, head_idx].float(),
                v[batch_idx, head_idx].float(),
                offsets,
                indices,
                block_size,
            )
    actual = scheduled_attention(
        q,
        k,
        v,
        offsets.to(device="cuda"),
        indices.to(device="cuda"),
        block_size,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
