import os
import sys

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_ctat_imports_as_package():
    from models.ctat_network import CTAT

    assert CTAT is not None


def test_skip_tokens_convert_to_spatial_without_scrambling_layout():
    from models.ctat_decoder import CTATDecoder

    batch, modalities, height, width, channels = 1, 2, 2, 2, 3
    tokens = torch.arange(batch * modalities * height * width * channels)
    tokens = tokens.view(batch, modalities, height, width, channels)
    tokens = tokens.reshape(batch, modalities * height * width, channels)

    decoder = CTATDecoder([channels] * 5, decoder_dim=channels, num_modalities=modalities)
    actual = decoder._skip2spatial(tokens, nn.Identity(), height, width)

    expected = tokens.view(batch, modalities, height, width, channels)
    expected = expected.permute(0, 1, 4, 2, 3).reshape(batch, modalities * channels, height, width)

    assert torch.equal(actual, expected)


def test_cta_block_is_identity_when_attention_and_ffn_updates_are_zero():
    from models.cta_block import CTABlock

    block = CTABlock(dim=8, num_heads=2, alpha=1.0)
    with torch.no_grad():
        for param in block.attn.parameters():
            param.zero_()
        for param in block.ffn.mlp.parameters():
            param.zero_()

    x = torch.randn(2, 4, 8)
    y = block(x)

    assert torch.allclose(y, x, atol=1e-6)


def test_ctat_forward_shapes_with_small_model():
    from models.ctat_network import CTAT

    model = CTAT(num_classes=5, embed_dim=16, num_heads=4, depths=[1, 0, 0, 0], alpha=1.5)
    x = torch.randn(1, 28, 256, 256)

    with torch.no_grad():
        main, aux = model(x, return_aux=True)

    assert main.shape == (1, 5, 256, 256)
    assert [a.shape for a in aux] == [(1, 5, 256, 256)] * 3


def test_modality_competition_selects_modalities_at_same_patch_location():
    from models.ctat_encoder import ModalityCompetitiveFusion

    fusion = ModalityCompetitiveFusion(dim=3, num_modalities=4, alpha=2.0)
    with torch.no_grad():
        fusion.score.weight.zero_()
        fusion.score.weight[0, 0] = 1.0
        fusion.score.bias.zero_()

    tokens = torch.zeros(1, 4, 2, 3)
    tokens[:, 0, :, 0] = 5.0
    tokens[:, 1, :, 0] = 1.0
    tokens[:, 2, :, 0] = -1.0
    tokens[:, 3, :, 0] = -2.0

    gated, gate = fusion(tokens)

    assert gate.shape == (1, 4, 2, 1)
    assert torch.allclose(gate.sum(dim=1), torch.ones(1, 2, 1), atol=1e-6)
    assert torch.all(gate[:, 0] > 0.99)
    assert torch.all(gate[:, 1:] < 1e-6)
    assert torch.allclose(gated[:, 1:], torch.zeros_like(gated[:, 1:]), atol=1e-6)


def test_encoder_adds_position_embedding_and_exposes_modality_gate():
    from models.ctat_encoder import CTATEncoder

    encoder = CTATEncoder(embed_dim=8, num_heads=2, depths=[0, 0, 0, 0], alpha=2.0)
    x = torch.randn(1, 28, 256, 256)

    with torch.no_grad():
        encoder(x)

    assert encoder.pos_embed.shape == (1, 1, 64 * 64, 8)
    assert encoder.last_modality_gate.shape == (1, 4, 64 * 64, 1)
    assert torch.allclose(
        encoder.last_modality_gate.sum(dim=1),
        torch.ones(1, 64 * 64, 1),
        atol=1e-5,
    )


def test_solver_alpha_schedule_uses_number_of_batches():
    from models.ctat_network import CTAT
    from models.ctat_solver import CTATSolver

    class DummyLoader:
        def __len__(self):
            return 4

    model = CTAT(num_classes=5, embed_dim=16, num_heads=4, depths=[1, 0, 0, 0], alpha=1.0)
    solver = CTATSolver(
        model=model,
        train_loader=DummyLoader(),
        total_epochs=3,
        device='cpu',
        exp_dir='/tmp/ctat-test-solver',
    )

    assert solver.alpha_scheduler.total_steps == 12


def test_train_script_orders_hdf5_files_by_modality_semantics():
    from scripts.train_ctat import order_modality_files

    files = [
        "/tmp/subj-dti-MidEigenvalue-coronal.hdf5",
        "/tmp/subj-dti-FractionalAnisotropy-coronal.hdf5",
        "/tmp/subj-dti-Trace-coronal.hdf5",
        "/tmp/subj-dti-MinEigenvalue-coronal.hdf5",
    ]

    ordered = order_modality_files(files)

    assert ordered == [
        "/tmp/subj-dti-FractionalAnisotropy-coronal.hdf5",
        "/tmp/subj-dti-Trace-coronal.hdf5",
        "/tmp/subj-dti-MinEigenvalue-coronal.hdf5",
        "/tmp/subj-dti-MidEigenvalue-coronal.hdf5",
    ]
