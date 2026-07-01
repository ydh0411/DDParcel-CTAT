# scripts/infer_ctat.py
"""Three-view CTAT inference + DDParcel-compatible post-processing."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import nibabel as nib
from models.ctat_network import CTAT
from data_loader.augmentation import ToTensorTest
from data_loader.load_neuroimaging_data import (
    OrigDataThickSlices_Fused_Input,
    map_label2aparc_aseg,
    map_prediction_sagittal2full,
)


def load_model(checkpoint_path, num_classes, device):
    model = CTAT(num_classes=num_classes, in_channels=7, num_modalities=4, embed_dim=96)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    model.set_alpha(2.0)  # Full competition at inference
    return model


def run_view_inference(model, data_list, device, batch_size=16, plane='Coronal', num_classes=82):
    """Run inference on one view, accumulate probabilities into pred_prob tensor."""
    dataset = OrigDataThickSlices_Fused_Input('ctat', data_list, plane=plane,
                                             transforms=ToTensorTest())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pred_prob = np.zeros((256, 256, 256, num_classes), dtype=np.float32)

    slice_idx = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            logits = model(images, return_aux=False)
            probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
            probs = probs.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
            for p in probs:
                if slice_idx < 256:
                    pred_prob[:, :, slice_idx, :] = p
                slice_idx += 1

    return pred_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axial_ckpt', help='Axial view checkpoint (.pkl)')
    parser.add_argument('--coronal_ckpt', help='Coronal view checkpoint (.pkl)')
    parser.add_argument('--sagittal_ckpt', help='Sagittal view checkpoint (.pkl)')
    parser.add_argument('--input_dir', required=True, help='Dir with 4 normalized .nii.gz DTI maps')
    parser.add_argument('--output', required=True, help='Output .mgz segmentation')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    # Load 4 DTI scalar maps (FA, Trace, MinEig, MidEig)
    modalities = ['FractionalAnisotropy', 'Trace', 'MinEigenvalue', 'MidEigenvalue']
    data = []
    for mod in modalities:
        img = nib.load(os.path.join(args.input_dir, f'dti-{mod}-reg-NormMasked.nii.gz'))
        data.append(img.get_fdata().astype(np.float32))
        assert data[-1].shape == (256, 256, 256), f"Expected 256^3, got {data[-1].shape}"

    pred_prob = np.zeros((256, 256, 256, 82), dtype=np.float32)
    view_weights = {'axial': 0.4, 'coronal': 0.4, 'sagittal': 0.2}

    # Axial view
    if args.axial_ckpt:
        model = load_model(args.axial_ckpt, 82, device)
        pred = run_view_inference(model, data, device, plane='Axial', num_classes=82)
        pred = np.moveaxis(pred, [1,2,0], [0,1,2])
        pred_prob += view_weights['axial'] * pred

    # Coronal view
    if args.coronal_ckpt:
        model = load_model(args.coronal_ckpt, 82, device)
        pred = run_view_inference(model, data, device, plane='Coronal', num_classes=82)
        pred_prob += view_weights['coronal'] * pred

    # Sagittal view
    if args.sagittal_ckpt:
        sag_classes = 54  # Sagittal uses 54-class mapping
        model = load_model(args.sagittal_ckpt, sag_classes, device)
        pred = run_view_inference(model, data, device, plane='Sagittal', num_classes=sag_classes)
        # Expand 54->82 classes
        pred_expanded = map_prediction_sagittal2full(pred.transpose(2, 3, 0, 1))
        pred_expanded = pred_expanded.transpose(2, 3, 0, 1)
        pred_expanded = np.moveaxis(pred_expanded, [1,2,0], [0,1,2])
        pred_prob += view_weights['sagittal'] * pred_expanded

    # Argmax and label remapping
    hard_labels = np.argmax(pred_prob, axis=-1)
    aseg = map_label2aparc_aseg(hard_labels)

    # Save as MGH
    img = nib.MGHImage(aseg.astype(np.int16), np.eye(4))
    nib.save(img, args.output)
    print(f"Saved segmentation to {args.output}")


if __name__ == '__main__':
    main()
