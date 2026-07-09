"""Download open-access PDFs for the DDParcel-CTAT learning bibliography."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


OPEN_PDFS = [
    {
        "id": "07_unet_2015",
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "url": "https://arxiv.org/pdf/1505.04597",
    },
    {
        "id": "08_vnet_2016",
        "title": "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation",
        "url": "https://arxiv.org/pdf/1606.04797",
    },
    {
        "id": "09_litjens_survey_2017",
        "title": "A survey on deep learning in medical image analysis",
        "url": "https://arxiv.org/pdf/1702.05747",
    },
    {
        "id": "10_densenet_2017",
        "title": "Densely Connected Convolutional Networks",
        "url": "https://arxiv.org/pdf/1608.06993",
    },
    {
        "id": "11_maxout_2013",
        "title": "Maxout Networks",
        "url": "https://arxiv.org/pdf/1302.4389",
    },
    {
        "id": "12_quicknat_2019",
        "title": "QuickNAT: A fully convolutional network for quick and accurate segmentation of neuroanatomy",
        "url": "https://arxiv.org/pdf/1801.04161",
    },
    {
        "id": "13_fastsurfer_2020",
        "title": "FastSurfer: A fast and accurate deep learning based neuroimaging pipeline",
        "url": "https://arxiv.org/pdf/1910.03866",
    },
    {
        "id": "15_attention_2017",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762",
    },
    {
        "id": "16_vit_2020",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "url": "https://arxiv.org/pdf/2010.11929",
    },
    {
        "id": "17_swin_2021",
        "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        "url": "https://arxiv.org/pdf/2103.14030",
    },
    {
        "id": "18_transunet_2021",
        "title": "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation",
        "url": "https://arxiv.org/pdf/2102.04306",
    },
    {
        "id": "19_unetr_2021",
        "title": "UNETR: Transformers for 3D Medical Image Segmentation",
        "url": "https://arxiv.org/pdf/2103.10504",
    },
    {
        "id": "20_sparsemax_2016",
        "title": "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification",
        "url": "https://arxiv.org/pdf/1602.02068",
    },
    {
        "id": "21_entmax_2019",
        "title": "Sparse Sequence-to-Sequence Models",
        "url": "https://arxiv.org/pdf/1905.05702",
    },
]

REFERENCE_ONLY = [
    {
        "id": "01_basser_dti_1994",
        "title": "MR diffusion tensor spectroscopy and imaging",
        "reason": "Publisher DOI page; no public PDF URL was assumed.",
        "link": "https://doi.org/10.1016/S0006-3495(94)80775-1",
    },
    {
        "id": "02_lebihan_dti_2001",
        "title": "Diffusion tensor imaging: concepts and applications",
        "reason": "Publisher DOI page; no public PDF URL was assumed.",
        "link": "https://doi.org/10.1002/jmri.1076",
    },
    {
        "id": "03_mori_tracking_2002",
        "title": "Fiber tracking: principles and strategies",
        "reason": "Publisher DOI page; no public PDF URL was assumed.",
        "link": "https://doi.org/10.1002/nbm.781",
    },
    {
        "id": "04_fischl_segmentation_2002",
        "title": "Whole brain segmentation: automated labeling of neuroanatomical structures",
        "reason": "Publisher DOI page; no public PDF URL was assumed.",
        "link": "https://doi.org/10.1016/S0896-6273(02)00569-X",
    },
    {
        "id": "05_fischl_parcellation_2004",
        "title": "Automatically parcellating the human cerebral cortex",
        "reason": "Publisher DOI page; no public PDF URL was assumed.",
        "link": "https://doi.org/10.1093/cercor/bhg087",
    },
    {
        "id": "06_desikan_atlas_2006",
        "title": "An automated labeling system for subdividing the human cerebral cortex",
        "reason": "Publisher DOI page; no public PDF URL was assumed.",
        "link": "https://doi.org/10.1016/j.neuroimage.2006.01.021",
    },
    {
        "id": "14_ddparcel_2024",
        "title": "DDParcel: Deep Learning Anatomical Brain Parcellation From Diffusion MRI",
        "reason": "IEEE article page; no open PDF URL was assumed.",
        "link": "https://doi.org/10.1109/TMI.2023.3331691",
    },
    {
        "id": "22_hcp",
        "title": "Human Connectome Project background",
        "reason": "Dataset/project reference rather than one required PDF.",
        "link": "https://www.humanconnectome.org/",
    },
    {
        "id": "23_cnp_ds000030",
        "title": "Consortium for Neuropsychiatric Phenomics / OpenNeuro ds000030",
        "reason": "Dataset/project reference rather than one required PDF.",
        "link": "https://openneuro.org/datasets/ds000030",
    },
    {
        "id": "24_ppmi",
        "title": "Parkinson's Progression Markers Initiative",
        "reason": "Dataset/project reference rather than one required PDF.",
        "link": "https://www.ppmi-info.org/",
    },
]


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="literature/pdfs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for paper in OPEN_PDFS:
        path = out_dir / f"{paper['id']}.pdf"
        status = "exists"
        if not path.exists() or path.stat().st_size == 0:
            try:
                download(paper["url"], path)
                status = "downloaded"
            except Exception as exc:  # noqa: BLE001
                status = f"failed: {exc}"
        results.append({**paper, "file": str(path), "status": status})
        print(f"{paper['id']}: {status}")

    manifest = {
        "downloaded_or_attempted": results,
        "reference_only": REFERENCE_ONLY,
    }
    manifest_path = out_dir.parent / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
