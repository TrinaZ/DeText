# DETexT — Refactored

SNV detection at low read depths by integrating mutational signatures into TextCNN.

## Project Structure

```
detext/
├── detext/
│   ├── __init__.py
│   ├── preprocess.py      # SAM/reference parsing, differential encoding, MS lookup
│   ├── dataset.py         # PyTorch Dataset + DataLoader helpers
│   ├── model.py           # TextCNN with MS prior integration
│   └── train.py           # Training, evaluation, VCF output
├── scripts/
│   ├── prepare_data.py    # CLI: SAM → processed dataset
│   └── run_detext.py      # CLI: train + test + output VCF
├── configs/
│   └── default.yaml       # All hyperparameters in one place
└── README.md
```

## Quick Start

### 1. Prepare data
```bash
python scripts/prepare_data.py \
  --sam        data/source/1x.sam \
  --ref        data/source/hg19.fa \
  --vcf_truth  data/source/chr2.sh \
  --ms_file    data/source/signatures_probabilities.txt \
  --chrom      chr1 \
  --out        data/processed/dataset_chr1_1x.tsv
```

### 2. Train & evaluate
```bash
python scripts/run_detext.py \
  --train  data/processed/dataset_chr2_1x.tsv \
  --test   data/processed/dataset_chr1_1x.tsv \
  --model  checkpoints/model.pth \
  --vcf    results/output.vcf
```

## Dependencies

```
torch>=1.13
scikit-learn
numpy
pyyaml
```
