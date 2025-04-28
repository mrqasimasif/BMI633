# Deep Multi Task Learning

## Data Downloaded from

1. cbmc  
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866> [downaloded]

2. goolam - E-MTAB3321
<https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3321/E-MTAB-3321.processed.1.zip>
<https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-3321>

3. klein  
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525> [tar_used]
<https://hemberg-lab.github.io/scRNA.seq.datasets/>

4. melanoma  
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72056> [one_file_used]

5. pbmc68k - 10x Genomics Datasets
<https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0>
Got Gene/Cell Matrix [filtered]
clustering analysis [cell_type]
<https://github.com/10XGenomics/single-cell-3prime-paper/blob/master/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv>

6. yan - GSE36552
<https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE36552> [not_used]
<https://www.ebi.ac.uk/gxa/sc/experiments/E-GEOD-36552/downloads> [utilized] Got Filtered TPM and ExpDesign [used]

7. Link for random datasets
<https://github.com/hemberg-lab/scRNA.seq.datasets/blob/master/utils/create_sce.R>
some useful stuff

## Architecture Overview

Input: gene expression vector (log-normalized) [~2000 genes]

Shared Encoder:
  Dense(1024) → ReLU → Dropout
  Dense(512)  → ReLU → Dropout
  Dense(256)  → ReLU

Task-specific Heads (one per dataset):
  CBMC Head:       Dense(13)  → Softmax
  Goolam Head:     Dense(5)   → Softmax
  Melanoma Head:   Dense(8)   → Softmax
  PBMC68k Head:    Dense(11)  → Softmax
  Yan Head:        Dense(7)   → Softmax
  Klein_mouse Head: Dense(4)  → Softmax
  Klein_human Head: Dense(4)  → Softmax
# BMI633
