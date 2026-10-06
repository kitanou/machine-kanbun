#!/bin/bash
# FLORES-200 (Meta, CC BY-SA 4.0) for Issue #39 stage 2. ~24 MB; extracted to data_ext/ (git-ignored).
cd "$(dirname "$0")/.." && mkdir -p data_ext && cd data_ext
curl -sL -o flores200.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz && tar xzf flores200.tar.gz
