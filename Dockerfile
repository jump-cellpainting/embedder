# Copyright 2022 Verily Life Sciences LLC
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

# Start from the actual Docker image used to compute the embeddings in 2022 and 2023.
FROM gcr.io/terra-solutions-jump-cp-dev/embedding_creation:20221027_090451

# Add the finalized scripts
ADD scatter_wells_s3.py /opt
ADD embedding_creation.py /opt
