# Copyright 2022 Verily Life Sciences LLC
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

#!/bin/bash

gcloud --project terra-solutions-jump-cp-dev builds submit \
  --timeout 20m \
  --tag gcr.io/terra-solutions-jump-cp-dev/embedding_creation:`date +"%Y%m%d_%H%M%S"` .
