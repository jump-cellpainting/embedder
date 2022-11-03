#!/bin/bash

gcloud --project terra-solutions-jump-cp-dev builds submit \
  --timeout 20m \
  --tag gcr.io/terra-solutions-jump-cp-dev/embedding_creation:`date +"%Y%m%d_%H%M%S"` .
