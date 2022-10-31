#!/usr/bin/env python3
#
# Copyright 2022 Verily Life Sciences LLC
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file or at https://developers.google.com/open-source/licenses/bsd
#
import json
import os
import shutil

import fsspec
import numpy as np
import pandas as pd

from absl import app
from absl import flags
from google.cloud import secretmanager

# Constants
CELL_PROFILER_FILE_NAME_PREFIX = 'FileName'
CELL_PROFILER_FILE_PATH_PREFIX = 'PathName'
CELL_PROFILER_IMAGE_FILE_NAME_PREFIX = 'FileName_Orig'
CELL_PROFILER_IMAGE_FILE_PATH_PREFIX = 'PathName_Orig'
CELL_PROFILER_ILLUM_FILE_NAME_PREFIX = 'FileName_Illum'
CELL_PROFILER_ILLUM_FILE_PATH_PREFIX = 'PathName_Illum'
CELL_PROFILER_FILE_SEP = '_'

FLAGS = flags.FLAGS

flags.DEFINE_string('load_data_with_illum_csv_file', None, 'The path to the load_data_with_illum.csv file to be parsed.')
flags.DEFINE_integer('modulus', 0, 'To determine the shard for a well, perform the modulus operation over the last two digits of the well name, '
                     + 'or use the numeric value verbatim if the modulus value is zero.', lower_bound=0)
flags.DEFINE_string('output_filename', 'shards.json', 'The name of the output file with newline separated JSON metadata for each shard.')


PROJECT_ID = YOURPROJECTNUMBER


def access_secret_version(secret_id, version_id='latest'):
  # Create the Secret Manager client.
  client = secretmanager.SecretManagerServiceClient()

  # Build the resource name of the secret version.
  name = f'projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}'

  # Access the secret version.
  response = client.access_secret_version(name=name)

  # Return the decoded payload.
  return response.payload.data.decode('UTF-8')


def create_shard_metadata(load_data_with_illum_csv: str, modulus: int):
  with fsspec.open(load_data_with_illum_csv,
                   mode='rb',
                   profile='jump-cp-role') as f:
    load_data = pd.read_csv(f, dtype='str')
  print(f'Loaded {load_data.shape} from {load_data_with_illum_csv} with {load_data.columns}.')

  # Determine the shards
  if modulus > 0:
    load_data['shard'] = load_data.apply(lambda x: int(x['Metadata_Well'][-2:]) % modulus, axis = 1)
  else:
    load_data['shard'] = load_data.apply(lambda x: int(x['Metadata_Well'][-2:]), axis = 1)
  print(f'With modulus {modulus} creating {len(load_data["shard"].unique())} shards with '
        + '{len(load_data["Metadata_Well"].unique()) / len(load_data["shard"].unique())} wells each.')

  # For each well in the shard, reshape into a list of image file paths.
  shards_metadata = []
  for shard in sorted(list(load_data['shard'].unique())):
    relevant_wells = load_data[load_data['shard'] == shard]
    assert relevant_wells.shape[0] > 0, f'No rows found in load_data.csv for shard {shard}'
    print(f'Identifying files found in load_data rows matching shard {shard} and wells {relevant_wells["Metadata_Well"].unique()}.')
    shard_metadata = {}
    shard_metadata['shard'] = str(shard)

    shard_metadata['wells'] = sorted(list(relevant_wells['Metadata_Well'].unique()))
    shards_metadata.append(shard_metadata)
  return shards_metadata

def write_metadata(shards_metadata: dict, output_file: str):
  with open(output_file, 'w', encoding='utf-8') as filehandle:
    for shard_metadata in shards_metadata:
      filehandle.write(f'{json.dumps(shard_metadata)}\n')

def main(_):
  config_string = access_secret_version('USERNAME-aws-config')
  credentials_string = access_secret_version('USERNAME-aws-credentials')

  aws_dir = os.path.join(os.path.expanduser('~'), '.aws')
  if os.path.exists(aws_dir):
    shutil.rmtree(aws_dir)
  os.mkdir(aws_dir)

  with open(os.path.join(aws_dir, 'config'), 'w') as f:
    f.write(config_string)
  with open(os.path.join(aws_dir, 'credentials'), 'w') as f:
    f.write(credentials_string)

  write_metadata(
      create_shard_metadata(FLAGS.load_data_with_illum_csv_file, FLAGS.modulus),
      FLAGS.output_filename
  )

if __name__ == '__main__':
  app.run(main)
