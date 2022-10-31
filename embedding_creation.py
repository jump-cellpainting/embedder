#!/usr/bin/env python
# coding: utf-8

# # Set up
# 
# **NOTE 1:** Papermill and/or Cromwell got stuck running the notebook forever with the kernel restart cell so it has been removed and the pip installs have been commented out.
# 
# **NOTE 2:** Only one cell should be tagged with 'parameters' for papermill.
# 
# **NOTE 3:** To download this file as a script that will run successfully, ensure that any cell magics (!, %, or %%) are commented out. Also ensure that `COMPUTE_VALIDATION = False` and `COMPUTE_VISUALIZATION = False`. Its a very good idea to check that the script runs successfully via the hardcoded parameter values before running it at scale. You can test it like so:
# ```
# echo after docker starts, run command: python3 /testing/embedding_creation.py
# 
# docker run -i -t --volume $HOME:/testing --entrypoint='' gcr.io/terra-solutions-jump-cp-dev/embedding_creation:20220808_223612 /bin/bash
# ```

# ## Python dependencies
# 
# For the workflow version of this, these are preinstalled in the Docker image.

# In[1]:


#%pip install --quiet apache-beam[gcp,dataframe]


# In[2]:


#%pip install --quiet fsspec[gcs,s3]


# In[3]:


# For reading in tiff images
#%pip install --quiet imagecodecs


# In[4]:


# For using API to access secrets
#%pip install --quiet google-cloud-secret-manager


# In[5]:


# Kernel restart if running as notebok.
# Need to wait a couple seconds and then run code below.
#import IPython
#app = IPython.Application.instance()
#app.kernel.do_shutdown(True)

# Future optimization: create a custom Terra Jupyter image with everything
# installed so that the above steps are unnecesary.


# In[6]:


from absl import flags


# In[7]:


# These imports will fail unless you restart the runtime after pip install.
import collections
import concurrent.futures
import os
import re
import sys

import apache_beam as beam
import apache_beam.dataframe.convert
import apache_beam.dataframe.io
from apache_beam.options import pipeline_options
import json
import pathlib
from PIL import Image as PilImage
import pandas as pd
import numpy as np
import numpy.typing as npt
import fsspec
import tensorflow as tf
import tensorflow_hub as hub
import pyarrow
import pyarrow.parquet as pq
from typing import Dict, Iterable, List, Tuple


# In[8]:


# Import for reading in actual data.
import imagecodecs
import tifffile as tiff


# ## Environment variables
# 
# For the workflow version of this, these are configured in the WDL.

# In[9]:


import os
import shutil

from google.cloud import secretmanager


PROJECT_ID = YOURPROJECTNUMBER


def access_secret_version(secret_id, version_id="latest"):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(name=name)

    # Return the decoded payload.
    return response.payload.data.decode('UTF-8')


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


# # Set parameters

# In[10]:


# Constants not set by Papermill

# Expect form of rXXcXXfXXp01-chXXsk1fk1fl1.tiff
# BROAD_FILENAME = re.compile(r'r(\d+)c(\d+)f(\d+)p01-ch(\d+)sk1fk1fl1.tiff')

# Selecting expected channel order as alphabetical for now.
# We use this mainly for creating the multi-channel site image.
# NOTE: Column order may not be preserved by parquet on read/write so 
# this may need to be reset on loading.
CHANNEL_ORDER = ['agp', 'dna', 'er', 'mito', 'rna']

# NOTE: Ignoring brightfield channels as they are inconsistently 
# collected by partners.
# Channel matching from:
# https://raw.githubusercontent.com/jump-cellpainting/2021_Chandrasekaran_submitted/main/deep_profiles/inputs/metadata/index.csv
#BROAD_CHANNEL_NUM_TO_NAME = {
#    '3': 'rna',
#    '4': 'er',
#    '2': 'agp',
#    '1': 'mito',
#    '5': 'dna',
#}
CELLPROFILER_CHANNEL_TO_NAME = {
    'agp': 'FileName_OrigAGP',
    'dna': 'FileName_OrigDNA',
    'er': 'FileName_OrigER',
    'mito': 'FileName_OrigMito',
    'rna': 'FileName_OrigRNA',
}

CELL_PROFILER_IMAGE_FILE_NAME_PREFIX = 'FileName_Orig'
CELL_PROFILER_IMAGE_FILE_PATH_PREFIX = 'PathName_Orig'
CELL_PROFILER_ILLUM_FILE_NAME_PREFIX = 'FileName_Illum'
CELL_PROFILER_ILLUM_FILE_PATH_PREFIX = 'PathName_Illum'
CELL_PROFILER_FILE_SEP = '_'


# In[11]:


# Papermill parameters. See https://papermill.readthedocs.io/en/latest/usage-parameterize.html
# To see/set tags, 'Activate the tagging toolbar by navigating to View, Cell Toolbar, and then Tags'

#---[ Inputs ]---
# TODO(mando): Remove source, batch, and plate metadata if we don't end up needing them for matching metadata
# from certain sources. For source_4, we can get these from the load_data csv.
__SOURCE_ID = 'source_4'
__BATCH_ID = '2021_05_17_Batch4'
__PLATE_ID = 'BR00123522'
__LOAD_DATA = 's3://cellpainting-gallery/jump/source_4/workspace/load_data_csv/2021_05_17_Batch4/BR00123522/load_data_with_illum.csv'
__CELL_CENTERS_PATH_PREFIX = 's3://cellpainting-gallery/jump/source_4/workspace/analysis/2021_05_17_Batch4/BR00123522/analysis'
__CELL_PATCH_DIM = 128
__TF_HUB_MODEL_PATH = (
    'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2'
)
__TF_HUB_MODEL_OUTPUT_EMB_SIZE = 1280
__TF_HUB_MODEL_INPUT_IMAGE_HEIGHT = 384
__TF_HUB_MODEL_INPUT_IMAGE_WIDTH = 384
__MODEL_BATCH_DIM = 64

__CELL_CENTERS_FILENAME = 'Nuclei.csv'
# Cell center location columns.
__CELL_CENTER_X = 'Location_Center_X'
__CELL_CENTER_Y = 'Location_Center_Y'

# See/change the sharding logic in notebook scatter_wells.ipynb.
__SHARD_METADATA = '{"shard": "1", "wells": ["A01", "B01", "C01", "D01", "E01", "F01", "G01", "H01", "I01", "J01", "K01", "L01", "M01", "N01", "O01", "P01"]}'

#---[ Outputs ]---
# This will be local and a later stage will move the output to the correct directory location.
# This helps to provide atomicity to processing.
# Since we later move the files, the output is just the current working directory.
__OUTPUT_ROOT_DIRECTORY = './'
__OUTPUT_FILENAME = 'embedding.parquet'


# In[12]:


_SOURCE_ID = flags.DEFINE_string('source_id', __SOURCE_ID, '')
_BATCH_ID = flags.DEFINE_string('batch_id', __BATCH_ID, '')
_PLATE_ID = flags.DEFINE_string('plate_id', __PLATE_ID, '')  # TODO: Remove this as it extracted from load_data csv
_LOAD_DATA = flags.DEFINE_string('load_data', __LOAD_DATA, '')
_SHARD_METADATA = flags.DEFINE_string('shard_metadata', __SHARD_METADATA, '')
_CELL_CENTERS_PATH_PREFIX = flags.DEFINE_string('cell_center_path_prefix', __CELL_CENTERS_PATH_PREFIX, '')
_CELL_PATCH_DIM = flags.DEFINE_integer('cell_patch_dim', __CELL_PATCH_DIM, '')
_CELL_CENTERS_FILENAME = flags.DEFINE_string('cell_centers_filename', __CELL_CENTERS_FILENAME, '')
_CELL_CENTER_X = flags.DEFINE_string('cell_center_x', __CELL_CENTER_X, '')
_CELL_CENTER_Y = flags.DEFINE_string('cell_center_y', __CELL_CENTER_Y, '')

_TF_HUB_MODEL_PATH = flags.DEFINE_string('tf_hub_model_path', __TF_HUB_MODEL_PATH, '')
_TF_HUB_MODEL_OUTPUT_EMB_SIZE = flags.DEFINE_integer('tf_hub_model_output_emb_size',
                                                     __TF_HUB_MODEL_OUTPUT_EMB_SIZE,
                                                     '')
_TF_HUB_MODEL_INPUT_IMAGE_HEIGHT = flags.DEFINE_integer('tf_hub_model_output_emb_height',
                                                        __TF_HUB_MODEL_INPUT_IMAGE_HEIGHT,
                                                        '')
_TF_HUB_MODEL_INPUT_IMAGE_WIDTH = flags.DEFINE_integer('tf_hub_model_output_emb_width',
                                                       __TF_HUB_MODEL_INPUT_IMAGE_WIDTH,
                                                       '')
_MODEL_BATCH_DIM = flags.DEFINE_integer('model_batch_dim', __MODEL_BATCH_DIM, '')

_OUTPUT_ROOT_DIRECTORY = flags.DEFINE_string('output_root_directory', __OUTPUT_ROOT_DIRECTORY, '')
_OUTPUT_FILENAME = flags.DEFINE_string('output_filename', __OUTPUT_FILENAME, '')


# In[13]:


# Parse flags
if 'JUPYTER_HOME' in os.environ:
    print('Using hard coded parameter values.')
    flags.FLAGS([
        'source_id',
        'batch_id',
        'plate_id',
        'shard_metadata'
        'load_data',
        'cell_center_path_prefix',
        'cell_patch_dim',
        'cell_centers_filename',
        'cell_center_x',
        'cell_center_y',
        'tf_hub_model_path',
        'tf_hub_model_output_emb_size',
        'tf_hub_model_output_emb_height',
        'tf_hub_model_output_emb_width',
        'model_batch_dim',
        'output_root_directory',
        'output_filename',
    ])
else:
    print('Parsing flags specified on the command line.')
    flags.FLAGS(sys.argv)


# In[14]:


CELL_CENTER_ROW = _CELL_CENTER_Y.value
CELL_CENTER_COL = _CELL_CENTER_X.value


# In[15]:


print(_SHARD_METADATA.value)


# In[16]:


# Parse shard metadata
shard_metadata = json.loads(_SHARD_METADATA.value)
shard_metadata


# In[17]:


def extract_image_column_names(
    load_data_df: pd.DataFrame,
    channel_order: List[str]) -> Tuple[List[str], List[str]]:
  """Find the column names for image and illum filenames and paths."""
  image_file_names = []
  image_file_paths = []
  illum_file_names = []
  illum_file_paths = []
  for col_name in load_data_df.columns:
    if col_name.startswith(CELL_PROFILER_IMAGE_FILE_NAME_PREFIX):
      image_file_names.append(col_name)
    elif col_name.startswith(CELL_PROFILER_IMAGE_FILE_PATH_PREFIX):
      image_file_paths.append(col_name)
    elif col_name.startswith(CELL_PROFILER_ILLUM_FILE_NAME_PREFIX):
      illum_file_names.append(col_name)
    elif col_name.startswith(CELL_PROFILER_ILLUM_FILE_PATH_PREFIX):
      illum_file_paths.append(col_name)
  # TODO(mando): Validate the columns.
  image_output_dict = collections.defaultdict(dict)
  for col_name in image_file_names:
    key = col_name.split(CELL_PROFILER_FILE_SEP)[1]
    image_output_dict[key]['image_filename'] = col_name
  for col_name in image_file_paths:
    key = col_name.split(CELL_PROFILER_FILE_SEP)[1]
    image_output_dict[key]['image_filepath'] = col_name
  illum_output_dict = collections.defaultdict(dict)
  for col_name in illum_file_names:
    key = col_name.split(CELL_PROFILER_FILE_SEP)[1]
    illum_output_dict[key]['illum_filename'] = col_name
  for col_name in illum_file_paths:
    key = col_name.split(CELL_PROFILER_FILE_SEP)[1]
    illum_output_dict[key]['illum_filepath'] = col_name
  # Order output by the desired channel order.
  image_channel_order = []
  illum_channel_order = []
  for channel in channel_order:
    for col_name in image_output_dict.keys():
      if col_name.lower().find(channel) > -1:
        image_channel_order.append(col_name)
    for col_name in illum_output_dict.keys():
      if col_name.lower().find(channel) > -1:
        illum_channel_order.append(col_name)
  image_output = []
  illum_output = []
  for channel in image_channel_order:
    image_output.append(image_output_dict[channel])
  for channel in illum_channel_order:
    illum_output.append(illum_output_dict[channel])
  return image_output, illum_output


def normalize_cell_center_df(
    full_cell_center_df: pd.DataFrame,
    object_number_colname: str = 'ObjectNumber',
    cell_center_x_colname: str = 'Location_Center_X',
    cell_center_y_colname: str = 'Location_Center_Y') -> pd.DataFrame:
  """Simplify and normalize cell center dataframe."""
  columns = [
      object_number_colname, cell_center_x_colname, cell_center_y_colname
  ]
  cell_center_df = full_cell_center_df[columns].copy()
  cell_center_df[cell_center_x_colname] = cell_center_df[
      cell_center_x_colname].round(decimals=0).astype(int)
  cell_center_df[cell_center_y_colname] = cell_center_df[
      cell_center_y_colname].round(decimals=0).astype(int)
  return cell_center_df


def load_metadata_files(load_data_with_illum_csv: str,
                        shard_wells: List[str],
                        channel_order: List[str],
                        cell_centers_path_prefix: str,
                        cell_centers_filename: str,
                        source: str,
                        batch: str,
                        path_transform_fn=None) -> Tuple[Dict, List[str]]:
  """Load all metadata files necessary to create all cell-level metadata."""
  with fsspec.open(os.path.join(load_data_with_illum_csv),
                   mode='rb',
                   profile='jump-cp-role') as f:
    load_data_df = pd.read_csv(f)

  image_col_names, illum_col_names = extract_image_column_names(
      load_data_df, channel_order)

  illum_filepaths = []
  for col_name_dict in illum_col_names:
    fps = list(load_data_df[col_name_dict['illum_filepath']].unique())
    fns = list(load_data_df[col_name_dict['illum_filename']].unique())
    assert len(fps) == len(fns) == 1, f'Expected 1 path and name {fps}, {fns}'
    filepath = os.path.join(fps[0], fns[0])
    if path_transform_fn:
      filepath = path_transform_fn(filepath)
    illum_filepaths.append(filepath)

  # Create per-site metadata.
  relevant_rows = load_data_df['Metadata_Well'].isin(shard_wells)
  relevant_data_df = load_data_df[relevant_rows]
  site_metadata_dict = collections.defaultdict(dict)
  for _, relevant_row_dict in relevant_data_df.iterrows():
    # Site-level:
    # Image filepaths, list ordered by channel order
    # cell center dataframe
    plate = relevant_row_dict['Metadata_Plate']
    well = relevant_row_dict['Metadata_Well']
    site = relevant_row_dict['Metadata_Site']
    site_key = (plate, well, site)
    site_dir = f'{plate}-{well}-{site}'
    print(site_dir)
    site_metadata_dict[site_key]['source'] = source
    site_metadata_dict[site_key]['batch'] = batch
    site_metadata_dict[site_key]['plate'] = plate
    site_metadata_dict[site_key]['well'] = well
    site_metadata_dict[site_key]['site'] = f'{site:02d}'
    # Add image filepaths for the site.
    filepaths = []
    for col_name_dict in image_col_names:
      filepath = os.path.join(
          relevant_row_dict[col_name_dict['image_filepath']],
          relevant_row_dict[col_name_dict['image_filename']])
      if path_transform_fn:
        filepath = path_transform_fn(filepath)
      filepaths.append(filepath)
    site_metadata_dict[site_key]['image_filepaths'] = filepaths
    # Add cell centers within site.
    cell_centers_path = os.path.join(cell_centers_path_prefix, site_dir,
                                     cell_centers_filename)
    with fsspec.open(cell_centers_path, mode='rb', profile='jump-cp-role') as f:
      site_cell_center_df = normalize_cell_center_df(pd.read_csv(f))
    site_metadata_dict[site_key]['cell_center_df'] = site_cell_center_df
  return site_metadata_dict, illum_filepaths


def load_ordered_illum_img(ordered_illum_image_filepaths):
  img_list = []
  for channel_filepath in ordered_illum_image_filepaths:
    print(channel_filepath)
    with fsspec.open(channel_filepath, mode='rb', profile='jump-cp-role') as f:
      img = np.load(f)
      # Add a channel dimension.
      img = np.expand_dims(img, -1)
      img_list.append(img)
  return np.concatenate(img_list, axis=-1)


def _read_img(channel_filepath):
  """Open an image and add a channel dimension."""
  with fsspec.open(channel_filepath, mode='rb', profile='jump-cp-role') as f:
    img = np.asarray(PilImage.open(f))
    # Add a channel dimension
    img = np.expand_dims(img, -1)
  # Returning the filepath so the images can be sorted.
  return channel_filepath, img


def parallel_load_img(elem, illum_img=None):
  """Load channel images for a site in parallel."""
  img_paths = elem['image_filepaths']
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(img_paths)) as executor:
    img_list = list(executor.map(_read_img, img_paths))
  # Ensure order matches the original channel order.
  img_list = sorted(img_list, key=lambda x: img_paths.index(x[0]))
  # Create a list of only images.
  img_list = [v[1] for v in img_list]
  img = np.concatenate(img_list, axis=-1).astype(np.float32)
  if illum_img is not None:
    img /= illum_img
  elem['image'] = img
  return elem


def extract_patches(elem, cell_patch_dim, cell_center_row, cell_center_col):
  """."""
  shared_metadata = ['source', 'batch', 'plate', 'well', 'site']
  half_patch_dim = cell_patch_dim // 2
  row_dim, col_dim, _ = tf.shape(elem['image'])
  paddings = tf.constant([[half_patch_dim, half_patch_dim],
                          [half_patch_dim, half_patch_dim], [0, 0]])
  padded_full_img = tf.pad(elem['image'], paddings, 'REFLECT')
  for _, data_row in elem['cell_center_df'].iterrows():
    cell_elem = {k: elem[k] for k in shared_metadata}
    cell_elem['cell_center_row'] = data_row[cell_center_row]
    cell_elem['cell_center_col'] = data_row[cell_center_col]
    cell_elem['region_outside_image'] = False
    if cell_elem['cell_center_row'] < half_patch_dim or cell_elem[
        'cell_center_col'] < half_patch_dim:
      cell_elem['region_outside_image'] = True
    if (row_dim - cell_elem['cell_center_row']) < half_patch_dim or (
        col_dim - cell_elem['cell_center_col']) < half_patch_dim:
      cell_elem['region_outside_image'] = True
    cell_elem['nuclei_object_number'] = data_row['ObjectNumber']
    # Add the offset for the padding to the center position.
    cell_patch_row_slice = slice(
        half_patch_dim + cell_elem['cell_center_row'] - half_patch_dim,
        half_patch_dim + cell_elem['cell_center_row'] + half_patch_dim)
    cell_patch_col_slice = slice(
        half_patch_dim + cell_elem['cell_center_col'] - half_patch_dim,
        half_patch_dim + cell_elem['cell_center_col'] + half_patch_dim)
    cell_elem['image'] = padded_full_img[cell_patch_row_slice,
                                         cell_patch_col_slice]
    yield cell_elem


def norm_img(elem):
  # NOTE: This normalizes at the site-image level and not per-patch.
  # One is not necessarily correct, but they do not yield the same results.
  image = elem['image']
  max_val = tf.math.reduce_max(image, axis=[0, 1], keepdims=True)
  min_val = tf.math.reduce_min(image, axis=[0, 1], keepdims=True)
  elem['image'] = (image - min_val) / (max_val - min_val)
  return elem


@tf.function
def make_per_channel_images(img):
  imgs = tf.transpose(img, perm=[2, 0, 1])
  imgs = tf.expand_dims(imgs, axis=-1)
  imgs = tf.image.grayscale_to_rgb(imgs)
  imgs = tf.unstack(imgs)
  return imgs


def add_embs(patch_dict_list, embs_list, channel_order):
  output = []
  for i, patch_dict in enumerate(patch_dict_list):
    for j, chan_name in enumerate(channel_order):
      emb_index = i * len(channel_order) + j
      patch_dict[f'{chan_name}_emb'] = embs_list[emb_index]
    output.append(patch_dict)
  return output


def make_output_table(patches, schema):
  output = collections.defaultdict(list)
  for patch_dict in patches:
    for name in schema.names:
      output[name].append(patch_dict[name])

  for name, parquet_type in zip(schema.names, schema.types):
    output[name] = pyarrow.array(output[name], type=parquet_type)

  return pyarrow.table(output, schema=schema)


def run_pipeline(site_metadata_dict,
                 illumination_img: npt.NDArray,
                 emb_model: tf.keras.Model,
                 channel_order: List[str],
                 model_output_emb_size: int,
                 cell_patch_dim: int,
                 cell_center_row_colname: str,
                 cell_center_col_colname: str,
                 model_batch_dim: str,
                 output_root_dir: str,
                 output_filename: str):
  """Extract patch images, computed embeddings, and save results."""
  metadata_field_schema = [
      ('source', pyarrow.string()),
      ('batch', pyarrow.string()),
      ('plate', pyarrow.string()),
      ('well', pyarrow.string()),
      ('site', pyarrow.string()),
      ('cell_center_row', pyarrow.uint16()),
      ('cell_center_col', pyarrow.uint16()),
      ('region_outside_image', pyarrow.bool_()),
      ('nuclei_object_number', pyarrow.uint16()),
  ]
  emb_fields = [f'{chan}_emb' for chan in channel_order]
  emb_pyarrow_type = pyarrow.list_(
      pyarrow.float32(), list_size=model_output_emb_size)
  schema = pyarrow.schema(metadata_field_schema +
                          [(x, emb_pyarrow_type) for x in emb_fields])

  patches = []
  for site_key, site_image_metadata in sorted(site_metadata_dict.items()):
    print(f'Processing site: {site_key}')
    imgs = norm_img(
        parallel_load_img(site_image_metadata, illum_img=illumination_img))
    patches.extend(
        list(
            extract_patches(imgs, cell_patch_dim, cell_center_row_colname,
                            cell_center_col_colname)))
    print(f'Number of input patches: {len(patches)}')

  # Make a generator function for all of the channel images.
  def _make_cell_data():
    for elem in patches:
      for channel_img in make_per_channel_images(elem['image']):
        yield channel_img

  all_imgs_batch_ds = tf.data.Dataset.from_generator(
      _make_cell_data,
      output_types=tf.float32,
      output_shapes=(cell_patch_dim, cell_patch_dim, 3),
  ).batch(
      model_batch_dim, num_parallel_calls=4, deterministic=True).prefetch(128)
  # Predict per-batch since just predict had an OOM issues. See issue elow as possibly related:
  # https://github.com/keras-team/keras/issues/13118
  patch_embs = []
  i = 0
  for b in all_imgs_batch_ds:
    if i % 10 == 0:
      print(i)
    i += 1
    patch_embs.append(emb_model.predict_on_batch(b))
  patch_embs = np.concatenate(patch_embs, axis=0)
  print('patch_embs', patch_embs.shape)
  patches_new = add_embs(patches, patch_embs, channel_order)
  print(f'Number of output patches with embs: {len(patches_new)}')
  raw_output = collections.defaultdict(list)
  for cell_dict in patches_new:
    site_dir = '-'.join([cell_dict['well'],
                         cell_dict['site']])
    raw_output[site_dir].append(cell_dict)
  for metadata_output_dir, cell_dict_list in raw_output.items():
    print(f'Writing {metadata_output_dir}')
    # Sort the cells for each site.
    cell_dict_list = sorted(cell_dict_list,
                            key=lambda x: x['nuclei_object_number'])
    patches_table = make_output_table(cell_dict_list, schema)
    # Create the output directory as well-site/output. This will be copied to
    # /source/embeddings/model_name/batch/plate/well-site/output
    output_dir = os.path.join(output_root_dir, metadata_output_dir)
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    full_output_parquet_filepath = os.path.join(output_dir, output_filename)
    with fsspec.open(full_output_parquet_filepath, mode='wb') as f:
      pq.write_table(patches_table, f)


# In[18]:


def _source4_path_transform_fn(filepath):
  return filepath.replace(
      '/home/ubuntu/bucket/projects/2021_04_26_Production',
      's3://cellpainting-gallery/jump/source_4/images')


# # Load metadata

# In[19]:


site_metadata_dict, illum_filepaths = load_metadata_files(_LOAD_DATA.value,
                              shard_metadata['wells'],
                              CHANNEL_ORDER,
                              _CELL_CENTERS_PATH_PREFIX.value,
                              _CELL_CENTERS_FILENAME.value,
                              _SOURCE_ID.value,
                              _BATCH_ID.value,
                              path_transform_fn=_source4_path_transform_fn)


# In[20]:


illumination_img = load_ordered_illum_img(illum_filepaths)


# In[21]:


emb_model = hub.KerasLayer(_TF_HUB_MODEL_PATH.value, trainable=False)


# When this notebook is run as a script, the print statement below will help us confirm its using the model cached at `/opt/hub_models/0260bc9660269daa54e7ae1ec6f4ba0b471f89bc` via Docker image `gcr.io/terra-solutions-jump-cp-dev/embedding_creation:20220808_223612`.

# In[22]:


print(hub.resolve(_TF_HUB_MODEL_PATH.value))


# In[23]:


resizing_emb_model = tf.keras.Sequential([
    tf.keras.layers.Resizing(_TF_HUB_MODEL_INPUT_IMAGE_HEIGHT.value,
                             _TF_HUB_MODEL_INPUT_IMAGE_WIDTH.value),
    emb_model,
])
resizing_emb_model.build([None, _CELL_PATCH_DIM.value, _CELL_PATCH_DIM.value, 3])


# # Run pipeline

# In[24]:


run_pipeline(site_metadata_dict,
             illumination_img,
             resizing_emb_model,
             CHANNEL_ORDER,
             model_output_emb_size=_TF_HUB_MODEL_OUTPUT_EMB_SIZE.value,
             cell_patch_dim=_CELL_PATCH_DIM.value,
             cell_center_row_colname=CELL_CENTER_ROW,
             cell_center_col_colname=CELL_CENTER_COL,
             model_batch_dim=_MODEL_BATCH_DIM.value,
             output_root_dir=_OUTPUT_ROOT_DIRECTORY.value,
             output_filename=_OUTPUT_FILENAME.value,
)


# In[25]:


os.listdir()


# # (Optional) Validate output

# In[43]:


COMPUTE_VALIDATION = False


# In[27]:


def load_data_with_illum_csv(load_data_with_illum_csv):
  with fsspec.open(os.path.join(load_data_with_illum_csv),
                   mode='rb',
                   profile='jump-cp-role') as f:
    load_data_df = pd.read_csv(f)
  return load_data_df

if COMPUTE_VALIDATION:
  load_data_df = load_data_with_illum_csv(_LOAD_DATA.value)
  print(load_data_df.head())


# In[28]:


def load_all_results(site_metadata_dict):
  output = {}
  for site_key, site_dict in site_metadata_dict.items():
    plate, well, site = site_key 
    site = f'{site:02d}'
    orig_num = site_dict['cell_center_df'].shape[0]
    filepath = f'{well}-{site}/embedding.parquet'
    processed_df = pd.read_parquet(filepath)
    output[site_key] = processed_df
  return output

if COMPUTE_VALIDATION:
  all_results_dict = load_all_results(site_metadata_dict)


# In[29]:


def show_all_results_head_and_tail(all_results_dict):
  for site_key, site_df in all_results_dict.items():
    print(site_key)
    print(site_df.head())
    print(site_df.tail())

if COMPUTE_VALIDATION:
  show_all_results_head_and_tail(all_results_dict)


# In[41]:


def validate_number_cell_centers(site_metadata_dict):
  for site_key, site_dict in site_metadata_dict.items():
    plate, well, site = site_key 
    site = f'{site:02d}'
    orig_num = site_dict['cell_center_df'].shape[0]
    filepath = f'{well}-{site}/embedding.parquet'
    processed_df = pd.read_parquet(filepath)
    processed_num = processed_df.shape[0]
    if orig_num == processed_num:
      print(site_key, orig_num, processed_num)
    else:
      print('INVALID')
      print(site_key, orig_num, processed_num)
      print('INVALID')


# In[44]:


if COMPUTE_VALIDATION:
    validate_number_cell_centers(site_metadata_dict)


# In[45]:


def validate_random_cell_embeddings(site_metadata_dict,
                                    illumination_img,
                                    cell_patch_dim,
                                    cell_center_row,
                                    cell_center_col,
                                    emb_model,
                                    channel_order):
  """."""
  for site_key, site_dict in site_metadata_dict.items():
    print(site_key)
    batch = site_dict['batch']
    plate, well, site = site_key
    site = f'{site:02d}'
    filepath = f'{well}-{site}/embedding.parquet'
    processed_df = pd.read_parquet(filepath)
    loaded_test_site_image_metadata = norm_img(parallel_load_img(
        site_dict, illum_img=illumination_img))
    cell_image_metadata = list(extract_patches(
        loaded_test_site_image_metadata,
        cell_patch_dim,
        cell_center_row,
        cell_center_col,
    ))
    random_index = np.random.choice(range(processed_df.shape[0]), size=1)[0]
    print(random_index)

    img = tf.image.resize(cell_image_metadata[random_index]['image'], (384, 384))
    batch_img = np.repeat(np.swapaxes(np.expand_dims(img, 0), 0, 3), 3, axis=3)
    embs = emb_model(batch_img)

    for i in range(len(channel_order)):
      orig_embs = processed_df[f'{channel_order[i]}_emb'].values
      emb_is_close = np.all(
          np.isclose(
              embs[i],
              orig_embs[random_index],
              atol=1e-5))
      print(channel_order[i], emb_is_close)
      if not emb_is_close:
        print(embs[i])
        print(orig_embs[random_index])


# In[46]:


if COMPUTE_VALIDATION:
    validate_random_cell_embeddings(site_metadata_dict, 
                                    illumination_img,
                                    cell_patch_dim=_CELL_PATCH_DIM.value,
                                    cell_center_row=CELL_CENTER_ROW,
                                    cell_center_col=CELL_CENTER_COL,
                                    emb_model=emb_model, 
                                    channel_order=CHANNEL_ORDER)


# # (Optional) Visualization

# In[34]:


COMPUTE_VISUALIZATION = False


# In[35]:


def show_images(img_list, img_name_list=()):
    num_imgs = len(img_list)
    if not img_name_list:
        img_name_list = [str(i) for i in range(num_imgs)]
    children = [widgets.Output() for _ in range(num_imgs)]
    tab = widgets.Tab(children=children)
    for i, img in enumerate(img_list):
        with children[i]:
            plt.show(plt.imshow(img))
        tab.set_title(i, img_name_list[i])
    display(tab)


# In[36]:


if COMPUTE_VISUALIZATION:
  import ipywidgets as widgets
  import matplotlib.pyplot as plt

  loaded_test_site_image_metadata = parallel_load_img(site_metadata_dict[('BR00123522', 'A15', 9)])
  img_list = [np.log1p(loaded_test_site_image_metadata['image'][:, :, i]) for i in range(len(CHANNEL_ORDER))]
  cell_elem_list = [v for v in extract_patches(loaded_test_site_image_metadata,
                                               _CELL_PATCH_DIM.value,
                                               CELL_CENTER_ROW,
                                               CELL_CENTER_COL)]
  cell_img_list = [e['image'] for e in cell_elem_list]


# In[37]:


if COMPUTE_VISUALIZATION:
  show_images(img_list, CHANNEL_ORDER)


# In[38]:


if COMPUTE_VISUALIZATION:
  show_images([img[:, :, 1] for img in cell_img_list[:20]])
