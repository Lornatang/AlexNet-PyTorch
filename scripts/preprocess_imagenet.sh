# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Process ImageNet_1K train dataset
# shellcheck disable=SC2164
cd ../data/ImageNet_1K/ILSVRC2012_img_train
tar -xvf ILSVRC2012_img_train.tar
rm ILSVRC2012_img_train.tar
# shellcheck disable=SC2162
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# shellcheck disable=SC2035
cd ../../scripts

# Process ImageNet_1K valid dataset
# shellcheck disable=SC2164
cd ../data/ImageNet_1K/ILSVRC2012_img_val
tar -xvf ILSVRC2012_img_val.tar
bash valprep.sh
# shellcheck disable=SC2035
rm *.JPEG
# shellcheck disable=SC2035
rm *.sh
cd ../../scripts