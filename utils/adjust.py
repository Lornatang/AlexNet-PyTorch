# Copyright 2019 Lorna Authors. All Rights Reserved.
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

"""Dynamic adjustment of parameters tool implementation"""


def adjust_learning_rate(initial_lr=None, optimizer=None, epoch=None, every_epoch=2.4, reduction_rate=0.97):
  """Sets the learning rate to the initial LR decayed by 0.97 every 2.4 epochs"""
  if epoch != 0:
    lr = initial_lr * (reduction_rate ** (epoch // every_epoch))
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr