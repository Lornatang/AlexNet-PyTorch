# Copyright 2020 Lorna Authors. All Rights Reserved.
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
from api import IMAGENET
from api import index
from django.conf.urls import url
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path

# noinspection PyInterpreter
urlpatterns = [
    url(r'^api/imagenet.html', IMAGENET.as_view(), name="AlexNet classifier IMAGENET"),
    path('', index),
    path('admin/', admin.site.urls),
    url('index/', index, name="index"),
]
urlpatterns += staticfiles_urlpatterns()
