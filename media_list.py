#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmoputh.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum


# SET THE MEDIAS OF YOUR INTERESTS HERE !!!
class Media(Enum):
    # Liberal
    BBC = 1094
    CNN = 1095
    NYT = 1
    NPR = 1096
    WSP = 2
    HUFF = 27502
    GDN = 18629

    # Neutral
    CNBC = 1755
    USA = 4
    WSJ = 1150
    CBS = 1752
    ABC = 1091

    # Conservative
    BLZ = 232790
    RLS = 24669
    SEA = 28136
    FOX = 1092
    BRB = 19334