# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
class SeedGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.counters = dict()

    def next(self, key='default'):
        if self.seed == None:
            return None

        if key not in self.counters:
            self.counters[key] = self.seed
            return self.counters[key]
        else:
            self.counters[key] += 1
            return self.counters[key]