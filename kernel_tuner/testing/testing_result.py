import json

from kernel_tuner.util import NpEncoder
from kernel_tuner.testing.test_case import TestCase


class TestingResult():
    
    def __init__(self, test_cases: list[TestCase]) -> None:
        self.test_cases = test_cases
        self.tune_data = None
        self.tune_meta = None

    def add_tune_data_meta(self, meta, data) -> None:
        self.tune_data = data
        self.tune_meta = meta

    def toJSON(self) -> dict:
        if self.tune_data and self.tune_meta:
            tune_res = self.tune_meta
            tune_res["data"] = self.tune_data
            return tune_res + {
                "test_cases":[test_case.toJSON() for test_case in self.test_cases]
            }
        else:
            return {
                "test_cases":[test_case.toJSON() for test_case in self.test_cases]
            }
    
    def exportJSONStr(self) -> str:
        return json.dumps(self.toJSON(), cls=NpEncoder)
    