from typing import List, Optional
from overrides import overrides

import json

from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor


@Predictor.register("glue")
class GluePredictor(Predictor):
    def __init__(self, *args, numeric: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.columns: Optional[List[str]] = None
        self.numeric = numeric

    def to_tsv_row(self, columns: List[str]) -> str:
        return "\t".join(str(value) for value in columns) + "\n"

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        output = ""
        if self.columns is None:
            self.columns = ["index", "prediction"]
            output += self.to_tsv_row(self.columns)
            output += "\t".join(self.columns) + "\n"

        if not self.numeric:
            prediction = outputs["label"]
        else:
            prediction = outputs["prediction"]
            if isinstance(prediction, float):
                prediction = min(max(prediction, 0), 5)
                prediction = f"{prediction:.3f}"

        output += self.to_tsv_row([outputs["index"], prediction])
        return output


@Predictor.register("glue-numeric")
class GlueNumericPredictor(GluePredictor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, numeric=True, **kwargs)


@Predictor.register("super_glue")
class SuperGluePredictor(Predictor):
    def __init__(self, *args, numeric: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.numeric = numeric

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:

        if not self.numeric:
            prediction = outputs["label"]
        else:
            prediction = outputs["prediction"]
            if isinstance(prediction, float):
                prediction = min(max(prediction, 0), 5)
                prediction = f"{prediction:.3f}"

        output = dict(idx=int(outputs["index"]), label=prediction)

        return json.dumps(output, ensure_ascii=False) + "\n"


@Predictor.register("super_glue-numeric")
class SuperGlueNumericPredictor(SuperGluePredictor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, numeric=True, **kwargs)


@Predictor.register("pseudolabeling")
class PseudoLabelingPredictor(SuperGluePredictor):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, numeric=True, **kwargs)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:

        if not self.numeric:
            prediction = outputs["label"]
        else:
            prediction = outputs["prediction"]
            if isinstance(prediction, float):
                prediction = min(max(prediction, 0), 5)
                prediction = f"{prediction:.3f}"

        output = {
            "idx": int(outputs["index"]),
            # "label": prediction,
            "pseudolabel": outputs["logits"],
            **outputs.get("raw_input", {})
        }

        return json.dumps(output, ensure_ascii=False) + "\n"
