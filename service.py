import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

clf = bentoml.sklearn.get("iris-demo:latest").to_runner()

service = bentoml.Service("iris-demo", runners=[clf])

# create an API Function
@service.api(input=NumpyNdarray(), output = NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    result = clf.predict.run(input_series)
    return result