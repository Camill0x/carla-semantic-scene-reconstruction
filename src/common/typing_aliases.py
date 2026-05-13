from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

JsonDict = Dict[str, Any]
ObjectDict = Dict[str, object]

ArrayAny = npt.NDArray[Any]
BoolArray = npt.NDArray[np.bool_]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
FloatArray = Float64Array
ImageArray = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int_]
StrArray = npt.NDArray[np.str_]

BoundaryMetadata = Dict[str, object]
BoundaryKey = Tuple[int, int, int, object]
