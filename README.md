# Accuracy Assessment Tools

Requires pandas and numpy.

Input data is expected to be in a table with a column containing the mapped
values for each point and a column containing the reference value for each
point. Each row of the table should be a separate point.

Additional data such as mapped proportions or strata proportions should be
given as a dictionary whose keys match the labels used in the columns e.g.
if your class labels are 0, 1, 2, ... then the keys of the given dictionary
should also be 0, 1, 2, ...

Running each file will print test values from the corresponding paper to
verify that the math is being done properly, e.g.

`python naive_acc_assessment.py`

## Usage Example

```python
import pandas as pd
from olofsson_acc_assessment import OlofssonAccuracyAssessment

data = pd.read_csv("/path/to/file/containing/assessment/points.csv")

mapped_areas = {"forest": 200000, "deforestation": 1000}

assessment = OlofssonAccuracyAssessment(
    data, "name of map value col", "name of ref value col",
    mapped_areas)

print(assessment.overall_accuracy())
print(assessment.users_accuracy("forest"))
```
