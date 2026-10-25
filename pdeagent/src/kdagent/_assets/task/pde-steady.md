## Task: steady PDE

The target is {target}; submit a homogeneous relation with `lhs` set to null.
Fields: {fields}. Axes: {axes}. Only eqgpt steady accepts this input kind.
Use `{{"steady": true, "steady_activation": "sin", "start_words": ["S"], "sparsity_alpha": 1.0}}`.
Do not delete boundary points or use polar evaluation. Every call, including
resume, retrains a 50000-epoch surrogate on the run's device; there is no reuse
channel.
The time limit must exceed training time to reach the search.

### Availability

{availability}

### Required settings

{required}
