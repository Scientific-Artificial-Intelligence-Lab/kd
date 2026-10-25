## Pinned input glossary

`id` is the data-tool identifier; `task_kind` selects the task instructions;
`target` names the fitted left-hand side; `report` describes the loaded data;
`inventory` describes the source container and its arrays (`key`, `shape`,
`dtype`), or is null for a catalog input. `report.name` is the input name,
`topology` its layout, and `warnings` the inspection warnings.
`fields` lists each field or column's `name`, `dtype`, `shape`, finite-entry
`min`, `max`, `mean`, and counts `nan_count` and `inf_count`.
A null `min`, `max` or `mean` means there were no finite values.

For a table, `topology` is `tabular` and `axes` is null. `fields` includes all
feature columns and the target column; the experience feature `n_fields`
counts both. `lhs_field` and `lhs_label` name that
target; `lhs_axis` is null and `lhs_order` is zero. The three diagnostic maps
`noise_estimate`, `first_derivative_gap`, `second_derivative_gap` are empty:
no coordinate-axis diagnostic was computed.

The ODE target is a time-derivative column estimated from trajectories, and
the feature columns are the state values and any per-trajectory parameters.
When the controller estimated it from a trajectory file, `projection` records
how (`target_state`; `method` and `state_source`: kd's finite differences on
the observed states; `trim`: the points dropped at each end of every
trajectory) and `trajectories` the trajectory count and each one's point
count; the report describes the projected table.
